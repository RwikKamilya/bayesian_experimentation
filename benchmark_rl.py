#!/usr/bin/env python
"""
Benchmark pre-trained RL policies against qLogEI and Random Search
on the COCO bbob-constrained suite, as in the BO practical assignment.

- Problems: F2, F4, F6, F50, F52, F54
- Instances: 0, 1, 2
- Dimensions: 2, 10
- Budget: 10 * dim evals
- Repetitions: 5

Outputs: a CSV with per-evaluation best feasible value for each method.
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn

import cocoex  # pip install cocoex

# BoTorch setup (qLogEI baseline)
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
import warnings
from gpytorch.utils.warnings import NumericalWarning
from botorch.exceptions.warnings import OptimizationWarning

# Silence GP noise-floor spam
warnings.filterwarnings("ignore", category=NumericalWarning)

# Silence BoTorch's scipy-optimizer spam
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Optimization failed in `gen_candidates_scipy`.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Optimization failed on the second try.*")
warnings.filterwarnings("ignore", category=OptimizationWarning)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

FUNCTIONS = [2, 4, 6, 50, 52, 54]
INSTANCES = [1, 2, 3]
DIMENSIONS = [2, 10]  # extend to 40 if you manage RL there
REPETITIONS = 5
BUDGET_FACTOR = 10   # evals = BUDGET_FACTOR * dim
N_INIT_FACTOR = 2    # initial random design = N_INIT_FACTOR * dim

# Where your trained RL policies live
MODEL_DIR = Path("models")
MODEL_PATTERN = "coco_policy_dim{dim}.pt"

OUT_CSV = Path("results_coco_rl_vs_qlogei.csv")
OUT_CSV_SUMMARY = Path("results_coco_rl_vs_qlogei_summary.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # BoTorch prefers double

# -----------------------------------------------------------------------------
# RL POLICY NETWORK – MUST MATCH TRAINING CODE
# -----------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Must match the PolicyNetwork used in training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          action: (batch, action_dim)
          log_prob: (batch,)
        """
        h = self.shared(state)
        mean = self.mean_layer(h)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


# -----------------------------------------------------------------------------
# RL OPTIMIZER WRAPPER
# -----------------------------------------------------------------------------

class RLOptimizer:
    """
    Wraps a pre-trained policy and turns it into a 'suggest-next-point' optimizer.
    This assumes:
      - Action space during training was unconstrained, then clamped to [0, 1]^D.
      - State representation is some summary of (X, y, c, step, budget).
    """

    def __init__(self, dim: int, ckpt_path: Path):
        self.dim = dim

        # Load the full checkpoint dict saved in train_rl.py
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        # Use the state_dim that the policy was trained with
        # (train_rl.py saved this as "state_dim")
        self.state_dim = int(ckpt.get("state_dim"))

        # Build the same architecture and load only the policy weights
        self.policy = PolicyNetwork(self.state_dim, dim).to(DEVICE)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

    def _build_state(
            self,
            X_hist: np.ndarray,
            f_hist: np.ndarray,
            c_hist: np.ndarray,
            step: int,
            budget: int,
    ) -> np.ndarray:
        """
        Build the state in the same way as CocoConstrainedBOEnv._get_state
        in train_rl.py, so the RL policy sees identical features at test time.
        Note: in training, y = -f (maximization), so we negate f_hist here.
        """
        # If nothing has been observed yet, return zeros
        if len(X_hist) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        # In training: y_observed stores y = -f
        y_arr = -np.array(f_hist, dtype=np.float32)
        c_arr = np.array(c_hist, dtype=np.float32)

        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            best_feasible = float(np.max(y_arr[feasible_mask]))
        else:
            best_feasible = -np.inf

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr) + 1e-8)
        y_max = float(np.max(y_arr))

        n_feasible = int(np.sum(feasible_mask))
        feasible_ratio = float(n_feasible) / float(len(c_arr))

        last_point = X_hist[-1]

        if feasible_mask.any():
            # argmax over feasible points
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_point = X_hist[best_idx]
        else:
            best_point = np.zeros_like(last_point)

        if np.isneginf(best_feasible):
            best_feasible = 0.0

        summary = np.array(
            [
                best_feasible,
                y_mean,
                y_std,
                y_max,
                float(n_feasible),
                feasible_ratio,
                float(len(X_hist)),
                float(step),  # steps_taken in env
                float(budget),  # horizon in env (we approximate with budget)
                float(self.dim),
            ],
            dtype=np.float32,
        )

        state = np.concatenate(
            [summary, last_point.astype(np.float32), best_point.astype(np.float32)]
        ).astype(np.float32)

        # Safety: pad or truncate to self.state_dim in case of mismatch
        if len(state) < self.state_dim:
            pad = np.zeros(self.state_dim - len(state), dtype=np.float32)
            state = np.concatenate([state, pad])
        else:
            state = state[: self.state_dim]

        return state

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        """
        Returns a point in [0, 1]^dim (normalized space).
        """
        state = self._build_state(X_hist, f_hist, c_hist, step, budget)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy(state_t)
        # Map unconstrained action -> [0, 1] via sigmoid; you might have used
        # something else in training, adapt if needed:
        x_unit = torch.sigmoid(action).cpu().numpy()[0]
        x_unit = np.clip(x_unit, 0.0, 1.0)
        return x_unit


# -----------------------------------------------------------------------------
# RANDOM SEARCH BASELINE
# -----------------------------------------------------------------------------

class RandomSearchOptimizer:
    def __init__(self, dim: int):
        self.dim = dim

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        return np.random.rand(self.dim)


# -----------------------------------------------------------------------------
# qLogEI BASELINE – BoTorch, constrained via aggregated constraint
# -----------------------------------------------------------------------------

class QLogEIOptimizer:
    """
    Implements a constrained qLogEI baseline using BoTorch, with q = 1 and
    one aggregated constraint: c(x) = max_j g_j(x) from COCO.

    Works in [0,1]^dim space; we map to COCO domain in the outer loop.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.device = DEVICE
        self.dtype = DTYPE
        self.mc_samples = 128
        self.num_restarts = 5
        self.raw_samples = 128

    def _build_model(self, X_hist, f_hist, c_hist):
        # X_hist: (n, d) in [0,1]^d
        # Objective: we maximize -f (since BoTorch is maximization)
        train_x = torch.tensor(X_hist, dtype=self.dtype, device=self.device)
        train_obj = (-torch.tensor(f_hist, dtype=self.dtype, device=self.device)).unsqueeze(-1)
        train_con = torch.tensor(c_hist, dtype=self.dtype, device=self.device).unsqueeze(-1)

        # Small observation noise to make GP numerically stable
        NOISE_SE = 1e-2  # was 1e-3; a bit more noise for numerical stability
        train_yvar_obj = torch.full_like(train_obj, NOISE_SE ** 2)
        train_yvar_con = torch.full_like(train_con, NOISE_SE ** 2)

        model_obj = SingleTaskGP(train_x, train_obj, train_yvar_obj)
        model_con = SingleTaskGP(train_x, train_con, train_yvar_con)
        model = ModelListGP(model_obj, model_con)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mll)
        return model

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        model = self._build_model(X_hist, f_hist, c_hist)

        def obj_callable(Z: torch.Tensor, X: torch.Tensor = None):
            # output 0 is objective
            return Z[..., 0]

        def constraint_callable(Z: torch.Tensor):
            # output 1 is aggregated constraint; feasible if <= 0
            return Z[..., 1]

        objective = GenericMCObjective(obj_callable)

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples])
        )

        train_obj = (-torch.tensor(f_hist, dtype=self.dtype, device=self.device)).unsqueeze(-1)
        train_con = torch.tensor(c_hist, dtype=self.dtype, device=self.device).unsqueeze(-1)

        feas_mask = (train_con <= 0.0)
        if feas_mask.any():
            best_f = (train_obj * feas_mask).max()
        else:
            # No feasible points yet – just treat best_f as current max
            best_f = train_obj.max()

        acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        bounds = torch.stack(
            [
                torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                torch.ones(self.dim, dtype=self.dtype, device=self.device),
            ]
        )

        try:
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            x_unit = candidates.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"[qLogEI] optimize_acqf failed with {e}, falling back to random.")
            x_unit = np.random.rand(self.dim)

        x_unit = np.clip(x_unit, 0.0, 1.0)
        return x_unit


# -----------------------------------------------------------------------------
# COCO EVALUATION UTILS
# -----------------------------------------------------------------------------

def map_unit_to_coco(x_unit: np.ndarray, problem) -> np.ndarray:
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)
    return lb + x_unit * (ub - lb)


def eval_coco(problem, x_unit: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate COCO problem at x_unit in [0,1]^d.

    Returns:
      f: objective (scalar)
      c_agg: aggregated constraint (max over individual constraints, <= 0 => feasible)
      x_real: point in original domain
    """
    x_real = map_unit_to_coco(x_unit, problem)

    # In bbob-constrained, problem(x) returns a scalar objective value
    f_val = problem(x_real)
    f = float(f_val)

    # Constraints are provided separately via problem.constraint(x)
    if hasattr(problem, "constraint"):
        g_vals = np.asarray(problem.constraint(x_real), dtype=float)
        if g_vals.size == 0:
            # No constraints returned – treat as unconstrained
            c_agg = 0.0
        else:
            c_agg = float(g_vals.max())
    else:
        # Unconstrained problem – always feasible
        c_agg = 0.0

    return f, c_agg, x_real


def run_single_algorithm(
    problem,
    dim: int,
    algo_name: str,
    optimizer_obj,
    budget: int,
    n_init: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Run one algorithm on one COCO problem.

    Returns:
      dict with history of:
        - best_feasible_per_eval
        - f_hist
        - c_hist
    """
    X_hist = []
    f_hist = []
    c_hist = []
    best_feasible = np.inf
    best_feasible_hist = []

    # Initial random design
    for _ in range(n_init):
        x_unit = rng.random(dim)
        f, c, _ = eval_coco(problem, x_unit)
        X_hist.append(x_unit)
        f_hist.append(f)
        c_hist.append(c)
        if c <= 0.0 and f < best_feasible:
            best_feasible = f
        best_feasible_hist.append(best_feasible)

    X_hist = np.asarray(X_hist, dtype=float)
    f_hist = np.asarray(f_hist, dtype=float)
    c_hist = np.asarray(c_hist, dtype=float)

    # Sequential loop
    for step in range(n_init, budget):
        x_unit = optimizer_obj.suggest(
            X_hist, f_hist, c_hist, step=step, budget=budget
        )
        f, c, _ = eval_coco(problem, x_unit)
        X_hist = np.vstack([X_hist, x_unit])
        f_hist = np.concatenate([f_hist, [f]])
        c_hist = np.concatenate([c_hist, [c]])

        if c <= 0.0 and f < best_feasible:
            best_feasible = f
        best_feasible_hist.append(best_feasible)

    return {
        "best_feasible": np.array(best_feasible_hist, dtype=float),
        "f_hist": f_hist,
        "c_hist": c_hist,
    }


# -----------------------------------------------------------------------------
# MAIN BENCHMARK LOOP
# -----------------------------------------------------------------------------

def run_benchmarks():
    suite = cocoex.Suite("bbob-constrained", "", "")
    results_rows = []
    summary_rows = []

    for dim in DIMENSIONS:
        budget = BUDGET_FACTOR * dim
        n_init = N_INIT_FACTOR * dim
        print(f"\n=== DIM = {dim}, budget = {budget}, n_init = {n_init} ===")

        # RL model setup (you must set state_dim to what you used in training)
        # If you are unsure, print(env.observation_space.shape[0]) from your
        # training code and hard-code it here.
        # As a safe default, we approximate: 6 summary scalars + dim last_x.
        rl_ckpt_path = MODEL_DIR / MODEL_PATTERN.format(dim=dim)
        if rl_ckpt_path.exists():
            rl_available = True
            # We don’t need state_dim here; it’s loaded from the checkpoint
            rl_optimizer_template = RLOptimizer(
                dim=dim,
                ckpt_path=rl_ckpt_path,
            )
        else:
            rl_available = False
            print(f"[WARN] RL model not found for dim={dim}: {rl_ckpt_path}")

        for fid in FUNCTIONS:
            for inst in INSTANCES:
                coco_problem_id = f"bbob-constrained_f{fid}_i{inst}_d{dim}"
                print(f"\nProblem F{fid}, instance {inst}, dim {dim}: {coco_problem_id}")

                # COCO Python API: get problem by (function, dimension, instance)
                problem = suite.get_problem_by_function_dimension_instance(
                    fid, dim, inst
                )

                for rep in range(REPETITIONS):
                    print(f"  Repetition {rep+1}/{REPETITIONS}...")

                    rng = np.random.default_rng(seed=1234 + rep)

                    # --- Random Search ---
                    rs_opt = RandomSearchOptimizer(dim=dim)
                    rs_hist = run_single_algorithm(
                        problem=problem,
                        dim=dim,
                        algo_name="Random",
                        optimizer_obj=rs_opt,
                        budget=budget,
                        n_init=n_init,
                        rng=rng,
                    )

                    for t, bf in enumerate(rs_hist["best_feasible"]):
                        results_rows.append(
                            dict(
                                method="Random",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                eval=t + 1,
                                best_feasible=bf,
                            )
                        )

                    # --- qLogEI ---
                    qlogei_opt = QLogEIOptimizer(dim=dim)
                    q_hist = run_single_algorithm(
                        problem=problem,
                        dim=dim,
                        algo_name="qLogEI",
                        optimizer_obj=qlogei_opt,
                        budget=budget,
                        n_init=n_init,
                        rng=rng,
                    )

                    for t, bf in enumerate(q_hist["best_feasible"]):
                        results_rows.append(
                            dict(
                                method="qLogEI",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                eval=t + 1,
                                best_feasible=bf,
                            )
                        )

                    # --- RL (if available for this dim) ---
                    if rl_available:
                        # New optimizer per run (stateless w.r.t. history)
                        rl_opt = RLOptimizer(
                            dim=dim,
                            ckpt_path=rl_ckpt_path,
                        )
                        rl_hist = run_single_algorithm(
                            problem=problem,
                            dim=dim,
                            algo_name="RL",
                            optimizer_obj=rl_opt,
                            budget=budget,
                            n_init=n_init,
                            rng=rng,
                        )

                        for t, bf in enumerate(rl_hist["best_feasible"]):
                            results_rows.append(
                                dict(
                                    method="RL",
                                    dim=dim,
                                    function=fid,
                                    instance=inst,
                                    repetition=rep,
                                    eval=t + 1,
                                    best_feasible=bf,
                                )
                            )

                    # ---- Per-repetition summary: final best feasible for each method ----
                    rs_final = float(rs_hist["best_feasible"][-1])
                    q_final = float(q_hist["best_feasible"][-1])
                    rl_final = None

                    if rl_available:
                        rl_final = float(rl_hist["best_feasible"][-1])

                    # Print a concise comparison line
                    if rl_available:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, RL={rl_final:.3e}"
                        )
                    else:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, RL=N/A"
                        )

                    # Store summary rows (one row per method per repetition)
                    summary_rows.append(
                        dict(
                            method="Random",
                            dim=dim,
                            function=fid,
                            instance=inst,
                            repetition=rep,
                            final_best_feasible=rs_final,
                        )
                    )
                    summary_rows.append(
                        dict(
                            method="qLogEI",
                            dim=dim,
                            function=fid,
                            instance=inst,
                            repetition=rep,
                            final_best_feasible=q_final,
                        )
                    )
                    if rl_available:
                        summary_rows.append(
                            dict(
                                method="RL",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                final_best_feasible=rl_final,
                            )
                        )

                # optionally free problem resources
                problem.free()

    df = pd.DataFrame(results_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved benchmark results to: {OUT_CSV}")

    # Per-repetition summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_CSV_SUMMARY, index=False)
    print(f"Saved per-repetition summary to: {OUT_CSV_SUMMARY}")



if __name__ == "__main__":
    run_benchmarks()
