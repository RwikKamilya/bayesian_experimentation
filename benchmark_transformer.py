#!/usr/bin/env python
"""
Benchmark pre-trained Transformer next-config policy against qLogEI and Random Search
on the COCO bbob-constrained suite, as in the BO practical assignment.

- Problems: F2, F4, F6, F50, F52, F54
- Instances: 1, 2, 3
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
DIMENSIONS = [2, 10]  # extend to 40 if you manage Transformer there
REPETITIONS = 5
BUDGET_FACTOR = 10   # evals = BUDGET_FACTOR * dim
N_INIT_FACTOR = 2    # initial random design = N_INIT_FACTOR * dim

# Where your trained Transformer models live
# Adjust if you saved them elsewhere
TRANSFORMER_MODEL_DIR = Path(".")
TRANSFORMER_MODEL_PATTERN = "next_config_transformer_dim{dim}.pt"

OUT_CSV = Path("results_coco_transformer_vs_qlogei.csv")
OUT_CSV_SUMMARY = Path("results_coco_transformer_vs_qlogei_summary.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # BoTorch prefers double for qLogEI


# -----------------------------------------------------------------------------
# TRANSFORMER MODEL – MUST MATCH TRAINING CODE
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class NextConfigTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        max_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        dim:       dimension of config space
        max_len:   maximum episode length (T_max used in training)
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.token_dim = dim + 2  # x (dim) + y + c

        self.input_proj = nn.Linear(self.token_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, dim),
        )

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens:    [batch, max_len, token_dim]
        attn_mask: [batch, max_len] (1 = real, 0 = pad)

        Returns:
            pred_x: [batch, dim]  (next config in normalized [0,1]^dim)
        """
        B, L, _ = tokens.shape
        x = self.input_proj(tokens)          # [B, L, d_model]
        x = self.pos_enc(x)                  # [B, L, d_model]

        # Transformer expects [L, B, d_model]
        x = x.transpose(0, 1)                # [L, B, d_model]

        # key_padding_mask: True at PAD positions
        key_padding_mask = (attn_mask == 0)  # [B, L]
        enc_out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [L, B, d_model]

        enc_out = enc_out.transpose(0, 1)    # [B, L, d_model]

        # get last *real* token hidden state for each batch element
        lengths = attn_mask.sum(dim=1).long() - 1  # [B]
        lengths = torch.clamp(lengths, min=0)

        idx = lengths.view(B, 1, 1).expand(-1, 1, enc_out.size(-1))  # [B, 1, d_model]
        last_hidden = enc_out.gather(1, idx).squeeze(1)  # [B, d_model]

        pred_x = self.head(last_hidden)  # [B, dim]
        return pred_x


# -----------------------------------------------------------------------------
# TRANSFORMER OPTIMIZER WRAPPER
# -----------------------------------------------------------------------------

class TransformerOptimizer:
    """
    Wraps a pre-trained NextConfigTransformer and turns it into a
    'suggest-next-point' optimizer.

    At each BO step, we feed the history sequence:
        (x_0, y_0, c_0), ..., (x_{t-1}, y_{t-1}, c_{t-1})
    and the model predicts x_t.
    """

    def __init__(self, dim: int, ckpt_path: Path):
        self.dim = dim
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.T_max = int(ckpt["T_max"])
        ckpt_dim = int(ckpt["dim"])
        if ckpt_dim != dim:
            raise ValueError(f"Checkpoint dim={ckpt_dim} does not match requested dim={dim}")

        self.model = NextConfigTransformer(dim=dim, max_len=self.T_max)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

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
        If no history yet, fall back to random.
        """
        n = len(X_hist)
        if n == 0:
            return np.random.rand(self.dim)

        # Use only the last T_max points
        L = min(n, self.T_max)
        X_seq = np.asarray(X_hist[-L:], dtype=np.float32)   # [L, dim]
        y_seq = np.asarray(f_hist[-L:], dtype=np.float32)   # [L]
        c_seq = np.asarray(c_hist[-L:], dtype=np.float32)   # [L]

        # Build tokens: [L, dim+2] then pad to [T_max, dim+2]
        tokens = np.concatenate(
            [X_seq, y_seq[:, None], c_seq[:, None]],
            axis=-1
        )  # [L, dim+2]

        token_dim = tokens.shape[-1]
        if L < self.T_max:
            pad = np.zeros((self.T_max - L, token_dim), dtype=np.float32)
            tokens = np.vstack([tokens, pad])  # [T_max, dim+2]

        attn_mask = np.zeros((self.T_max,), dtype=np.float32)
        attn_mask[:L] = 1.0

        tokens_t = torch.tensor(tokens, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, T_max, token_dim]
        attn_mask_t = torch.tensor(attn_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, T_max]

        with torch.no_grad():
            pred_x = self.model(tokens_t, attn_mask_t)  # [1, dim]

        x_unit = pred_x.squeeze(0).cpu().numpy()
        # Model was trained on normalized configs in [0,1], but to be safe:
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

        # Transformer model setup
        tr_ckpt_path = TRANSFORMER_MODEL_DIR / TRANSFORMER_MODEL_PATTERN.format(dim=dim)
        if tr_ckpt_path.exists():
            transformer_available = True
            print(f"[INFO] Found Transformer model for dim={dim}: {tr_ckpt_path}")
        else:
            transformer_available = False
            print(f"[WARN] Transformer model not found for dim={dim}: {tr_ckpt_path}")

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

                    # --- Transformer (if available for this dim) ---
                    tr_final = None
                    if transformer_available:
                        tr_opt = TransformerOptimizer(
                            dim=dim,
                            ckpt_path=tr_ckpt_path,
                        )
                        tr_hist = run_single_algorithm(
                            problem=problem,
                            dim=dim,
                            algo_name="Transformer",
                            optimizer_obj=tr_opt,
                            budget=budget,
                            n_init=n_init,
                            rng=rng,
                        )

                        for t, bf in enumerate(tr_hist["best_feasible"]):
                            results_rows.append(
                                dict(
                                    method="Transformer",
                                    dim=dim,
                                    function=fid,
                                    instance=inst,
                                    repetition=rep,
                                    eval=t + 1,
                                    best_feasible=bf,
                                )
                            )
                        tr_final = float(tr_hist["best_feasible"][-1])

                    # ---- Per-repetition summary: final best feasible for each method ----
                    rs_final = float(rs_hist["best_feasible"][-1])
                    q_final = float(q_hist["best_feasible"][-1])

                    # Print a concise comparison line
                    if transformer_available:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, Transformer={tr_final:.3e}"
                        )
                    else:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, Transformer=N/A"
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
                    if transformer_available:
                        summary_rows.append(
                            dict(
                                method="Transformer",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                final_best_feasible=tr_final,
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
