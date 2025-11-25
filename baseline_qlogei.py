import warnings
warnings.filterwarnings('ignore')

"""
Baseline: BoTorch Constrained Bayesian Optimization with qLogEI (q=1)

This is the mandatory baseline from the assignment:
- Uses qLogExpectedImprovement with batch size q=1
- Based on the BoTorch tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only/
- Adapted to run on COCO constrained BBOB benchmarks

This baseline will be compared against the RL-enhanced approach.
"""

import numpy as np
import torch
from typing import Optional
import cocoex

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Noise standard error (same as tutorial)
NOISE_SE = 0.25
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)


# =============================================================================
# COCO PROBLEM WRAPPER
# =============================================================================

class COCOConstrainedProblem:
    """Wrapper for COCO constrained BBOB problems for use with BoTorch"""

    def __init__(self, coco_problem):
        self.problem = coco_problem
        self.dim = coco_problem.dimension
        self.lower_bounds = torch.tensor(
            coco_problem.lower_bounds, device=device, dtype=dtype
        )
        self.upper_bounds = torch.tensor(
            coco_problem.upper_bounds, device=device, dtype=dtype
        )

        # Track all evaluations
        self.X_all = []
        self.y_all = []
        self.c_all = []

    def normalize(self, x):
        """Normalize point to [0, 1]^D"""
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def denormalize(self, x):
        """Denormalize point from [0, 1]^D to original bounds"""
        return x * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def evaluate(self, x_normalized):
        """
        Evaluate point on COCO function
        Returns: (objective, constraint) where constraint <= 0 means feasible
        """
        x = self.denormalize(x_normalized)

        # Handle both single points and batches
        if x.dim() == 1:
            x_np = x.cpu().numpy()
            objective = float(self.problem(x_np))

            if hasattr(self.problem, 'constraint'):
                c_vals = np.asarray(self.problem.constraint(x_np))
                constraint = float(np.max(c_vals))
            else:
                constraint = -1.0  # Always feasible if no constraint

            return objective, constraint
        else:
            # Batch evaluation
            objectives = []
            constraints = []
            for i in range(x.shape[0]):
                x_np = x[i].cpu().numpy()
                obj = float(self.problem(x_np))

                if hasattr(self.problem, 'constraint'):
                    c_vals = np.asarray(self.problem.constraint(x_np))
                    con = float(np.max(c_vals))
                else:
                    con = -1.0

                objectives.append(obj)
                constraints.append(con)

            return objectives, constraints

    def get_best_feasible(self, X, y_obj, y_con):
        """Get best feasible objective value"""
        feasible_mask = y_con <= 0
        if not feasible_mask.any():
            return float('-inf')
        return float(y_obj[feasible_mask].max())


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def generate_initial_data(problem, n=10):
    """Generate initial random data for the problem"""
    dim = problem.dim

    # Generate random points in [0, 1]^D
    train_x = torch.rand(n, dim, device=device, dtype=dtype)

    # Evaluate on real function
    objectives = []
    constraints = []
    for i in range(n):
        obj, con = problem.evaluate(train_x[i])
        objectives.append(obj)
        constraints.append(con)

    train_obj = torch.tensor(objectives, device=device, dtype=dtype).unsqueeze(-1)
    train_con = torch.tensor(constraints, device=device, dtype=dtype).unsqueeze(-1)

    # Add observation noise
    train_obj_noisy = train_obj + NOISE_SE * torch.randn_like(train_obj)
    train_con_noisy = train_con + NOISE_SE * torch.randn_like(train_con)

    # Store in problem
    problem.X_all = train_x.clone()
    problem.y_all = train_obj_noisy.clone()
    problem.c_all = train_con_noisy.clone()

    best_observed_value = problem.get_best_feasible(train_x, train_obj, train_con)

    return train_x, train_obj_noisy, train_con_noisy, best_observed_value


def initialize_model(train_x, train_obj, train_con, state_dict=None):
    """Initialize ModelListGP with separate GPs for objective and constraint"""
    # Define models for objective and constraint
    model_obj = SingleTaskGP(
        train_x,
        train_obj,
        train_yvar.expand_as(train_obj),
        input_transform=Normalize(d=train_x.shape[-1]),
    ).to(train_x)

    model_con = SingleTaskGP(
        train_x,
        train_con,
        train_yvar.expand_as(train_con),
        input_transform=Normalize(d=train_x.shape[-1]),
    ).to(train_x)

    # Combine into multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # Load state dict if provided
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


# =============================================================================
# ACQUISITION FUNCTION HELPERS
# =============================================================================

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    """Extract objective from GP output"""
    return Z[..., 0]


def constraint_callable(Z):
    """Extract constraint from GP output"""
    return Z[..., 1]


# Define objective for acquisition function
objective = GenericMCObjective(objective=obj_callable)


def optimize_acqf_and_get_observation(acq_func, problem, bounds, q=1):
    """
    Optimize acquisition function and get observation

    Args:
        acq_func: The acquisition function to optimize
        problem: COCOConstrainedProblem instance
        bounds: Optimization bounds
        q: Batch size (default 1 for baseline)

    Returns:
        new_x: New query point(s)
        new_obj: Observed objective value(s) with noise
        new_con: Observed constraint value(s) with noise
    """
    # Optimize acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Evaluate on real function
    new_x = candidates.detach()

    if q == 1:
        exact_obj, exact_con = problem.evaluate(new_x[0])
        exact_obj = torch.tensor([[exact_obj]], device=device, dtype=dtype)
        exact_con = torch.tensor([[exact_con]], device=device, dtype=dtype)
    else:
        objectives, constraints = problem.evaluate(new_x)
        exact_obj = torch.tensor(objectives, device=device, dtype=dtype).unsqueeze(-1)
        exact_con = torch.tensor(constraints, device=device, dtype=dtype).unsqueeze(-1)

    # Add observation noise
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)

    return new_x, new_obj, new_con


# =============================================================================
# MAIN BO LOOP
# =============================================================================

def run_qlogei_bo(problem, budget, n_initial=None, q=1, verbose=False):
    """
    Run Bayesian Optimization with qLogEI on a COCO problem

    Args:
        problem: COCOConstrainedProblem instance
        budget: Total evaluation budget
        n_initial: Number of initial random points (default: 2*D)
        q: Batch size for acquisition function (default: 1 for baseline)
        verbose: Print progress

    Returns:
        best_feasible: Best feasible objective value found
        history: List of best feasible values at each iteration
    """
    dim = problem.dim
    n_initial = n_initial or (2 * dim)
    n_initial = min(n_initial, budget)

    # Optimization bounds (normalized to [0, 1]^D)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

    # Generate initial data
    train_x, train_obj, train_con, best_observed = generate_initial_data(
        problem, n=n_initial
    )

    # Initialize model
    mll, model = initialize_model(train_x, train_obj, train_con)

    # Track best observed values
    best_history = [best_observed]

    # Number of BO iterations
    n_iterations = (budget - n_initial) // q

    if verbose:
        print(f"Initial random samples: {n_initial}")
        print(f"BO iterations with q={q}: {n_iterations}")
        print(f"Total budget: {budget}")
        print(f"Initial best feasible: {best_observed:.4f}\n")

    # Main BO loop
    for iteration in range(n_iterations):
        # Fit GP models
        fit_gpytorch_mll(mll)

        # Define qLogEI acquisition function
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # For best_f, use best observed feasible value
        feasible_mask = train_con <= 0
        if feasible_mask.any():
            best_f = (train_obj * feasible_mask.to(train_obj)).max()
        else:
            best_f = train_obj.min() - 1.0  # Encourage exploration if no feasible points

        qLogEI = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        # Optimize acquisition function and get new observation
        new_x, new_obj, new_con = optimize_acqf_and_get_observation(
            qLogEI, problem, bounds, q=q
        )

        # Update training data
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_con = torch.cat([train_con, new_con])

        # Update problem history
        problem.X_all = train_x.clone()
        problem.y_all = train_obj.clone()
        problem.c_all = train_con.clone()

        # Track best feasible value (use exact values without noise for fair comparison)
        exact_obj_all = []
        exact_con_all = []
        for i in range(len(train_x)):
            obj, con = problem.evaluate(train_x[i])
            exact_obj_all.append(obj)
            exact_con_all.append(con)

        exact_obj_tensor = torch.tensor(exact_obj_all, device=device, dtype=dtype)
        exact_con_tensor = torch.tensor(exact_con_all, device=device, dtype=dtype)
        best_value = problem.get_best_feasible(train_x, exact_obj_tensor, exact_con_tensor)
        best_history.append(best_value)

        # Reinitialize model with current state for faster fitting
        mll, model = initialize_model(
            train_x, train_obj, train_con, model.state_dict()
        )

        if verbose and (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}: "
                  f"Best feasible = {best_value:.4f}")

    # Final best feasible value
    final_best = best_history[-1]

    if verbose:
        print(f"\nFinal best feasible: {final_best:.4f}")

    return final_best, best_history


# =============================================================================
# BENCHMARKING ON COCO SUITE
# =============================================================================

def run_baseline_experiments(
    functions=[2, 4, 6, 50, 52, 54],
    instances=[0, 1, 2],
    dimensions=[2, 10, 40],
    repetitions=5,
    budget_multiplier=30,
    verbose=False
):
    """
    Run baseline qLogEI experiments on COCO constrained BBOB suite

    Args:
        functions: List of COCO function IDs to test
        instances: List of instance IDs (0, 1, 2 map to COCO 1, 2, 3)
        dimensions: List of dimensions to test
        repetitions: Number of repetitions per configuration
        budget_multiplier: Budget = budget_multiplier * D
        verbose: Print detailed progress

    Returns:
        results: Dictionary of results
        all_histories: Dictionary of convergence histories for plotting
    """
    results = {}
    all_histories = {}

    # Initialize COCO suite
    suite = cocoex.Suite("bbob-constrained", "", "")

    print("=" * 70)
    print("BASELINE: qLogEI with q=1 (BoTorch)")
    print("=" * 70)

    for dim in dimensions:
        print(f"\n{'=' * 70}")
        print(f"DIMENSION {dim}")
        print(f"{'=' * 70}")

        budget = budget_multiplier * dim
        print(f"Budget: {budget} evaluations per run")

        for func in functions:
            for inst in instances:
                coco_inst = inst + 1  # COCO instances start at 1

                print(f"\nFunction F{func}, Instance {inst} (COCO inst {coco_inst}), Dim {dim}")

                try:
                    coco_problem = suite.get_problem_by_function_dimension_instance(
                        func, dim, coco_inst
                    )
                except Exception as e:
                    print(f"  Could not load problem: {e}")
                    continue

                rep_results = []
                rep_histories = []

                for rep in range(repetitions):
                    problem = COCOConstrainedProblem(coco_problem)

                    best_value, history = run_qlogei_bo(
                        problem=problem,
                        budget=budget,
                        n_initial=2 * dim,
                        q=1,  # Baseline uses q=1
                        verbose=verbose
                    )

                    rep_results.append(best_value)
                    rep_histories.append(history)
                    print(f"  Rep {rep + 1}/{repetitions}: {best_value:.4f}")

                key = f"F{func}_i{inst}_d{dim}"
                results[key] = {
                    "mean": float(np.mean(rep_results)),
                    "std": float(np.std(rep_results)),
                    "all": rep_results,
                }
                all_histories[key] = rep_histories

    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    for key, val in results.items():
        print(f"{key}: {val['mean']:.4f} ± {val['std']:.4f}")

    return results, all_histories


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run qLogEI baseline on COCO benchmarks")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 40],
                        help="Dimensions to test")
    parser.add_argument("--functions", type=int, nargs="+", default=[2, 4, 6, 50, 52, 54],
                        help="COCO function IDs")
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2],
                        help="Instance IDs")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of repetitions")
    parser.add_argument("--budget", type=int, default=30,
                        help="Budget multiplier (budget = multiplier * D)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--save", type=str, default="baseline_results.npy",
                        help="File to save results")

    args = parser.parse_args()

    results, histories = run_baseline_experiments(
        functions=args.functions,
        instances=args.instances,
        dimensions=args.dimensions,
        repetitions=args.repetitions,
        budget_multiplier=args.budget,
        verbose=args.verbose
    )

    # Save results
    np.save(args.save, {"results": results, "histories": histories})
    print(f"\nResults saved to {args.save}")
