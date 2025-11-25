"""
Baseline BoTorch module for constrained Bayesian Optimization

This module provides a drop-in replacement for the RL-enhanced approach,
using qLogNoisyExpectedImprovement (qLogNEI) with batch size q=1.

Based on: https://botorch.org/docs/tutorials/closed_loop_botorch_only/

Usage in bayesian_optimization.py main():
    Replace:
        agent = train_rl_agent(dim=dim, n_episodes=n_episodes, horizon=5)
        best = run_rl_bo_on_coco(agent, problem, budget)

    With:
        agent = create_baseline_agent(dim=dim)
        best = run_baseline_bo_on_coco(agent, problem, budget)
"""

import numpy as np
import torch
from typing import Optional

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition import qLogNoisyExpectedImprovement
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
# BASELINE "AGENT" (Placeholder for interface compatibility)
# =============================================================================

class BaselineAgent:
    """
    Placeholder agent for baseline method.

    The baseline doesn't need training, but we create this object
    to maintain interface compatibility with the RL approach.
    """
    def __init__(self, dim):
        self.dim = dim
        self.method = "qLogNEI"
        self.batch_size = 1

    def __repr__(self):
        return f"BaselineAgent(dim={self.dim}, method={self.method}, q={self.batch_size})"


def create_baseline_agent(dim, **kwargs):
    """
    Create baseline agent (no training needed).

    This replaces train_rl_agent() in the main loop.

    Args:
        dim: Problem dimension
        **kwargs: Ignored (for compatibility with RL agent interface)

    Returns:
        BaselineAgent instance
    """
    print(f"Creating baseline agent for {dim}D problems (qLogNEI, q=1)")
    print("No training required for baseline method.\n")
    return BaselineAgent(dim)


# =============================================================================
# ACQUISITION FUNCTION HELPERS
# =============================================================================

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    """Extract objective from GP output (first output)"""
    return Z[..., 0]


def constraint_callable(Z):
    """Extract constraint from GP output (second output)"""
    return Z[..., 1]


# Define objective for acquisition function
objective = GenericMCObjective(objective=obj_callable)


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def initialize_model(train_x, train_obj, train_con, state_dict=None):
    """
    Initialize ModelListGP with separate GPs for objective and constraint.

    Args:
        train_x: Training inputs [n, dim]
        train_obj: Training objective values [n, 1]
        train_con: Training constraint values [n, 1]
        state_dict: Optional state dict for warm starting

    Returns:
        mll: Marginal log likelihood
        model: ModelListGP instance
    """
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

    # Load state dict if provided (for warm starting)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


# =============================================================================
# MAIN BASELINE BO FUNCTION
# =============================================================================

def run_baseline_bo_on_coco(agent, coco_problem, budget, n_initial=None):
    """
    Run baseline BO with qLogNEI on a COCO problem.

    This replaces run_rl_bo_on_coco() in the main loop.

    Args:
        agent: BaselineAgent instance (unused, for interface compatibility)
        coco_problem: COCO problem instance
        budget: Total evaluation budget
        n_initial: Number of initial random points (default: 2*D)

    Returns:
        best_feasible: Best feasible objective value found
    """
    dim = coco_problem.dimension
    lower_bounds = coco_problem.lower_bounds
    upper_bounds = coco_problem.upper_bounds

    # Default initial samples: 2*D
    n_initial = n_initial or (2 * dim)
    n_initial = min(n_initial, budget)

    # Normalize bounds to [0, 1]^D
    def normalize(x):
        """Normalize from problem bounds to [0, 1]^D"""
        return (x - lower_bounds) / (upper_bounds - lower_bounds)

    def denormalize(x):
        """Denormalize from [0, 1]^D to problem bounds"""
        return x * (upper_bounds - lower_bounds) + lower_bounds

    def evaluate_point(x_normalized):
        """Evaluate a single normalized point on COCO function"""
        if isinstance(x_normalized, torch.Tensor):
            x_np = denormalize(x_normalized.cpu().numpy())
        else:
            x_np = denormalize(x_normalized)

        # Evaluate objective
        objective_value = float(coco_problem(x_np))

        # Evaluate constraint (≤ 0 means feasible)
        if hasattr(coco_problem, 'constraint'):
            c_vals = np.asarray(coco_problem.constraint(x_np))
            constraint_value = float(np.max(c_vals))
        else:
            constraint_value = -1.0  # Always feasible

        return objective_value, constraint_value

    # Optimization bounds in normalized space
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

    # ==========================================================================
    # PHASE 1: Initial random sampling
    # ==========================================================================

    train_x = torch.rand(n_initial, dim, device=device, dtype=dtype)

    objectives = []
    constraints = []
    for i in range(n_initial):
        obj, con = evaluate_point(train_x[i])
        objectives.append(obj)
        constraints.append(con)

    train_obj = torch.tensor(objectives, device=device, dtype=dtype).unsqueeze(-1)
    train_con = torch.tensor(constraints, device=device, dtype=dtype).unsqueeze(-1)

    # Add observation noise
    train_obj_noisy = train_obj + NOISE_SE * torch.randn_like(train_obj)
    train_con_noisy = train_con + NOISE_SE * torch.randn_like(train_con)

    # Initialize models
    mll, model = initialize_model(train_x, train_obj_noisy, train_con_noisy)

    # ==========================================================================
    # PHASE 2: Bayesian Optimization loop
    # ==========================================================================

    n_bo_iterations = budget - n_initial

    for iteration in range(n_bo_iterations):
        # Fit GPs
        fit_gpytorch_mll(mll)

        # Create qLogNEI acquisition function
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        qLogNEI = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,  # Use all observations for NEI
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        # Optimize acquisition function to get next point
        candidates, _ = optimize_acqf(
            acq_function=qLogNEI,
            bounds=bounds,
            q=1,  # Batch size = 1
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        new_x = candidates.detach()

        # Evaluate new point on true function
        new_obj, new_con = evaluate_point(new_x[0])
        new_obj_tensor = torch.tensor([[new_obj]], device=device, dtype=dtype)
        new_con_tensor = torch.tensor([[new_con]], device=device, dtype=dtype)

        # Add observation noise
        new_obj_noisy = new_obj_tensor + NOISE_SE * torch.randn_like(new_obj_tensor)
        new_con_noisy = new_con_tensor + NOISE_SE * torch.randn_like(new_con_tensor)

        # Update training data
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj_tensor])
        train_con = torch.cat([train_con, new_con_tensor])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj_noisy])
        train_con_noisy = torch.cat([train_con_noisy, new_con_noisy])

        # Reinitialize model with warm start
        mll, model = initialize_model(
            train_x,
            train_obj_noisy,
            train_con_noisy,
            model.state_dict()
        )

    # ==========================================================================
    # PHASE 3: Return best feasible value
    # ==========================================================================

    # Find best feasible point (using true objective values, not noisy)
    feasible_mask = (train_con <= 0).squeeze()

    if feasible_mask.any():
        best_feasible = float(train_obj[feasible_mask].max())
    else:
        best_feasible = float('-inf')

    return best_feasible


# =============================================================================
# CONVENIENCE FUNCTION (Optional)
# =============================================================================

def run_baseline_with_history(agent, coco_problem, budget, n_initial=None):
    """
    Extended version that also returns convergence history.

    Args:
        agent: BaselineAgent instance
        coco_problem: COCO problem instance
        budget: Total evaluation budget
        n_initial: Number of initial random points

    Returns:
        best_feasible: Best feasible objective value
        history: List of best feasible values at each iteration
    """
    dim = coco_problem.dimension
    lower_bounds = coco_problem.lower_bounds
    upper_bounds = coco_problem.upper_bounds

    n_initial = n_initial or (2 * dim)
    n_initial = min(n_initial, budget)

    def normalize(x):
        return (x - lower_bounds) / (upper_bounds - lower_bounds)

    def denormalize(x):
        return x * (upper_bounds - lower_bounds) + lower_bounds

    def evaluate_point(x_normalized):
        if isinstance(x_normalized, torch.Tensor):
            x_np = denormalize(x_normalized.cpu().numpy())
        else:
            x_np = denormalize(x_normalized)

        objective_value = float(coco_problem(x_np))

        if hasattr(coco_problem, 'constraint'):
            c_vals = np.asarray(coco_problem.constraint(x_np))
            constraint_value = float(np.max(c_vals))
        else:
            constraint_value = -1.0

        return objective_value, constraint_value

    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

    # Initial sampling
    train_x = torch.rand(n_initial, dim, device=device, dtype=dtype)

    objectives = []
    constraints = []
    for i in range(n_initial):
        obj, con = evaluate_point(train_x[i])
        objectives.append(obj)
        constraints.append(con)

    train_obj = torch.tensor(objectives, device=device, dtype=dtype).unsqueeze(-1)
    train_con = torch.tensor(constraints, device=device, dtype=dtype).unsqueeze(-1)

    train_obj_noisy = train_obj + NOISE_SE * torch.randn_like(train_obj)
    train_con_noisy = train_con + NOISE_SE * torch.randn_like(train_con)

    mll, model = initialize_model(train_x, train_obj_noisy, train_con_noisy)

    # Track history
    history = []
    feasible_mask = (train_con <= 0).squeeze()
    if feasible_mask.any():
        history.append(float(train_obj[feasible_mask].max()))
    else:
        history.append(float('-inf'))

    # BO loop
    n_bo_iterations = budget - n_initial

    for iteration in range(n_bo_iterations):
        fit_gpytorch_mll(mll)

        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        qLogNEI = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        candidates, _ = optimize_acqf(
            acq_function=qLogNEI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        new_x = candidates.detach()
        new_obj, new_con = evaluate_point(new_x[0])
        new_obj_tensor = torch.tensor([[new_obj]], device=device, dtype=dtype)
        new_con_tensor = torch.tensor([[new_con]], device=device, dtype=dtype)

        new_obj_noisy = new_obj_tensor + NOISE_SE * torch.randn_like(new_obj_tensor)
        new_con_noisy = new_con_tensor + NOISE_SE * torch.randn_like(new_con_tensor)

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj_tensor])
        train_con = torch.cat([train_con, new_con_tensor])
        train_obj_noisy = torch.cat([train_obj_noisy, new_obj_noisy])
        train_con_noisy = torch.cat([train_con_noisy, new_con_noisy])

        # Update history
        feasible_mask = (train_con <= 0).squeeze()
        if feasible_mask.any():
            history.append(float(train_obj[feasible_mask].max()))
        else:
            history.append(float('-inf'))

        mll, model = initialize_model(
            train_x,
            train_obj_noisy,
            train_con_noisy,
            model.state_dict()
        )

    best_feasible = history[-1] if history else float('-inf')

    return best_feasible, history


if __name__ == "__main__":
    print("Baseline BoTorch module for constrained Bayesian Optimization")
    print("=" * 70)
    print("\nThis module provides qLogNoisyExpectedImprovement with q=1")
    print("as a drop-in replacement for the RL-enhanced approach.")
    print("\nTo use in bayesian_optimization.py:")
    print("  1. Import: from baseline_botorch import create_baseline_agent, run_baseline_bo_on_coco")
    print("  2. Replace train_rl_agent with create_baseline_agent")
    print("  3. Replace run_rl_bo_on_coco with run_baseline_bo_on_coco")
    print("\nOr add a command-line flag to switch between methods.")
