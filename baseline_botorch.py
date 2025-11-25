"""
Baseline BoTorch module for constrained Bayesian Optimization

PURE IMPLEMENTATION of qLogEI (q=1) from BoTorch tutorial
Assignment requirement: "BoTorch constrained BO with qLogEI (batch size q=1)"

Based on: https://botorch.org/docs/tutorials/closed_loop_botorch_only/

This is a drop-in replacement for the RL-enhanced approach.
"""

import numpy as np
import torch
from typing import Optional

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
# BASELINE "AGENT" (Placeholder for interface compatibility)
# =============================================================================

class BaselineAgent:
    """
    Placeholder agent for baseline method.
    The baseline doesn't need training.
    """
    def __init__(self, dim):
        self.dim = dim
        self.method = "qLogEI"
        self.batch_size = 1

    def __repr__(self):
        return f"BaselineAgent(dim={self.dim}, method={self.method}, q={self.batch_size})"


def create_baseline_agent(dim, **kwargs):
    """
    Create baseline agent (dummy).
    Replaces train_rl_agent() in main loop.
    """
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
    Initialize ModelListGP.
    """
    # define models for objective and constraint
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

    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


# =============================================================================
# MAIN BASELINE BO FUNCTION
# =============================================================================

def run_baseline_bo_on_coco(agent, coco_problem, budget, n_initial=None):
    """
    Run baseline BO with qLogEI (q=1) on COCO problem.

    Following BoTorch tutorial:
    - COCO problems instead of Hartmann6
    - q=1 instead of q=3
    """
    dim = coco_problem.dimension
    lower_bounds = coco_problem.lower_bounds
    upper_bounds = coco_problem.upper_bounds

    # Default: 2*D initial samples (same as RL method)
    n_initial = n_initial or (2 * dim)
    n_initial = min(n_initial, budget)

    # Normalization functions
    def normalize(x):
        return (x - lower_bounds) / (upper_bounds - lower_bounds)

    def denormalize(x):
        return x * (upper_bounds - lower_bounds) + lower_bounds

    def evaluate_point(x_normalized):
        """Evaluate on COCO function"""
        if isinstance(x_normalized, torch.Tensor):
            x_np = denormalize(x_normalized.cpu().numpy())
        else:
            x_np = denormalize(x_normalized)

        # Objective
        objective_value = float(coco_problem(x_np))

        # Constraint (≤ 0 means feasible)
        if hasattr(coco_problem, 'constraint'):
            c_vals = np.asarray(coco_problem.constraint(x_np))
            constraint_value = float(np.max(c_vals))
        else:
            constraint_value = -1.0

        return objective_value, constraint_value

    # Bounds in normalized space [0,1]^D
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

    # ==========================================================================
    # PHASE 1: Generate initial data  
    # ==========================================================================

    train_x = torch.rand(n_initial, dim, device=device, dtype=dtype)

    exact_obj_list = []
    exact_con_list = []
    for i in range(n_initial):
        obj, con = evaluate_point(train_x[i])
        exact_obj_list.append(obj)
        exact_con_list.append(con)

    exact_obj = torch.tensor(exact_obj_list, device=device, dtype=dtype).unsqueeze(-1)
    exact_con = torch.tensor(exact_con_list, device=device, dtype=dtype).unsqueeze(-1)

    # Add observation noise  
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)

    # Initialize model  
    mll, model = initialize_model(train_x, train_obj, train_con)

    # ==========================================================================
    # PHASE 2: BO Loop with qLogEI  
    # ==========================================================================

    n_iterations = budget - n_initial

    for iteration in range(n_iterations):
        # Fit the model  
        fit_gpytorch_mll(mll)

        # Define qLogEI acquisition function  
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # *** KEY PART: qLogEI using best_f   ***
        # For best_f, use best observed noisy values as approximation
        best_f = (train_obj * (train_con <= 0).to(train_obj)).max()

        qLogEI = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=qmc_sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        # Optimize acquisition function  
        candidates, _ = optimize_acqf(
            acq_function=qLogEI,
            bounds=bounds,
            q=1,  # *** Batch size = 1 as required ***
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # Get new observation (adapted for COCO)
        new_x = candidates.detach()
        new_obj, new_con = evaluate_point(new_x[0])

        exact_obj_new = torch.tensor([[new_obj]], device=device, dtype=dtype)
        exact_con_new = torch.tensor([[new_con]], device=device, dtype=dtype)

        # Add observation noise  
        new_obj_noisy = exact_obj_new + NOISE_SE * torch.randn_like(exact_obj_new)
        new_con_noisy = exact_con_new + NOISE_SE * torch.randn_like(exact_con_new)

        # Update training points  
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj_noisy])
        train_con = torch.cat([train_con, new_con_noisy])

        # Keep track of exact values for final evaluation
        exact_obj = torch.cat([exact_obj, exact_obj_new])
        exact_con = torch.cat([exact_con, exact_con_new])

        # Reinitialize model (PURE from tutorial - warm starting)
        mll, model = initialize_model(
            train_x,
            train_obj,
            train_con,
            model.state_dict(),
        )

    # ==========================================================================
    # PHASE 3: Return best feasible value
    # ==========================================================================

    # Use exact values (without noise) for final best
    feasible_mask = (exact_con <= 0).squeeze()

    if feasible_mask.any():
        best_feasible = float(exact_obj[feasible_mask].max())
    else:
        best_feasible = float('-inf')

    return best_feasible


if __name__ == "__main__":
    print("=" * 70)
    print("PURE qLogEI Baseline (q=1) - BoTorch Tutorial Implementation")
    print("=" * 70)
    print("\nThis is the EXACT qLogEI implementation from the BoTorch tutorial,")
    print("adapted ONLY for:")
    print("  - COCO problems (instead of Hartmann6)")
    print("  - q=1 batch size (instead of q=3)")
    print("  - Interface compatibility with RL method")
    print("\nUsage in bayesian_optimization.py:")
    print("  python bayesian_optimization.py --baseline")
