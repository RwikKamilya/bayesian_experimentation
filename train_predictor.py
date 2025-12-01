"""
train_learned_acquisition.py

Meta-learned acquisition / policy for constrained BO on COCO problems.

- Teacher: constrained Expected Improvement (cEI) from GP surrogates
- Student: deep neural net that maps BO state + candidate features -> scalar score
- Training: offline, across many COCO BBOB-constrained problems

Requires:
    pip install numpy torch scikit-learn scipy cocoex
"""

import math
import random
import time
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import torch
import torch.nn as nn
import torch.optim as optim

import cocoex  # COCO experimentation tool

import warnings
from sklearn.exceptions import ConvergenceWarning

# Silence GP optimizer / kernel convergence warnings
# Silence ALL sklearn convergence warnings globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =============================================================================
# Utility: evaluate COCO constrained problem
# =============================================================================

def parse_coco_ids(problem):
    """
    Parse function id, instance id, and dimension from problem.id.

    Example id: 'bbob-constrained_f050_i07_d02'
    Returns:
        fid (int), iid (int), dim (int)
    """
    pid = problem.id  # e.g. 'bbob-constrained_f050_i07_d02'
    parts = pid.split('_')  # ['bbob-constrained', 'f050', 'i07', 'd02']
    f_str = parts[1]  # 'f050'
    i_str = parts[2]  # 'i07'
    d_str = parts[3]  # 'd02'
    fid = int(f_str[1:])
    iid = int(i_str[1:])
    dim = int(d_str[1:])
    return fid, iid, dim


def evaluate_coco_problem(problem, x_normalized: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate a COCO constrained problem at normalized x in [0,1]^D.

    Returns:
        f (float): objective value
        c (float): scalar constraint (<=0 means feasible)
    """
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)

    x = lb + x_normalized * (ub - lb)

    f_val = float(problem(x))

    # Proper COCO constrained API
    if getattr(problem, "number_of_constraints", 0) > 0:
        c_vals = np.asarray(problem.constraint(x), dtype=float)
        c_val = float(np.max(c_vals))  # aggregate to single constraint
    else:
        # Treat as unconstrained
        c_val = -1.0

    return f_val, c_val


# =============================================================================
# Teacher acquisition: constrained Expected Improvement (cEI)
# =============================================================================

def compute_p_feas(mu_c: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
    """Probability of feasibility for scalar constraint c(x) <= 0."""
    sigma_c = np.maximum(sigma_c, 1e-12)
    z = (0.0 - mu_c) / sigma_c
    return norm.cdf(z)


def compute_ei(mu_f: np.ndarray, sigma_f: np.ndarray, f_best: float) -> np.ndarray:
    """
    Expected Improvement for minimization, given best feasible f_best.
    EI(x) = E[max(0, f_best - f(x))].
    """
    sigma_f = np.maximum(sigma_f, 1e-12)
    improvement = f_best - mu_f
    z = improvement / sigma_f

    ei = improvement * norm.cdf(z) + sigma_f * norm.pdf(z)
    ei = np.where(sigma_f < 1e-10, 0.0, ei)
    ei = np.maximum(ei, 0.0)
    return ei


def teacher_cEI(
    gp_f: GaussianProcessRegressor,
    gp_c: GaussianProcessRegressor,
    X_candidates: np.ndarray,
    y_obs: np.ndarray,
    c_obs: np.ndarray,
) -> np.ndarray:
    """
    Teacher acquisition: constrained EI = EI * p_feas.

    Returns:
        cEI values for each candidate in X_candidates (shape [M]).
    """
    if len(y_obs) == 0:
        return np.zeros(X_candidates.shape[0], dtype=float)

    # Best feasible so far
    feasible_mask = c_obs <= 0.0
    if feasible_mask.any():
        f_best = float(np.min(y_obs[feasible_mask]))
    else:
        # No feasible point yet => EI = 0 by most constrained BO conventions
        return np.zeros(X_candidates.shape[0], dtype=float)

    mu_f, sigma_f = gp_f.predict(X_candidates, return_std=True)
    mu_c, sigma_c = gp_c.predict(X_candidates, return_std=True)

    p_feas = compute_p_feas(mu_c, sigma_c)
    ei = compute_ei(mu_f, sigma_f, f_best)

    return ei * p_feas


# =============================================================================
# Feature construction
# =============================================================================

def compute_global_features(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    c_obs: np.ndarray,
    dim: int,
    t: int,
    budget: int,
) -> np.ndarray:
    """
    Global BO state features, shared by all candidates at step t.

    Returns:
        g_t: np.ndarray shape [d_g]
    """
    if X_obs.shape[0] == 0:
        # Cold-start (should not really happen in our data collection)
        return np.zeros(8, dtype=np.float32)

    y_arr = y_obs.astype(float)
    c_arr = c_obs.astype(float)

    feasible_mask = c_arr <= 0.0
    if feasible_mask.any():
        f_best = float(np.min(y_arr[feasible_mask]))
    else:
        # If no feasible: use best overall as placeholder
        f_best = float(np.min(y_arr))

    frac_feas = float(np.mean(feasible_mask))

    y_mean = float(np.mean(y_arr))
    y_std = float(np.std(y_arr) + 1e-8)

    c_violation = np.maximum(0.0, c_arr)
    c_mean = float(np.mean(c_violation))
    c_std = float(np.std(c_violation) + 1e-8)

    progress = float(t) / float(budget)

    g = np.array(
        [
            dim,
            progress,
            f_best,
            frac_feas,
            y_mean,
            y_std,
            c_mean,
            c_std,
        ],
        dtype=np.float32,
    )
    return g


def compute_geometric_features(
    x: np.ndarray,
    X_obs: np.ndarray,
    c_obs: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Distances in normalized space:
        - dist to best feasible point
        - min dist to any feasible point
        - min dist to any infeasible point
    """
    if X_obs.shape[0] == 0:
        return 0.0, 0.0, 0.0

    c_arr = c_obs.astype(float)
    feasible_mask = c_arr <= 0.0
    infeasible_mask = ~feasible_mask

    # Dist to best feasible
    if feasible_mask.any():
        idx_best_feas = np.argmin(c_arr[feasible_mask])  # not ideal; approx
        # Slight hack: use first feasible as "best"
        feas_indices = np.where(feasible_mask)[0]
        best_idx = feas_indices[0]
        x_best_feas = X_obs[best_idx]
        dist_best_feas = float(np.linalg.norm(x - x_best_feas))
    else:
        dist_best_feas = 0.0

    if feasible_mask.any():
        dists_feas = np.linalg.norm(X_obs[feasible_mask] - x, axis=1)
        min_dist_feas = float(np.min(dists_feas))
    else:
        min_dist_feas = 0.0

    if infeasible_mask.any():
        dists_infeas = np.linalg.norm(X_obs[infeasible_mask] - x, axis=1)
        min_dist_infeas = float(np.min(dists_infeas))
    else:
        min_dist_infeas = 0.0

    return dist_best_feas, min_dist_feas, min_dist_infeas


def build_candidate_features(
    x: np.ndarray,
    gp_f: GaussianProcessRegressor,
    gp_c: GaussianProcessRegressor,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    c_obs: np.ndarray,
) -> np.ndarray:
    """
    Build feature vector φ(H_t, x) for a single candidate (normalized x).

    Returns:
        feat: np.ndarray shape [d_phi]
    """
    mu_f, sigma_f = gp_f.predict(x[None, :], return_std=True)
    mu_c, sigma_c = gp_c.predict(x[None, :], return_std=True)

    mu_f = float(mu_f[0])
    sigma_f = float(sigma_f[0])
    mu_c = float(mu_c[0])
    sigma_c = float(sigma_c[0])

    p_feas = float(compute_p_feas(np.array([mu_c]), np.array([sigma_c]))[0])

    # Best feasible so far
    feasible_mask = c_obs <= 0.0
    if feasible_mask.any():
        f_best = float(np.min(y_obs[feasible_mask]))
    else:
        f_best = float(np.min(y_obs))

    delta = f_best - mu_f

    dist_best_feas, min_dist_feas, min_dist_infeas = compute_geometric_features(
        x, X_obs, c_obs
    )

    feat = np.concatenate(
        [
            x.astype(np.float32),  # raw location in [0,1]^D
            np.array(
                [
                    mu_f,
                    sigma_f,
                    mu_c,
                    sigma_c,
                    p_feas,
                    f_best,
                    delta,
                    dist_best_feas,
                    min_dist_feas,
                    min_dist_infeas,
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    return feat


# =============================================================================
# Model: deep MLP for learned acquisition
# =============================================================================

class LearnedAcquisitionNet(nn.Module):
    """
    Deep acquisition network:
        input = [candidate features φ(H_t, x), global features g_t]
        output = scalar score (approx teacher cEI)

    You can later add exploration knobs by modifying the score at test time.
    """

    def __init__(self, d_phi: int, d_g: int):
        super().__init__()
        d_in = d_phi + d_g

        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.GELU(),
            nn.LayerNorm(256),

            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [batch, d_phi]
        global_feat: [batch, d_g] or [1, d_g] (broadcastable)
        """
        if global_feat.ndim == 2 and global_feat.size(0) == 1:
            global_feat = global_feat.expand(feat.size(0), -1)

        x = torch.cat([feat, global_feat], dim=-1)
        out = self.net(x)
        return out.squeeze(-1)  # shape [batch]


# =============================================================================
# Data collection from COCO problems
# =============================================================================

def collect_data_for_dim(
    dim: int,
    function_ids: List[int],
    instance_ids: List[int],
    target_points_per_pair: int,
    n_candidates_per_iter: int = 64,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect training data for a single dimension `dim`, for a specific set
    of BBOB-constrained function IDs and instance IDs.

    For each (fid, iid) pair, we collect at least `target_points_per_pair`
    candidate-level samples.

    Returns:
        features: [N, d_phi]
        global_feats: [N, d_g]
        targets: [N]
    """
    np.random.seed(seed)
    random.seed(seed)

    suite = cocoex.Suite(
        "bbob-constrained",
        "",
        f"dimensions:{dim}"
    )

    all_features = []
    all_gfeatures = []
    all_targets = []

    # Map (fid, iid) -> problem
    for problem in suite:
        fid, iid, d = parse_coco_ids(problem)
        if d != dim:
            continue
        if fid not in function_ids:
            continue
        if iid not in instance_ids:
            continue

        pair_key = (fid, iid)
        print(f"[Data dim={dim}] collecting for f={fid}, i={iid}, id={problem.id}")

        f, g, t = collect_data_for_problem(
            problem=problem,
            dim=dim,
            target_points=target_points_per_pair,
            n_initial=2 * dim,
            n_candidates_per_iter=n_candidates_per_iter,
            random_state=seed + fid * 100 + iid,
        )

        all_features.append(f)
        all_gfeatures.append(g)
        all_targets.append(t)

    if not all_features:
        raise RuntimeError(f"No problems found for dim={dim} with given fids/instances.")

    features = np.concatenate(all_features, axis=0)
    global_feats = np.concatenate(all_gfeatures, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print(
        f"[Data dim={dim}] Total samples: {features.shape[0]} "
        f"(~{target_points_per_pair} per (fid,iid))"
    )
    return features, global_feats, targets

def train_learned_acquisition_for_dim(
    dim: int,
    function_ids: List[int],
    instance_ids: List[int],
    target_points_per_pair: int = 100_000,
    n_candidates_per_iter: int = 64,
    batch_size: int = 512,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    model_path: str = None,
):
    """
    Train a learned acquisition model for a single dimension `dim`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train dim={dim}] Using device: {device}")

    # 1. Collect data for this dim only
    features, global_feats, targets = collect_data_for_dim(
        dim=dim,
        function_ids=function_ids,
        instance_ids=instance_ids,
        target_points_per_pair=target_points_per_pair,
        n_candidates_per_iter=n_candidates_per_iter,
        seed=seed,
    )

    d_phi = features.shape[1]
    d_g = global_feats.shape[1]

    model = LearnedAcquisitionNet(d_phi=d_phi, d_g=d_g).to(device)
    print(model)

    X_feat = torch.from_numpy(features).to(device)
    X_g = torch.from_numpy(global_feats).to(device)
    y = torch.from_numpy(targets).to(device)

    N = X_feat.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    split = int(0.9 * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_feat_train = X_feat[train_idx]
    X_g_train = X_g[train_idx]
    y_train = y[train_idx]

    X_feat_val = X_feat[val_idx]
    X_g_val = X_g[val_idx]
    y_val = y[val_idx]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    num_batches = math.ceil(train_idx.shape[0] / batch_size)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(X_feat_train.size(0), device=device)

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, X_feat_train.size(0))
            idx_b = perm[start:end]

            xb = X_feat_train[idx_b]
            gb = X_g_train[idx_b]
            yb = y_train[idx_b]

            optimizer.zero_grad()
            pred = model(xb, gb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= X_feat_train.size(0)

        model.eval()
        with torch.no_grad():
            pred_val = model(X_feat_val, X_g_val)
            val_loss = criterion(pred_val, y_val).item()

        print(
            f"[Epoch {epoch:03d} dim={dim}] "
            f"train MSE={epoch_loss:.4e} | val MSE={val_loss:.4e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "d_phi": d_phi,
                "d_g": d_g,
            }

    if model_path is None:
        model_path = f"learned_acquisition_cEI_dim{dim}.pt"

    if best_state is not None:
        torch.save(best_state, model_path)
        print(f"[Train dim={dim}] Saved best model to: {model_path}")
    else:
        print(f"[Train dim={dim}] WARNING: no best state recorded!")


def collect_data_for_problem(
    problem,
    dim: int,
    target_points: int,
    n_initial: int = None,
    n_candidates_per_iter: int = 64,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run teacher BO on a single COCO problem until we have at least
    `target_points` candidate-level training samples.

    Returns:
        features:     [N, d_phi]
        global_feats: [N, d_g]
        targets:      [N]
    """
    rng = np.random.RandomState(random_state)

    if n_initial is None:
        n_initial = 2 * dim

    X_obs = []
    y_obs = []
    c_obs = []

    # Initial random design in [0,1]^dim
    for _ in range(n_initial):
        x0 = rng.uniform(0.0, 1.0, size=(dim,))
        f0, c0 = evaluate_coco_problem(problem, x0)
        X_obs.append(x0)
        y_obs.append(f0)
        c_obs.append(c0)

    X_obs = np.array(X_obs, dtype=float)
    y_obs = np.array(y_obs, dtype=float)
    c_obs = np.array(c_obs, dtype=float)

    kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
        length_scale=0.2,
        length_scale_bounds=(1e-3, 10.0),
        nu=2.5,
    )
    gp_f = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=random_state,
    )
    gp_c = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=random_state + 1,
    )

    gp_f.fit(X_obs, y_obs)
    gp_c.fit(X_obs, c_obs)

    all_feats = []
    all_gfeats = []
    all_targets = []

    t = n_initial
    max_iters = 10_000  # safety cap

    while len(all_feats) < target_points and t < max_iters:
        # Candidate set
        X_cand = rng.uniform(0.0, 1.0, size=(n_candidates_per_iter, dim))

        # Teacher cEI
        cEI_vals = teacher_cEI(gp_f, gp_c, X_cand, y_obs, c_obs)

        # Global features
        # NOTE: budget is arbitrary here, we just use progress = t / (t + something)
        pseudo_budget = n_initial + target_points // n_candidates_per_iter
        g_t = compute_global_features(X_obs, y_obs, c_obs, dim, t, pseudo_budget)

        # Store all candidates as training samples
        for k in range(n_candidates_per_iter):
            xk = X_cand[k]
            feat_k = build_candidate_features(xk, gp_f, gp_c, X_obs, y_obs, c_obs)
            all_feats.append(feat_k)
            all_gfeats.append(g_t)
            all_targets.append(cEI_vals[k])

            if len(all_feats) >= target_points:
                break

        # Teacher picks next evaluation point
        best_idx = int(np.argmax(cEI_vals))
        x_next = X_cand[best_idx]
        f_next, c_next = evaluate_coco_problem(problem, x_next)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, f_next)
        c_obs = np.append(c_obs, c_next)

        gp_f.fit(X_obs, y_obs)
        gp_c.fit(X_obs, c_obs)

        t += 1

    features = np.asarray(all_feats, dtype=np.float32)
    global_feats = np.asarray(all_gfeats, dtype=np.float32)
    targets = np.asarray(all_targets, dtype=np.float32)
    return features, global_feats, targets



def collect_data_from_suite(
    dims: List[int],
    problems_per_dim: int = 4,
    budget_factor: int = 10,
    n_candidates_per_iter: int = 64,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect training data across multiple COCO BBOB-constrained problems.

    Returns:
        features: [N, d_phi]
        global_feats: [N, d_g]
        targets: [N]
    """
    random.seed(seed)
    np.random.seed(seed)

    all_features = []
    all_gfeatures = []
    all_targets = []

    for dim in dims:
        suite = cocoex.Suite(
            "bbob-constrained",
            "",
            f"dimensions:{dim}"
        )
        problem_indices = list(range(len(suite)))
        random.shuffle(problem_indices)
        problem_indices = problem_indices[:problems_per_dim]

        for idx in problem_indices:
            problem = suite[idx]
            problem_id = getattr(problem, "id", f"problem_{idx}")
            print(f"[Data] dim={dim}, problem_index={idx}, id={problem_id}")

            budget = budget_factor * dim
            f, g, t = collect_data_for_problem(
                problem=problem,
                dim=dim,
                budget=budget,
                n_initial=2 * dim,
                n_candidates_per_iter=n_candidates_per_iter,
                random_state=seed + idx,
            )
            all_features.append(f)
            all_gfeatures.append(g)
            all_targets.append(t)

    features = np.concatenate(all_features, axis=0)
    global_feats = np.concatenate(all_gfeatures, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print(f"[Data] Total samples: {features.shape[0]}")
    return features, global_feats, targets


# =============================================================================
# Training loop
# =============================================================================

def train_learned_acquisition(
    dims: List[int] = [2, 10],
    problems_per_dim: int = 4,
    budget_factor: int = 10,
    n_candidates_per_iter: int = 64,
    batch_size: int = 512,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    model_path: str = "learned_acquisition_cEI.pt",
):
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # 1. Collect data
    features, global_feats, targets = collect_data_from_suite(
        dims=dims,
        problems_per_dim=problems_per_dim,
        budget_factor=budget_factor,
        n_candidates_per_iter=n_candidates_per_iter,
        seed=seed,
    )

    d_phi = features.shape[1]
    d_g = global_feats.shape[1]

    # 2. Build model
    model = LearnedAcquisitionNet(d_phi=d_phi, d_g=d_g).to(device)
    print(model)

    # 3. Make tensors
    X_feat = torch.from_numpy(features).to(device)
    X_g = torch.from_numpy(global_feats).to(device)
    y = torch.from_numpy(targets).to(device)

    # 4. Train/val split
    N = X_feat.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    split = int(0.9 * N)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_feat_train = X_feat[train_idx]
    X_g_train = X_g[train_idx]
    y_train = y[train_idx]

    X_feat_val = X_feat[val_idx]
    X_g_val = X_g[val_idx]
    y_val = y[val_idx]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 5. Training loop
    num_batches = math.ceil(train_idx.shape[0] / batch_size)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(X_feat_train.size(0), device=device)

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, X_feat_train.size(0))
            idx_b = perm[start:end]

            xb = X_feat_train[idx_b]
            gb = X_g_train[idx_b]
            yb = y_train[idx_b]

            optimizer.zero_grad()
            pred = model(xb, gb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= X_feat_train.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_feat_val, X_g_val)
            val_loss = criterion(pred_val, y_val).item()

        print(
            f"[Epoch {epoch:03d}] "
            f"train MSE={epoch_loss:.4e} | val MSE={val_loss:.4e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "d_phi": d_phi,
                "d_g": d_g,
            }

    # 6. Save best model
    if best_state is not None:
        torch.save(best_state, model_path)
        print(f"[Train] Saved best model to: {model_path}")
    else:
        print("[Train] WARNING: no best state recorded!")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    TARGET_FIDS = [2, 4, 6, 50, 52, 54]
    TARGET_INSTANCES = [1, 2, 3, 4, 5]  # change if needed

    # Train separate models for each dimension you care about
    for dim in [2, 10]:  # add 40 if needed
        train_learned_acquisition_for_dim(
            dim=dim,
            function_ids=TARGET_FIDS,
            instance_ids=TARGET_INSTANCES,
            target_points_per_pair=100_000,  # your requirement
            n_candidates_per_iter=64,
            batch_size=512,
            n_epochs=200,
            lr=1e-3,
            weight_decay=1e-5,
            seed=42 + dim,
            model_path=f"learned_acquisition_cEI_dim{dim}.pt",
        )
