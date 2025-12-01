"""
train_next_config_transformer.py

Meta-learned next-config predictor for constrained BO on COCO problems.

- Offline: for each COCO problem, sample a large DOE (e.g. 10k points).
- Teacher: constrained EI (cEI) from GP surrogates picks a sequence of points.
- Student: Transformer that, given a prefix of the sequence
           (x_1, y_1, c_1), ..., (x_t, y_t, c_t),
           predicts the next config x_{t+1}.

This matches the "next word prediction" view, but with configs instead of words.
"""

import math
import random
import time
from typing import List, Tuple

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
from cocoex.exceptions import InvalidProblemException

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# COCO utilities
# =========================

def collect_data_for_dim(
        dim: int,
        function_ids: List[int],
        instance_ids: List[int],
        target_points_per_pair: int,
        n_offline_points: int = 10_000,
        T_min: int = 20,
        T_max: int = 50,
        n_candidates_per_episode: int = 128,
        seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect training data for a single dimension `dim`.

    For each (fid, iid) in function_ids × instance_ids we:
      - precompute n_offline_points evaluations
      - sample multiple episodes with T in [T_min, T_max]
      - in each episode, let the teacher (cEI) label n_candidates_per_episode points

    Until we have at least `target_points_per_pair` candidate samples
    per (fid, iid) pair.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Use the same pattern as bayesian_optimization.py
    suite = cocoex.Suite("bbob-constrained", "", "")

    selected_problems = []
    for fid in function_ids:
        for iid in instance_ids:
            try:
                problem = suite.get_problem_by_function_dimension_instance(
                    fid, dim, iid
                )
                selected_problems.append((problem, fid, iid))
            except Exception as e:
                print(
                    f"[Data dim={dim}] Could not load problem F{fid}, "
                    f"inst={iid}: {e}"
                )

    total_pairs = len(selected_problems)
    if total_pairs == 0:
        raise RuntimeError(
            f"No (fid,instance) problems found for dim={dim} with given fids/instances."
        )

    print(
        f"[Data dim={dim}] Will collect for {total_pairs} (fid,instance) pairs: "
        f"fids={function_ids}, instances={instance_ids}, "
        f"target_points_per_pair={target_points_per_pair}, "
        f"n_offline_points={n_offline_points}, "
        f"T∈[{T_min},{T_max}], n_candidates_per_episode={n_candidates_per_episode}"
    )

    all_features = []
    all_gfeatures = []
    all_targets = []

    for pair_idx, (problem, fid, iid) in enumerate(selected_problems, start=1):
        print(
            f"[Data dim={dim}] Pair {pair_idx}/{total_pairs}: "
            f"start F{fid}, inst={iid}"
        )
        pair_start = time.time()

        rng = np.random.RandomState(seed + fid * 100 + iid)

        f, g, t = collect_data_for_problem_from_offline(
            problem=problem,
            dim=dim,
            n_offline_points=n_offline_points,
            target_points=target_points_per_pair,
            T_min=T_min,
            T_max=T_max,
            n_candidates_per_episode=n_candidates_per_episode,
            rng=rng,
        )

        pair_elapsed = time.time() - pair_start
        print(
            f"[Data dim={dim}] Pair {pair_idx}/{total_pairs}: "
            f"done F{fid}, inst={iid} | samples={f.shape[0]} | "
            f"time={pair_elapsed / 60:.1f} min"
        )

        all_features.append(f)
        all_gfeatures.append(g)
        all_targets.append(t)

    features = np.concatenate(all_features, axis=0)
    global_feats = np.concatenate(all_gfeatures, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print(
        f"[Data dim={dim}] Total samples: {features.shape[0]} "
        f"(~{target_points_per_pair} per (fid,instance))"
    )
    return features, global_feats, targets


def train_learned_acquisition_for_dim(
        dim: int,
        function_ids: List[int],
        instance_ids: List[int],
        target_points_per_pair: int = 100_000,
        n_offline_points: int = 10_000,
        T_min: int = 20,
        T_max: int = 50,
        n_candidates_per_episode: int = 128,
        batch_size: int = 512,
        n_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 0,
        model_path: str = None,
):
    """
    Train a learned acquisition model for a single dimension `dim`
    using the offline episode schedule.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train dim={dim}] Using device: {device}")

    # 1. Collect data for this dim only (offline 10k + episodes)
    features, global_feats, targets = collect_data_for_dim(
        dim=dim,
        function_ids=function_ids,
        instance_ids=instance_ids,
        target_points_per_pair=target_points_per_pair,
        n_offline_points=n_offline_points,
        T_min=T_min,
        T_max=T_max,
        n_candidates_per_episode=n_candidates_per_episode,
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


def collect_data_for_problem_from_offline(
        problem,
        dim: int,
        n_offline_points: int,
        target_points: int,
        T_min: int,
        T_max: int,
        n_candidates_per_episode: int,
        rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single COCO problem:

        1) Build a big offline DOE (n_offline_points).
        2) Sample multiple episodes, each with T ∈ [T_min, T_max] history points
           and n_candidates_per_episode candidates, until ~target_points
           candidate training samples are collected.

    Returns:
        features:     [N, d_phi]
        global_feats: [N, d_g]
        targets:      [N]
    """
    fid, iid, _ = parse_coco_ids(problem)
    print(
        f"[Offline dim={dim} f={fid} i={iid}] "
        f"n_offline_points={n_offline_points}, target_points={target_points}"
    )

    X_all, y_all, c_all = precompute_offline_dataset(
        problem, dim=dim, n_points=n_offline_points, rng=rng
    )

    all_feats = []
    all_gfeats = []
    all_targets = []

    ep = 0
    while len(all_feats) * n_candidates_per_episode < target_points:
        T = rng.randint(T_min, T_max + 1)
        feat_ep, g_ep, t_ep = sample_training_episode(
            X_all, y_all, c_all,
            dim=dim,
            T=T,
            n_candidates=n_candidates_per_episode,
            rng=rng,
        )

        all_feats.append(feat_ep)
        all_gfeats.append(g_ep)
        all_targets.append(t_ep)

        ep += 1
        n_samples = sum(f.shape[0] for f in all_feats)
        if ep % 5 == 0 or n_samples >= target_points:
            print(
                f"  [Offline dim={dim} f={fid} i={iid}] "
                f"episodes={ep} | samples={n_samples}/{target_points}"
            )

    features = np.concatenate(all_feats, axis=0)
    global_feats = np.concatenate(all_gfeats, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return features, global_feats, targets


def parse_coco_ids(problem):
    pid = problem.id  # e.g. 'bbob-constrained_f050_i07_d02'
    parts = pid.split('_')
    f_str = parts[1]  # 'f050'
    i_str = parts[2]  # 'i07'
    d_str = parts[3]  # 'd02'
    fid = int(f_str[1:])
    iid = int(i_str[1:])
    dim = int(d_str[1:])
    return fid, iid, dim


def evaluate_coco_problem(problem, x_normalized: np.ndarray) -> Tuple[float, float]:
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)
    x = lb + x_normalized * (ub - lb)
    f_val = float(problem(x))

    if getattr(problem, "number_of_constraints", 0) > 0:
        c_vals = np.asarray(problem.constraint(x), dtype=float)
        c_val = float(np.max(c_vals)) if c_vals.size > 0 else 0.0
    else:
        c_val = 0.0  # treat unconstrained as feasible

    if not np.isfinite(f_val):
        # you can either resample or clip
        f_val = 1e6
    if not np.isfinite(c_val):
        c_val = 1e6  # big positive => clearly infeasible

    return f_val, c_val



# =========================
# Teacher: constrained EI
# =========================

def compute_p_feas(mu_c: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
    sigma_c = np.maximum(sigma_c, 1e-12)
    z = (0.0 - mu_c) / sigma_c
    return norm.cdf(z)


def compute_ei(mu_f: np.ndarray, sigma_f: np.ndarray, f_best: float) -> np.ndarray:
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
    if len(y_obs) == 0:
        return np.zeros(X_candidates.shape[0], dtype=float)

    feasible_mask = c_obs <= 0.0
    if feasible_mask.any():
        f_best = float(np.min(y_obs[feasible_mask]))
    else:
        # no feasible yet -> no EI
        return np.zeros(X_candidates.shape[0], dtype=float)

    mu_f, sigma_f = gp_f.predict(X_candidates, return_std=True)
    mu_c, sigma_c = gp_c.predict(X_candidates, return_std=True)

    p_feas = compute_p_feas(mu_c, sigma_c)
    ei = compute_ei(mu_f, sigma_f, f_best)
    return ei * p_feas


# =========================
# Offline dataset + episodes
# =========================

def precompute_offline_dataset(
        problem,
        dim: int,
        n_points: int,
        rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_all = rng.uniform(0.0, 1.0, size=(n_points, dim))
    y_all = np.zeros(n_points, dtype=float)
    c_all = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        # retry a few times in case COCO complains (dimension, reset, etc.)
        for attempt in range(3):
            try:
                f_val, c_val = evaluate_coco_problem(problem, X_all[i])
                y_all[i] = f_val
                c_all[i] = c_val
                break
            except InvalidProblemException as e:
                print(
                    f"[WARN] InvalidProblemException at offline sample {i}, "
                    f"attempt {attempt + 1}: {e}"
                )
                # resample x and try again
                X_all[i] = rng.uniform(0.0, 1.0, size=(dim,))
        else:
            # After 3 attempts still failing → raise so you see the message.
            raise RuntimeError(
                f"Giving up on offline sample {i} for this problem; "
                f"check dimension and COCO setup."
            )

    return X_all, y_all, c_all


def build_teacher_sequence_from_offline(
        X_all: np.ndarray,
        y_all: np.ndarray,
        c_all: np.ndarray,
        dim: int,
        T_max: int,
        rng: np.random.RandomState,
        n_init: int = None,
        n_candidates_per_step: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a single teacher sequence of length T_max using offline data.

    - Start with n_init random points as initial history.
    - At each step, fit GPs on history.
    - From a random candidate subset of remaining offline points,
      pick argmax cEI as next point.

    Returns:
        X_seq: [T_max, dim]
        y_seq: [T_max]
        c_seq: [T_max]
    """
    N = X_all.shape[0]
    assert N >= T_max, "offline dataset must have at least T_max points"

    if n_init is None:
        n_init = min(2 * dim, T_max // 2)
        n_init = max(4, n_init)

    indices = rng.permutation(N)
    history_idx = list(indices[:n_init])
    remaining = list(indices[n_init:])

    X_obs = X_all[history_idx]
    y_obs = y_all[history_idx]
    c_obs = c_all[history_idx]

    seq_indices = list(history_idx)

    # GP models
    kernel = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(
        length_scale=0.2,
        length_scale_bounds=(1e-3, 10.0),
        nu=2.5,
    )
    gp_f = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,  # <-- no LBFGS, just use initial kernel params
    )

    gp_c = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
    )

    for t in range(n_init, T_max):
        gp_f.fit(X_obs, y_obs)
        gp_c.fit(X_obs, c_obs)

        if not remaining:
            break

        cand_size = min(n_candidates_per_step, len(remaining))
        cand_indices = rng.choice(remaining, size=cand_size, replace=False)
        X_cand = X_all[cand_indices]

        cEI_vals = teacher_cEI(gp_f, gp_c, X_cand, y_obs, c_obs)

        best_local = int(np.argmax(cEI_vals))
        best_global_idx = int(cand_indices[best_local])

        seq_indices.append(best_global_idx)

        X_obs = np.vstack([X_obs, X_all[best_global_idx]])
        y_obs = np.append(y_obs, y_all[best_global_idx])
        c_obs = np.append(c_obs, c_all[best_global_idx])

        remaining.remove(best_global_idx)

    # Truncate / pad to T_max length
    seq_indices = seq_indices[:T_max]
    if len(seq_indices) < T_max:
        # pad by repeating last index (rare)
        last_idx = seq_indices[-1]
        seq_indices += [last_idx] * (T_max - len(seq_indices))

    seq_indices = np.array(seq_indices, dtype=int)
    X_seq = X_all[seq_indices]
    y_seq = y_all[seq_indices]
    c_seq = c_all[seq_indices]
    return X_seq, y_seq, c_seq


def build_sequences_for_dim(
        dim: int,
        function_ids: List[int],
        instance_ids: List[int],
        n_offline_points: int = 10_000,
        episodes_per_pair: int = 50,
        T_max: int = 50,
        seed: int = 0,
):
    """
    For a given dimension:
        - for each (fid, iid) in function_ids x instance_ids,
        - build one offline dataset (n_offline_points pts),
        - from that, sample `episodes_per_pair` teacher sequences of length T_max.

    Returns:
        X_seqs: [N_seq, T_max, dim]
        y_seqs: [N_seq, T_max]
        c_seqs: [N_seq, T_max]
    """
    np.random.seed(seed)
    random.seed(seed)

    # Use the *safe* way of getting COCO problems, as in bayesian_optimization.py
    suite = cocoex.Suite("bbob-constrained", "", "")

    problems = []
    for fid in function_ids:
        for iid in instance_ids:
            try:
                problem = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
                problems.append((problem, fid, iid))
            except Exception as e:
                print(
                    f"[Seq dim={dim}] Could not load problem F{fid}, inst={iid}: {e}"
                )

    if not problems:
        raise RuntimeError(f"No matching problems found for dim={dim}.")

    print(
        f"[Seq dim={dim}] Will build sequences for {len(problems)} (fid,instance) pairs "
        f"with n_offline={n_offline_points}, episodes_per_pair={episodes_per_pair}, T_max={T_max}"
    )

    X_seq_list = []
    y_seq_list = []
    c_seq_list = []

    for p_idx, (problem, fid, iid) in enumerate(problems, start=1):
        print(
            f"[Seq dim={dim}] Problem {p_idx}/{len(problems)}: F{fid}, inst={iid}"
        )
        start_t = time.time()

        rng = np.random.RandomState(seed + fid * 100 + iid)

        # ---- build offline DOE; SKIP problem if it keeps failing ----
        try:
            X_all, y_all, c_all = precompute_offline_dataset(
                problem, dim=dim, n_points=n_offline_points, rng=rng
            )
        except RuntimeError as e:
            print(
                f"  [Seq dim={dim} F{fid} i={iid}] "
                f"Skipping problem due to offline failure: {e}"
            )
            continue

        # ---- sample teacher sequences ----
        for ep in range(episodes_per_pair):
            X_seq, y_seq, c_seq = build_teacher_sequence_from_offline(
                X_all, y_all, c_all,
                dim=dim,
                T_max=T_max,
                rng=rng,
            )
            X_seq_list.append(X_seq[None, :, :])
            y_seq_list.append(y_seq[None, :])
            c_seq_list.append(c_seq[None, :])

            if (ep + 1) % 10 == 0:
                print(
                    f"  [Seq dim={dim} F{fid} i={iid}] "
                    f"episodes={ep + 1}/{episodes_per_pair}"
                )

        elapsed = (time.time() - start_t) / 60.0
        print(
            f"[Seq dim={dim}] Done F{fid}, inst={iid}, episodes={episodes_per_pair}, "
            f"time={elapsed:.1f} min"
        )

    if not X_seq_list:
        raise RuntimeError(
            f"[Seq dim={dim}] No sequences built (all problems failed offline)."
        )

    X_seqs = np.concatenate(X_seq_list, axis=0).astype(np.float32)
    y_seqs = np.concatenate(y_seq_list, axis=0).astype(np.float32)
    c_seqs = np.concatenate(c_seq_list, axis=0).astype(np.float32)

    for name, arr in [("X_seqs", X_seqs), ("y_seqs", y_seqs), ("c_seqs", c_seqs)]:
        print(
            f"{name}: shape={arr.shape}, finite={np.isfinite(arr).all()}, "
            f"min={arr.min()}, max={arr.max()}"
        )

    print(
        f"[Seq dim={dim}] Built {X_seqs.shape[0]} sequences of length {X_seqs.shape[1]}"
    )
    return X_seqs, y_seqs, c_seqs


# =========================
# Dataset for Transformer
# =========================

class NextConfigDataset(torch.utils.data.Dataset):
    """
    Each episode is a sequence:
        (x_0, y_0, c_0), ..., (x_{T-1}, y_{T-1}, c_{T-1})

    We turn this into many training examples:
        prefix length L in {1,...,T-1}
        input:   tokens[0:L]   (padded to max_len)
        target:  x_L  (the next config)

    We encode 'all prefixes' by mapping dataset index -> (episode_idx, target_step).

    IMPORTANT: we normalize y and c to [0,1] with clipping so that all
    token features are in a small, bounded range. This prevents the
    Transformer from seeing huge magnitudes and producing NaNs.
    """

    def __init__(self, X_seqs: np.ndarray, y_seqs: np.ndarray, c_seqs: np.ndarray):
        """
        X_seqs: [N_episodes, T_max, dim]  (x in [0,1]^dim)
        y_seqs: [N_episodes, T_max]
        c_seqs: [N_episodes, T_max]
        """
        assert X_seqs.ndim == 3
        # Store as torch tensors
        self.X_seqs = torch.from_numpy(X_seqs.astype(np.float32))  # [N, T, D]
        self.y_seqs = torch.from_numpy(y_seqs.astype(np.float32))  # [N, T]
        self.c_seqs = torch.from_numpy(c_seqs.astype(np.float32))  # [N, T]

        self.num_episodes, self.max_len, self.dim = self.X_seqs.shape
        self.prefixes_per_episode = self.max_len - 1
        self.total_examples = self.num_episodes * self.prefixes_per_episode

        # ---------- GLOBAL NORMALIZATION STATS FOR y AND c ----------
        # Flatten over all episodes and timesteps
        y_flat = self.y_seqs.view(-1)
        c_flat = self.c_seqs.view(-1)

        # Replace NaNs/Infs if any slipped through
        y_flat = torch.nan_to_num(y_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        c_flat = torch.nan_to_num(c_flat, nan=0.0, posinf=1e6, neginf=-1e6)

        # Clip extreme values so outliers / sentinels don't dominate stats
        y_flat = torch.clamp(y_flat, min=-1e3, max=1e3)
        c_flat = torch.clamp(c_flat, min=-1e3, max=1e3)

        # Compute mean/std on the clipped data
        self.y_mean = y_flat.mean()
        self.y_std = y_flat.std().clamp_min(1e-6)

        self.c_mean = c_flat.mean()
        self.c_std = c_flat.std().clamp_min(1e-6)

        # How strongly to clip standardized values before mapping to [0,1]
        self._K = 5.0  # z in [-K, K] -> mapped to [0,1]

    def __len__(self):
        return self.total_examples

    def _norm_yc(self, y_hist: torch.Tensor, c_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize y and c to [0,1] using dataset-wide mean/std, with clipping.
        """
        # Standardize
        z_y = (y_hist - self.y_mean) / self.y_std
        z_c = (c_hist - self.c_mean) / self.c_std

        # Clip to avoid huge magnitudes
        z_y = torch.clamp(z_y, min=-self._K, max=self._K)
        z_c = torch.clamp(z_c, min=-self._K, max=self._K)

        # Map [-K, K] -> [0,1]
        y_norm = (z_y + self._K) / (2.0 * self._K)
        c_norm = (z_c + self._K) / (2.0 * self._K)

        return y_norm, c_norm

    def __getitem__(self, idx: int):
        ep_idx = idx // self.prefixes_per_episode
        step_idx = idx % self.prefixes_per_episode  # 0..T-2
        L = step_idx + 1  # prefix length (1..T-1)
        target_step = L   # index of next point

        X_seq = self.X_seqs[ep_idx]      # [T, D]
        y_seq = self.y_seqs[ep_idx]      # [T]
        c_seq = self.c_seqs[ep_idx]      # [T]

        # Prefix [0:L]
        x_hist = X_seq[:L]                       # [L, D], already ~[0,1]
        y_hist = y_seq[:L].unsqueeze(-1)         # [L, 1]
        c_hist = c_seq[:L].unsqueeze(-1)         # [L, 1]

        # Normalize y and c to [0,1]
        y_norm, c_norm = self._norm_yc(y_hist, c_hist)

        # Safety: clamp x_hist to [0,1] in case of numerical drift
        x_hist = torch.clamp(x_hist, 0.0, 1.0)

        tokens = torch.cat([x_hist, y_norm, c_norm], dim=-1)  # [L, D+2]

        # Pad to max_len
        token_dim = tokens.size(-1)
        if L < self.max_len:
            pad = torch.zeros(self.max_len - L, token_dim, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad], dim=0)

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = torch.zeros(self.max_len, dtype=torch.float32)
        attn_mask[:L] = 1.0

        target_x = X_seq[target_step]  # [D], still in [0,1]

        # Final safety: ensure tokens and target_x are finite
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1.0, neginf=0.0)
        if not torch.isfinite(target_x).all():
            target_x = torch.nan_to_num(target_x, nan=0.0, posinf=1.0, neginf=0.0)

        return tokens, attn_mask, target_x


# =========================
# Transformer model
# =========================

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
        max_len:   maximum episode length (T_max)
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
            pred_x: [batch, dim]  (next config)
        """
        B, L, _ = tokens.shape
        x = self.input_proj(tokens)  # [B, L, d_model]
        x = self.pos_enc(x)  # [B, L, d_model]

        # Transformer expects [L, B, d_model]
        x = x.transpose(0, 1)  # [L, B, d_model]

        # key_padding_mask: True at PAD positions
        key_padding_mask = (attn_mask == 0)  # [B, L]
        enc_out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [L, B, d_model]

        enc_out = enc_out.transpose(0, 1)  # [B, L, d_model]

        # get last *real* token hidden state for each batch element
        lengths = attn_mask.sum(dim=1).long() - 1  # [B]
        lengths = torch.clamp(lengths, min=0)

        idx = lengths.view(B, 1, 1).expand(-1, 1, enc_out.size(-1))  # [B, 1, d_model]
        last_hidden = enc_out.gather(1, idx).squeeze(1)  # [B, d_model]

        pred_x = self.head(last_hidden)  # [B, dim]
        return pred_x


# =========================
# Training for a single dim
# =========================

def train_next_config_model_for_dim(
        dim: int,
        function_ids: List[int],
        instance_ids: List[int],
        n_offline_points: int = 10_000,
        episodes_per_pair: int = 50,
        T_max: int = 50,
        batch_size: int = 128,
        n_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 0,
        model_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train dim={dim}] Using device: {device}")

    # 1) build sequences
    X_seqs, y_seqs, c_seqs = build_sequences_for_dim(
        dim=dim,
        function_ids=function_ids,
        instance_ids=instance_ids,
        n_offline_points=n_offline_points,
        episodes_per_pair=episodes_per_pair,
        T_max=T_max,
        seed=seed,
    )

    # 2) split episodes into train/val
    num_episodes = X_seqs.shape[0]
    indices = np.arange(num_episodes)
    np.random.shuffle(indices)
    split = int(0.9 * num_episodes)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X_seqs[train_idx]
    y_train = y_seqs[train_idx]
    c_train = c_seqs[train_idx]

    X_val = X_seqs[val_idx]
    y_val = y_seqs[val_idx]
    c_val = c_seqs[val_idx]

    train_dataset = NextConfigDataset(X_train, y_train, c_train)
    val_dataset = NextConfigDataset(X_val, y_val, c_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 3) model + optimizer
    model = NextConfigTransformer(dim=dim, max_len=T_max).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        num_train = 0

        for tokens, attn_mask, target_x in train_loader:
            tokens = tokens.to(device)
            attn_mask = attn_mask.to(device)
            target_x = target_x.to(device)

            optimizer.zero_grad()
            pred_x = model(tokens, attn_mask)
            loss = criterion(pred_x, target_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = tokens.size(0)
            train_loss += loss.item() * bs
            num_train += bs

        train_loss /= max(num_train, 1)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        num_val = 0
        with torch.no_grad():
            for tokens, attn_mask, target_x in val_loader:
                tokens = tokens.to(device)
                attn_mask = attn_mask.to(device)
                target_x = target_x.to(device)

                pred_x = model(tokens, attn_mask)
                loss = criterion(pred_x, target_x)

                bs = tokens.size(0)
                val_loss += loss.item() * bs
                num_val += bs

        val_loss /= max(num_val, 1)

        print(
            f"[Epoch {epoch:03d} dim={dim}] "
            f"train MSE={train_loss:.4e} | val MSE={val_loss:.4e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "dim": dim,
                "T_max": T_max,
            }

    if model_path is None:
        model_path = f"next_config_transformer_dim{dim}.pt"

    if best_state is not None:
        torch.save(best_state, model_path)
        print(f"[Train dim={dim}] Saved best model to: {model_path}")
    else:
        print(f"[Train dim={dim}] WARNING: no best state recorded!")


def main():
    TARGET_FIDS = [2, 4, 6, 50, 52, 54]
    TARGET_INSTANCES = [1, 2, 3]
    dims_to_train = [2, 10]

    total_pairs_per_dim = len(TARGET_FIDS) * len(TARGET_INSTANCES)
    total_pairs_all = total_pairs_per_dim * len(dims_to_train)

    print(
        f"[Main] Training next-config Transformer for dims={dims_to_train} "
        f"({total_pairs_all} (fid,instance) pairs total, "
        f"{total_pairs_per_dim} per dimension, "
        f"offline_points=10000, episodes_per_pair=50, T_max=50)"
    )

    for dim in dims_to_train:
        train_next_config_model_for_dim(
            dim=dim,
            function_ids=TARGET_FIDS,
            instance_ids=TARGET_INSTANCES,
            n_offline_points=1000,
            episodes_per_pair=30,
            T_max=50,
            batch_size=128,
            n_epochs=200,
            lr=1e-3,
            weight_decay=1e-5,
            seed=42 + dim,
            model_path=f"next_config_transformer_dim{dim}.pt",
        )


if __name__ == "__main__":
    main()
