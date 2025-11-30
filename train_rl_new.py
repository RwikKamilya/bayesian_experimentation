#!/usr/bin/env python
"""
Meta-training config prediction policy on constrained COCO problems.

Setup:
- Suite: bbob-constrained
- Functions: [2, 4, 6, 50, 52, 54]
- Instances: [1, 2, 3]
- Dimensions: [2, 10] (separate policy per dimension)

For each dimension d:
- Episodes: 10_000
- Initial design: n_init = 4 * d random points in [0,1]^d
- Horizon: H = 15 RL steps per episode

At each episode:
- Sample a problem uniformly from the 18 (6 functions × 3 instances) for that d.
- Evaluate n_init random points (objective + constraints).
- Build state: summary stats + last point + best feasible point + local surrogate features.
- Run PPO for H steps to maximize a shaped reward:
  * Phase 1 (no feasible yet): big reward on first feasible, mild penalties otherwise + novelty bonus.
  * Phase 2 (after feasibility): normalized improvement in best feasible objective + light penalty for violation + novelty bonus.

The trained policies are saved as:
  models/coco_policy_dim2.pt
  models/coco_policy_dim10.pt

These checkpoints contain:
  - policy_state_dict
  - value_state_dict
  - dim
  - horizon
  - state_dim (needed at benchmark time)
"""

import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import cocoex  # pip install cocopp (which provides cocoex)
from gymnasium import spaces


# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------

FUNCTIONS = [2, 4, 6, 50, 52, 54]
INSTANCES = [1, 2, 3]
DIMENSIONS = [2, 10]

N_EPISODES = 10_000
HORIZON = 15
N_INITIAL_FACTOR = 4  # n_initial = N_INITIAL_FACTOR * dim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

# Reproducibility-ish
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# -----------------------------------------------------------------------------
# PPO COMPONENTS
# -----------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Gaussian policy: state -> mean action in R^d, with trainable log_std.
    Action is later squashed to [0, 1]^d in the environment wrapper.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
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
        h = self.shared(state)
        mean = self.mean_layer(h)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOAgent:
    """
    Basic PPO agent with Gaussian policy and value network.
    Uses full-batch updates per episode (small horizons).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        max_grad_norm: float = 0.5,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.value = ValueNetwork(state_dim).to(DEVICE)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Given a state (numpy), sample an action and return (action_in_[0,1]^d, log_prob).
        """
        state_t = torch.tensor(state, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(state_t)
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action).sum(dim=-1)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Map from R^d to [0,1]^d via tanh + rescale
        action = torch.tanh(action)
        action = (action + 1.0) / 2.0  # [-1,1] -> [0,1]
        action = action.clamp(0.0, 1.0)

        return action.cpu().numpy()[0].astype(np.float32), float(log_prob.cpu().item())

    def update(self, trajectories: List[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]]):
        """
        trajectories: list of (state, action, log_prob, reward, next_state, done)
        We use TD(0) targets: r + gamma * V(s') * (1-done).
        """
        if not trajectories:
            return

        states = torch.tensor(
            np.stack([t[0] for t in trajectories], axis=0),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        actions = torch.tensor(
            np.stack([t[1] for t in trajectories], axis=0),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        old_log_probs = torch.tensor(
            np.array([t[2] for t in trajectories], dtype=np.float32),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        rewards = torch.tensor(
            np.array([t[3] for t in trajectories], dtype=np.float32),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        next_states = torch.tensor(
            np.stack([t[4] for t in trajectories], axis=0),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        dones = torch.tensor(
            np.array([t[5] for t in trajectories], dtype=np.float32),
            dtype=TORCH_DTYPE,
            device=DEVICE,
        )

        with torch.no_grad():
            values = self.value(states)
            next_values = self.value(next_states)
            targets = rewards + self.gamma * next_values * (1.0 - dones)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            mean, std = self.policy(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = ratio.clamp(0.1, 10.0)  # numerical safety

            clipped_ratio = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages,
            ).mean()

            value_pred = self.value(states)
            value_loss = nn.MSELoss()(value_pred, targets)

            # Policy update
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # Value update
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()


# -----------------------------------------------------------------------------
# ENVIRONMENT: CONSTRAINED COCO META-BO
# -----------------------------------------------------------------------------

class CocoConstrainedBOEnv:
    def __init__(
        self,
        dim: int,
        horizon: int = 15,
        n_initial: int = None,
    ):
        self.dim = dim
        self.horizon = horizon
        self.n_initial = n_initial or (4 * dim)

        # Create the full constrained suite (no options string here)
        self.suite = cocoex.Suite("bbob-constrained", "", "")

        # We will sample (function, instance) pairs explicitly.
        # For this dimension, we only care about:
        #   FUNCTIONS = [2, 4, 6, 50, 52, 54]
        #   INSTANCES = [1, 2, 3]
        self.problem_keys = [
            (fid, inst)
            for fid in FUNCTIONS
            for inst in INSTANCES
        ]

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

        # 10 summary scalars + 3 surrogate scalars + last_x + best_x
        self.state_dim = 13 + 2 * dim

        self._current_problem = None
        self.lower_bounds = None
        self.upper_bounds = None

        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.c_observed: List[float] = []
        self.steps_taken: int = 0
        self.best_feasible_value: float = -np.inf

    # ------------------ Problem sampling & evaluation ------------------ #

    def _sample_problem(self):
        """
        Sample one of the (function, instance) pairs from FUNCTIONS × INSTANCES
        for this dimension, and reset the episode on that problem.
        """
        fid, inst = random.choice(self.problem_keys)

        # Use the same API as benchmark_rl.py
        self._current_problem = self.suite.get_problem_by_function_dimension_instance(
            fid, self.dim, inst
        )

        self.lower_bounds = np.array(self._current_problem.lower_bounds, dtype=np.float32)
        self.upper_bounds = np.array(self._current_problem.upper_bounds, dtype=np.float32)

        self.X_observed = []
        self.y_observed = []
        self.c_observed = []
        self.steps_taken = 0
        self.best_feasible_value = -np.inf

        # Initial random design
        for _ in range(self.n_initial):
            x_norm = np.random.uniform(0.0, 1.0, self.dim).astype(np.float32)
            y, c = self._evaluate(x_norm)
            self._add_observation(x_norm, y, c)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return self.lower_bounds + x_norm * (self.upper_bounds - self.lower_bounds)

    def _evaluate(self, x_norm: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate current COCO problem at x_norm in [0,1]^d.

        Returns:
          y: transformed objective (y = -f(x), to maximize)
          c: aggregated constraint (max_j g_j(x)), feasible if <= 0
        """
        x = self._denormalize(x_norm)
        f_val = float(self._current_problem(x))
        y = -f_val  # maximize -f

        if hasattr(self._current_problem, "constraint"):
            c_vals = np.asarray(self._current_problem.constraint(x), dtype=np.float32)
            c = float(np.max(c_vals)) if c_vals.size > 0 else -1.0
        else:
            c = -1.0  # treat as always feasible

        return y, c

    def _add_observation(self, x_norm: np.ndarray, y: float, c: float):
        self.X_observed.append(x_norm)
        self.y_observed.append(y)
        self.c_observed.append(c)

        c_arr = np.array(self.c_observed)
        y_arr = np.array(self.y_observed)
        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            self.best_feasible_value = float(np.max(y_arr[feasible_mask]))

    # ------------------ Local surrogate features (tiny RBF model) ------------------ #

    def _local_surrogate_features(self) -> np.ndarray:
        """
        Fit a tiny RBF ridge regression on recent points and return:
          [mu_last, mu_best, residual_std]
        If not enough data, returns zeros.
        """
        n = len(self.X_observed)
        if n < 5:
            return np.zeros(3, dtype=np.float32)

        X = np.array(self.X_observed, dtype=np.float32)
        y = np.array(self.y_observed, dtype=np.float32)

        # Use up to the last MAX_POINTS points for a local model
        MAX_POINTS = 50
        if n > MAX_POINTS:
            X = X[-MAX_POINTS:]
            y = y[-MAX_POINTS:]

        # Choose a few RBF centers (here: 5) from these points
        MAX_CENTERS = min(5, len(X))
        # pick centers as a subset (last points)
        centers = X[-MAX_CENTERS:]

        # Compute RBF design matrix: K_ij = exp(-||x_i - c_j||^2 / (2*sigma^2))
        # Use sigma ~ 0.2 * sqrt(dim) in normalized space
        sigma = 0.2 * np.sqrt(self.dim)
        if sigma <= 0.0:
            return np.zeros(3, dtype=np.float32)
        denom = 2.0 * (sigma ** 2)

        def rbf_kernel(A, B):
            # A: (n, d), B: (m, d)
            dists_sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-dists_sq / denom)

        K = rbf_kernel(X, centers)  # (n_points, n_centers)

        # Ridge regression: y ~ K @ alpha
        lam = 1e-3
        KT_K = K.T @ K  # (C,C)
        KT_y = K.T @ y  # (C,)
        A = KT_K + lam * np.eye(MAX_CENTERS, dtype=np.float32)
        try:
            alpha = np.linalg.solve(A, KT_y)
        except np.linalg.LinAlgError:
            return np.zeros(3, dtype=np.float32)

        # Predictions on training points
        y_hat = K @ alpha
        residuals = y - y_hat
        residual_std = float(np.std(residuals))

        # Predictions at last point and best feasible point
        last_x = X[-1:];  # shape (1,d)
        # Best feasible point index (w.r.t. global list)
        c_arr = np.array(self.c_observed, dtype=np.float32)
        y_arr = np.array(self.y_observed, dtype=np.float32)
        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_x = np.array(self.X_observed[best_idx], dtype=np.float32)[None, :]
        else:
            best_x = last_x.copy()

        K_last = rbf_kernel(last_x, centers)  # (1,C)
        K_best = rbf_kernel(best_x, centers)  # (1,C)

        mu_last = float((K_last @ alpha).ravel()[0])
        mu_best = float((K_best @ alpha).ravel()[0])

        return np.array([mu_last, mu_best, residual_std], dtype=np.float32)

    # ------------------ State & reward ------------------ #

    def _get_state(self) -> np.ndarray:
        if len(self.X_observed) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        y_arr = np.array(self.y_observed, dtype=np.float32)
        c_arr = np.array(self.c_observed, dtype=np.float32)
        feasible_mask = c_arr <= 0.0
        best_feasible = (
            float(np.max(y_arr[feasible_mask])) if feasible_mask.any() else -np.inf
        )

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr) + 1e-8)
        y_max = float(np.max(y_arr))
        n_feasible = int(np.sum(feasible_mask))
        feasible_ratio = float(n_feasible) / float(len(c_arr))

        last_point = np.array(self.X_observed[-1], dtype=np.float32)
        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_point = np.array(self.X_observed[best_idx], dtype=np.float32)
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
                float(len(self.X_observed)),
                float(self.steps_taken),
                float(self.horizon),
                float(self.dim),
            ],
            dtype=np.float32,
        )

        surrogate_feats = self._local_surrogate_features()  # (3,)

        state = np.concatenate(
            [summary, surrogate_feats, last_point, best_point],
            axis=0,
        ).astype(np.float32)

        # Safety: enforce state_dim
        if state.shape[0] != self.state_dim:
            # pad or truncate
            if state.shape[0] < self.state_dim:
                pad = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
                state = np.concatenate([state, pad], axis=0)
            else:
                state = state[: self.state_dim]

        return state

    def _compute_reward(self, old_best: float, y: float, c: float, x_norm: np.ndarray) -> float:
        """
        Feasibility-first reward with normalization and a small exploration bonus.

        - Phase 1: If no feasible point existed before this step, give a big
          bonus for the *first* feasible point, and only modest penalties for
          infeasible points, plus a novelty bonus.
        - Phase 2: Once feasibility exists, reward normalized improvement in
          best feasible y (remember: y = -f) and lightly penalize violation,
          again with a novelty bonus.

        Novelty: min distance of x_norm to past points in [0,1]^d.
        """
        is_feasible = c <= 0.0

        # Novelty: distance to previous points (ignoring the just-added one)
        if len(self.X_observed) > 1:
            prev_points = np.array(self.X_observed[:-1], dtype=np.float32)
            dists = np.linalg.norm(prev_points - x_norm[None, :], axis=1)
            min_dist = float(np.min(dists))
            novelty = np.clip(min_dist / np.sqrt(self.dim), 0.0, 1.0)
        else:
            novelty = 1.0

        had_feasible_before = not np.isneginf(old_best)
        reward = 0.0

        if not had_feasible_before:
            # Phase 1: still hunting for first feasible
            if is_feasible:
                # big bonus on first feasible hit
                reward = 5.0 + 0.5 * novelty
            else:
                violation = max(0.0, c)
                viol_norm = np.tanh(violation)  # [0,1)
                reward = -1.0 * viol_norm + 0.1 * novelty
        else:
            # Phase 2: already have at least one feasible
            if is_feasible:
                improvement = max(0.0, self.best_feasible_value - old_best)
                denom = abs(old_best) + 1.0
                norm_improvement = improvement / denom
                reward = 3.0 * norm_improvement + 0.2 * novelty
            else:
                violation = max(0.0, c)
                viol_norm = np.tanh(violation)
                reward = -0.5 * viol_norm + 0.1 * novelty

        reward = float(np.clip(reward, -5.0, 5.0))
        return reward

    # ------------------ Gym-like interface ------------------ #

    def reset(self) -> np.ndarray:
        self._sample_problem()
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Action expected in [0,1]^d
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        old_best = self.best_feasible_value
        y, c = self._evaluate(action)
        self._add_observation(action, y, c)

        self.steps_taken += 1
        done = self.steps_taken >= self.horizon

        reward = self._compute_reward(old_best, y, c, action)
        state = self._get_state()
        info: Dict[str, Any] = {"y": y, "c": c}
        return state, reward, done, info


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------

def train_coco_policy(
    dim: int,
    n_episodes: int = N_EPISODES,
    horizon: int = HORIZON,
    save_path: str = "models/coco_policy_dim.pt",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    env = CocoConstrainedBOEnv(
        dim=dim,
        horizon=horizon,
        n_initial=N_INITIAL_FACTOR * dim,
    )
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=dim,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        max_grad_norm=0.5,
    )

    episode_rewards: List[float] = []

    for episode in range(n_episodes):
        state = env.reset()
        trajectories: List[
            Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]
        ] = []
        episode_reward = 0.0

        for t in range(horizon):
            action, log_prob = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)

            trajectories.append(
                (state, action, log_prob, reward, next_state, float(done))
            )
            episode_reward += reward
            state = next_state

            if done:
                break

        if trajectories:
            agent.update(trajectories)

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            recent = float(np.mean(episode_rewards[-100:]))
            print(
                f"[dim={dim}] Episode {episode+1}/{n_episodes} | "
                f"AvgReward(100ep)={recent:.3f}"
            )

    # Save policy and value networks + metadata
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "value_state_dict": agent.value.state_dict(),
            "dim": dim,
            "horizon": horizon,
            "state_dim": env.state_dim,
        },
        save_path,
    )
    print(f"Saved PPO policy for dim={dim} to {save_path}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    for d in DIMENSIONS:
        train_coco_policy(
            dim=d,
            n_episodes=N_EPISODES,
            horizon=HORIZON,
            save_path=f"models/coco_policy_dim{d}.pt",
        )
