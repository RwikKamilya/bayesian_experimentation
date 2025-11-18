import warnings

warnings.filterwarnings('ignore')

"""
RL-Enhanced Constrained Bayesian Optimization
Uses PPO trained on GP surrogate to select next query points with lookahead
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.distributions import Normal
import cocoex


# =============================================================================
# ENVIRONMENT: Surrogate-based BO Environment
# =============================================================================

class SurrogateConstrainedBOEnv(gym.Env):
    """
    RL Environment for Constrained BO that uses GP surrogate for training.
    The agent learns to select points that maximize long-term improvement.
    """

    def __init__(self, dim, horizon=5, n_initial=None):
        super().__init__()
        self.dim = dim
        self.horizon = horizon  # Lookahead steps for RL episode
        self.n_initial = n_initial or (2 * dim)

        # Action space: normalized coordinates in [0, 1]^D
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

        # State space: summary statistics + recent points
        state_dim = 10 + 2 * dim  # Summary stats + last point + best point
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # GPs for objective and constraints - more stable settings
        kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) * \
                 Matern(length_scale=0.5, length_scale_bounds=(0.1, 2.0), nu=2.5)
        self.gp_objective = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4, normalize_y=True, n_restarts_optimizer=2
        )
        self.gp_constraint = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4, normalize_y=True, n_restarts_optimizer=2
        )

        # GPs for objective and constraints - fixed hyperparameters (no L-BFGS)
        # kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) * \
        #          Matern(length_scale=0.5, length_scale_bounds=(0.1, 2.0), nu=2.5)
        # self.gp_objective = GaussianProcessRegressor(
        #     kernel=kernel, alpha=1e-4, normalize_y=True, optimizer=None
        # )
        # self.gp_constraint = GaussianProcessRegressor(
        #     kernel=kernel, alpha=1e-4, normalize_y=True, optimizer=None
        # )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize with random points
        self.X_observed = np.random.uniform(0, 1, (self.n_initial, self.dim))

        # Generate synthetic function for this episode (random GP sample)
        self.true_objective = self._sample_random_function()
        self.true_constraint = self._sample_random_constraint()

        # Evaluate initial points
        self.y_observed = np.array([self.true_objective(x) for x in self.X_observed])
        self.c_observed = np.array([self.true_constraint(x) for x in self.X_observed])

        # Fit initial GPs
        self._update_gps()

        # Track best feasible
        self.best_feasible_value = self._get_best_feasible()
        self.steps_taken = 0

        return self._get_state(), {}

    def _sample_random_function(self):
        """Sample a random smooth function from a GP prior"""
        n_inducing = 50
        X_inducing = np.random.uniform(0, 1, (n_inducing, self.dim))

        kernel = ConstantKernel(1.0) * RBF(length_scale=0.3)
        K = kernel(X_inducing, X_inducing)
        K += 1e-6 * np.eye(n_inducing)

        y_inducing = np.random.multivariate_normal(
            mean=np.zeros(n_inducing), cov=K
        )

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(X_inducing, y_inducing)

        return lambda x: gp.predict(x.reshape(1, -1))[0]

    def _sample_random_constraint(self):
        """Sample a random constraint function (negative = feasible)"""
        # Create a constraint that's feasible in ~50% of space
        center = np.random.uniform(0.3, 0.7, self.dim)
        radius = np.random.uniform(0.3, 0.5)

        def constraint(x):
            dist = np.linalg.norm(x - center)
            return dist - radius  # Negative inside sphere

        return constraint

    def _update_gps(self):
        """Fit GPs to current observations"""
        # Add small noise to avoid numerical issues
        y_train = self.y_observed + np.random.normal(0, 1e-6, len(self.y_observed))
        c_train = self.c_observed + np.random.normal(0, 1e-6, len(self.c_observed))

        try:
            self.gp_objective.fit(self.X_observed, y_train)
            self.gp_constraint.fit(self.X_observed, c_train)
        except Exception as e:
            # If fitting fails, use previous GP
            pass

    def _get_best_feasible(self):
        """Get best feasible objective value seen so far"""
        feasible_mask = self.c_observed <= 0
        if not feasible_mask.any():
            return -np.inf
        return np.max(self.y_observed[feasible_mask])

    def _get_state(self):
        """Encode current state for RL agent"""
        best_feasible = self.best_feasible_value
        if np.isinf(best_feasible) or np.isnan(best_feasible):
            best_feasible = -10.0

        # Summary statistics
        y_mean = np.mean(self.y_observed)
        y_std = np.std(self.y_observed) + 1e-6
        y_max = np.max(self.y_observed)

        n_feasible = np.sum(self.c_observed <= 0)
        feasible_ratio = n_feasible / len(self.c_observed)

        # Last queried point
        last_point = self.X_observed[-1]

        # Best feasible point
        feasible_mask = self.c_observed <= 0
        if feasible_mask.any():
            best_idx = np.argmax(self.y_observed * feasible_mask)
            best_point = self.X_observed[best_idx]
        else:
            best_point = np.zeros(self.dim)

        # Remaining budget
        budget_remaining = (self.horizon - self.steps_taken) / self.horizon

        state = np.concatenate([
            [best_feasible, y_mean, y_std, y_max, feasible_ratio,
             budget_remaining, len(self.X_observed) / 100],
            [0, 0, 0],  # Padding to make 10 summary stats
            last_point,
            best_point
        ]).astype(np.float32)

        # Clip to prevent extreme values
        state = np.clip(state, -100, 100)

        return state

    def step(self, action):
        """
        Take action (query a point) on the SURROGATE model.
        This makes training cheap!
        """
        x_next = np.clip(action, 0, 1)

        # Predict using GP (cheap!)
        y_pred, y_std = self.gp_objective.predict(x_next.reshape(1, -1), return_std=True)
        c_pred, c_std = self.gp_constraint.predict(x_next.reshape(1, -1), return_std=True)

        y_pred = y_pred[0]
        c_pred = c_pred[0]
        y_std = y_std[0]
        c_std = c_std[0]

        # Add noise to simulate uncertainty
        y_sample = y_pred + np.random.normal(0, max(y_std, 0.01))
        c_sample = c_pred + np.random.normal(0, max(c_std, 0.01))

        # Update observations with sampled values
        self.X_observed = np.vstack([self.X_observed, x_next])
        self.y_observed = np.append(self.y_observed, y_sample)
        self.c_observed = np.append(self.c_observed, c_sample)

        # Refit GPs
        self._update_gps()

        # Compute reward
        reward = self._compute_reward(y_sample, c_sample, y_std)

        # Update tracking
        old_best = self.best_feasible_value
        self.best_feasible_value = self._get_best_feasible()
        self.steps_taken += 1

        # Episode ends after horizon steps
        done = self.steps_taken >= self.horizon

        # Bonus reward for improvement at end of episode
        if done and self.best_feasible_value > old_best:
            reward += 10.0 * (self.best_feasible_value - old_best)

        return self._get_state(), reward, done, False, {}

    def _compute_reward(self, y, c, uncertainty):
        """Compute reward for querying this point"""
        is_feasible = c <= 0

        # Clip values to prevent extreme rewards
        y = np.clip(y, -100, 100)
        c = np.clip(c, -100, 100)
        uncertainty = np.clip(uncertainty, 0, 10)

        if is_feasible:
            # Reward improvement over current best
            if not np.isinf(self.best_feasible_value):
                improvement = max(0, y - self.best_feasible_value)
            else:
                improvement = 1.0  # First feasible point
            reward = 10.0 * improvement

            # Small bonus for high uncertainty (exploration)
            reward += 0.1 * uncertainty
        else:
            # Penalty for infeasibility, but small reward for exploring boundary
            violation = max(0, c)
            reward = -1.0 * violation

            # Reward for reducing uncertainty in infeasible region
            reward += 0.05 * uncertainty

        # Clip final reward
        reward = np.clip(reward, -50, 50)

        return float(reward)


# =============================================================================
# PPO AGENT
# =============================================================================

class PolicyNetwork(nn.Module):
    """Policy network for continuous action space"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        # Check for NaN in input
        if torch.isnan(state).any():
            state = torch.nan_to_num(state, nan=0.0)

        shared = self.shared(state)
        mean = torch.sigmoid(self.mean_layer(shared))  # Bounded to [0, 1]
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)

        # Safety check
        mean = torch.nan_to_num(mean, nan=0.5)
        std = torch.nan_to_num(std, nan=0.1)

        return mean, std

    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            return mean
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class ValueNetwork(nn.Module):
    """Value network for critic"""

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, epochs=10):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if deterministic:
                action = self.policy.get_action(state_tensor, deterministic=True)
                return action.numpy()[0], None
            else:
                action, log_prob = self.policy.get_action(state_tensor)
                return action.numpy()[0], log_prob.item()

    def update(self, trajectories):
        """Update policy and value networks using collected trajectories"""
        states = torch.FloatTensor(np.array([t[0] for t in trajectories]))
        actions = torch.FloatTensor(np.array([t[1] for t in trajectories]))
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in trajectories]))
        rewards = torch.FloatTensor(np.array([t[3] for t in trajectories]))
        next_states = torch.FloatTensor(np.array([t[4] for t in trajectories]))
        dones = torch.FloatTensor(np.array([t[5] for t in trajectories]))

        # Check for NaN/Inf in data
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("Warning: NaN/Inf in states, skipping update")
            return

        # Compute returns and advantages
        with torch.no_grad():
            values = self.value(states)
            next_values = self.value(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.epochs):
            # Policy loss
            mean, std = self.policy(states)

            # Check for NaN
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("Warning: NaN in policy output, skipping update")
                return

            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            # Value loss
            value_pred = self.value(states)
            value_loss = nn.MSELoss()(value_pred, targets)

            # Update with gradient clipping
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optimizer.step()


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_rl_agent(dim, n_episodes=500, horizon=5):
    """Train the RL agent on surrogate environment"""
    env = SurrogateConstrainedBOEnv(dim=dim, horizon=horizon)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=1e-4,  # Lower learning rate for stability
        gamma=0.95,
        clip_epsilon=0.2,
        epochs=5  # Fewer epochs per update
    )

    print(f"Training RL agent for {dim}D problems...")

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        trajectories = []
        episode_reward = 0

        for step in range(horizon):
            action, log_prob = agent.select_action(state)

            # Check for invalid action
            if np.isnan(action).any():
                print(f"Warning: NaN action at episode {episode}, using random action")
                action = np.random.uniform(0, 1, env.action_space.shape[0])
                log_prob = 0.0

            next_state, reward, done, _, _ = env.step(action)

            trajectories.append([state, action, log_prob, reward, next_state, done])
            episode_reward += reward
            state = next_state

            if done:
                break

        # Update agent only if we have valid trajectories
        if len(trajectories) > 0 and not np.isnan(episode_reward):
            agent.update(trajectories)

        episode_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            recent_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{n_episodes}, Avg Reward (last 50): {recent_reward:.2f}")

    print("Training complete!\n")
    return agent


# =============================================================================
# REAL EVALUATION ON COCO BENCHMARKS
# =============================================================================

class RealConstrainedBOEnv:
    """Wrapper for real COCO benchmark problems"""

    def __init__(self, coco_problem):
        self.problem = coco_problem
        self.dim = coco_problem.dimension
        self.lower_bounds = coco_problem.lower_bounds
        self.upper_bounds = coco_problem.upper_bounds

        # GPs for objective and constraints
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.2, nu=2.5)

        # Reduce evaluations in 40D case

        self.gp_objective = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5
        )
        self.gp_constraint = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5
        )

        # self.gp_objective = GaussianProcessRegressor(
        #     kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None
        # )
        # self.gp_constraint = GaussianProcessRegressor(
        #     kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None
        # )

        self.X_observed = []
        self.y_observed = []
        self.c_observed = []

    def normalize(self, x):
        """Normalize point to [0, 1]^D"""
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def denormalize(self, x):
        """Denormalize point from [0, 1]^D to original bounds"""
        return x * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def evaluate(self, x_normalized):
        """Evaluate point on real COCO function"""
        x = self.denormalize(x_normalized)
        objective = float(self.problem(x))

        # For BBOB-constrained, constraint info is in problem.constraint(x)
        if hasattr(self.problem, 'constraint'):
            c_vals = np.asarray(self.problem.constraint(x))
            # scalar constraint: max of all constraints (<= 0 means feasible)
            constraint = float(np.max(c_vals))
        else:
            # If for some reason no constraint is defined, treat as always feasible
            constraint = -1.0

        return objective, constraint

    def add_observation(self, x_normalized, y, c):
        """Add observation and refit GPs"""
        self.X_observed.append(x_normalized)
        self.y_observed.append(y)
        self.c_observed.append(c)

        X = np.array(self.X_observed)
        y_arr = np.array(self.y_observed)
        c_arr = np.array(self.c_observed)

        self.gp_objective.fit(X, y_arr)
        self.gp_constraint.fit(X, c_arr)

    def get_state(self):
        """Get state representation for RL agent"""
        if len(self.X_observed) == 0:
            # 10 summary stats + last_point (D) + best_point (D)
            return np.zeros(10 + 2 * self.dim, dtype=np.float32)

        y_arr = np.asarray(self.y_observed)
        c_arr = np.asarray(self.c_observed)

        feasible_mask = c_arr <= 0.0
        best_feasible = np.max(y_arr[feasible_mask]) if feasible_mask.any() else -np.inf

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr) + 1e-6)
        y_max = float(np.max(y_arr))

        n_feasible = int(np.sum(feasible_mask))
        feasible_ratio = n_feasible / len(c_arr)

        last_point = np.asarray(self.X_observed[-1], dtype=np.float32)

        # Best feasible point (only over feasible points)
        if feasible_mask.any():
            masked_y = np.where(feasible_mask, y_arr, -np.inf)
            best_idx = int(np.argmax(masked_y))
            best_point = np.asarray(self.X_observed[best_idx], dtype=np.float32)
        else:
            best_point = np.zeros(self.dim, dtype=np.float32)

        summary = np.array(
            [
                best_feasible,  # 0
                y_mean,  # 1
                y_std,  # 2
                y_max,  # 3
                feasible_ratio,  # 4
                1.0,  # 5 (dummy / budget flag)
                len(self.X_observed) / 100.0,  # 6 (scaled eval count)
                0.0, 0.0, 0.0  # 7–9 padding
            ],
            dtype=np.float32
        )

        state = np.concatenate(
            [summary, last_point.astype(np.float32), best_point.astype(np.float32)]
        )

        return state


def run_rl_bo_on_coco(agent, coco_problem, budget, n_initial=None):
    """Run trained RL agent on real COCO problem"""
    env = RealConstrainedBOEnv(coco_problem)
    dim = env.dim

    # default: 2 * D initial random samples
    n_initial = n_initial or (2 * dim)
    n_initial = int(min(n_initial, budget))

    # Initial random samples
    for _ in range(n_initial):
        x_norm = np.random.uniform(0.0, 1.0, dim)
        y, c = env.evaluate(x_norm)
        env.add_observation(x_norm, y, c)

    # RL-guided optimization
    remaining = max(0, budget - n_initial)
    for _ in range(remaining):
        state = env.get_state()
        action, _ = agent.select_action(state, deterministic=True)

        # Evaluate on real function
        y, c = env.evaluate(action)
        env.add_observation(action, y, c)

    # Return best feasible value found (larger = better in your current convention)
    y_arr = np.asarray(env.y_observed)
    c_arr = np.asarray(env.c_observed)
    feasible_mask = c_arr <= 0.0

    if feasible_mask.any():
        return float(np.max(y_arr[feasible_mask]))
    else:
        return float("-inf")



# =============================================================================
# MAIN BENCHMARKING SCRIPT
# =============================================================================

def main():
    """
    Run benchmarks on COCO constrained BBOB functions.

    Functions:   2, 4, 6, 50, 52, 54
    Instances:   0, 1, 2  (mapped to COCO instances 1, 2, 3)
    Dimensions:  2, 10
    Repetitions: 5 per (function, instance, dimension)

    Budget per run:
      - Minimum: 10 * D function evaluations
      - Using:   30 * D (if computationally feasible)
    """

    functions   = [2, 4, 6, 50, 52, 54]
    instances   = [0, 1, 2]    # assignment instances (we map to COCO by +1)
    dimensions  = [2, 10]
    repetitions = 5

    results = {}

    # Initialise COCO suite once (we pick problems by function / dim / instance)
    suite = cocoex.Suite("bbob-constrained", "", "")

    for dim in dimensions:
        print(f"\n{'=' * 60}")
        print(f"TRAINING AGENT FOR DIMENSION {dim}")
        print(f"{'=' * 60}\n")

        # Dynamic number of training episodes per dimension
        if dim == 2:
            n_episodes = 500
        elif dim == 10:
            n_episodes = 300
        else:
            n_episodes = 200

        # Train one PPO agent per dimension on the surrogate environment
        agent = train_rl_agent(dim=dim, n_episodes=n_episodes, horizon=5)

        # Budget settings
        min_budget = 10 * dim
        max_budget = 30 * dim
        budget = max_budget        # satisfies "≥ 10D" and uses 30D if possible
        print(f"Using budget {budget} evaluations per run (min allowed {min_budget}).")

        for func in functions:
            for inst in instances:
                coco_inst = inst + 1   # COCO instances start at 1

                print(f"\nEvaluating F{func}, Instance {inst} (COCO inst {coco_inst}), Dim {dim}")

                try:
                    problem = suite.get_problem_by_function_dimension_instance(
                        func, dim, coco_inst
                    )
                except Exception as e:
                    print(f"  Could not load problem F{func} (dim={dim}, inst={coco_inst}): {e}")
                    continue

                rep_results = []
                for rep in range(repetitions):
                    best = run_rl_bo_on_coco(agent, problem, budget)
                    rep_results.append(best)
                    print(f"  Rep {rep + 1}: {best:.4f}")

                key = f"F{func}_i{inst}_d{dim}"
                results[key] = {
                    "mean": float(np.mean(rep_results)),
                    "std":  float(np.std(rep_results)),
                    "all":  rep_results,
                }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, val in results.items():
        print(f"{key}: {val['mean']:.4f} ± {val['std']:.4f}")

    return results



if __name__ == "__main__":
    # For demonstration without actual COCO suite
    # print("RL-Enhanced Constrained Bayesian Optimization")
    # print("=" * 60)
    # print("\nDemo: Training on 2D problems")

    # Train agent on surrogate
    # agent_2d = train_rl_agent(dim=2, n_episodes=200, horizon=5)

    # print("\nAgent trained! Ready to use on real COCO benchmarks.")
    # print("\nTo run full benchmarks, ensure cocoex is installed:")
    # print("  pip install cocoex")
    # print("\nThen uncomment and run main()")

    # Uncomment to run full benchmarks:
    results = main()
