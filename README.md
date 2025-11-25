# Constrained Bayesian Optimization: RL-Enhanced vs. Baseline

This repository contains implementations for the Bayesian Optimization course practical assignment 2025.

## Overview

We implement and compare two approaches for constrained Bayesian Optimization:

1. **Baseline**: BoTorch constrained BO with qLogEI (batch size q=1)
   - Standard acquisition function approach
   - Based on the [BoTorch tutorial](https://botorch.org/docs/tutorials/closed_loop_botorch_only/)

2. **Novel Method**: RL-Enhanced Constrained Bayesian Optimization
   - Uses PPO (Proximal Policy Optimization) trained on GP surrogates
   - Incorporates lookahead planning for query point selection

## Project Structure

```
.
├── baseline_qlogei.py          # Baseline implementation (qLogEI with q=1)
├── bayesian_optimization.py    # RL-enhanced BO implementation
├── compare_methods.py          # Comparison and plotting script
├── closed_loop_botorch_only.ipynb  # BoTorch tutorial reference
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `torch` - PyTorch for neural networks and BoTorch
- `botorch` - Bayesian Optimization framework
- `gpytorch` - Gaussian Process library
- `cocoex` - COCO benchmark suite
- `numpy`, `scipy` - Numerical computing
- `matplotlib` - Plotting
- `gymnasium` - RL environment framework
- `scikit-learn` - Gaussian Processes for RL surrogate

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42

## Benchmarking Setup

As per assignment requirements, we test on:

- **Functions**: F2, F4, F6, F50, F52, F54 from COCO constrained BBOB
- **Instances**: 0, 1, 2
- **Dimensions**: 2, 10, 40
- **Repetitions**: 5 per instance
- **Budget**: 30*D evaluations (minimum 10*D as required)

## Usage

### 1. Run Baseline (qLogEI)

Run the baseline with default settings (dimensions 2, 10, 40):

```bash
python baseline_qlogei.py
```

Custom configuration:

```bash
python baseline_qlogei.py \
    --dimensions 2 10 \
    --functions 2 4 6 50 52 54 \
    --instances 0 1 2 \
    --repetitions 5 \
    --budget 30 \
    --verbose \
    --save baseline_results.npy
```

Arguments:
- `--dimensions`: Dimensions to test (default: [2, 10, 40])
- `--functions`: COCO function IDs (default: [2, 4, 6, 50, 52, 54])
- `--instances`: Instance IDs (default: [0, 1, 2])
- `--repetitions`: Number of repetitions (default: 5)
- `--budget`: Budget multiplier - total budget = budget * D (default: 30)
- `--verbose`: Print detailed progress
- `--save`: Output file for results (default: baseline_results.npy)

### 2. Run RL-Enhanced Method

Run the RL-enhanced approach:

```bash
python bayesian_optimization.py
```

Note: The RL method trains a separate agent for each dimension, which takes longer than the baseline.

### 3. Compare Methods

After running both methods, generate comparison plots and statistics:

```bash
python compare_methods.py \
    --baseline-results baseline_results.npy \
    --rl-results rl_results.npy \
    --save-dir plots
```

This will generate:
- Convergence plots for each problem
- Aggregate performance comparisons by dimension
- Statistical analysis (paired t-test)
- Results summary table

## Quick Start Example

For a quick test on a smaller problem set (2D only, fewer repetitions):

```bash
# Run baseline on 2D problems only
python baseline_qlogei.py \
    --dimensions 2 \
    --functions 2 4 \
    --repetitions 3 \
    --verbose

# Compare results
python compare_methods.py --plot-only
```

## Experimental Results

Results are saved in NumPy format (`.npy`) containing:
- `results`: Dictionary with mean, std, and all repetition values
- `histories`: Convergence history for each problem configuration

### Result Keys

Results are stored with keys in format: `F{func}_i{inst}_d{dim}`

Example: `F2_i0_d10` means Function 2, Instance 0, Dimension 10

### Accessing Results

```python
import numpy as np

# Load results
data = np.load('baseline_results.npy', allow_pickle=True).item()
results = data['results']
histories = data['histories']

# Get specific result
key = 'F2_i0_d10'
mean_value = results[key]['mean']
std_value = results[key]['std']
all_reps = results[key]['all']

# Get convergence history
convergence = histories[key]  # List of histories from each repetition
```

## Implementation Details

### Baseline (qLogEI)

The baseline follows the BoTorch tutorial approach:

1. **Model**: `ModelListGP` with separate `SingleTaskGP` for objective and constraint
2. **Acquisition Function**: `qLogExpectedImprovement` with q=1
3. **Optimization**: `optimize_acqf` with 10 restarts, 512 raw samples
4. **Noise Model**: Homoskedastic noise with σ = 0.25
5. **Input Normalization**: All inputs normalized to [0, 1]^D

Key features:
- Constraint handling: feasibility constraint c(x) ≤ 0
- MC sampling: 256 Sobol QMC samples
- GP fitting: MLE via L-BFGS
- Warm starting: GP state reused between iterations

### RL-Enhanced Method

The novel method uses reinforcement learning:

1. **Training Phase**:
   - Train PPO agent on GP surrogate environments
   - Agent learns to select points with good lookahead
   - Separate agent trained for each dimension

2. **Evaluation Phase**:
   - Use trained agent to select query points
   - Maintain GPs for objective and constraint
   - Agent sees summary statistics and GP predictions

Features:
- Horizon: 5-step lookahead during training
- Training episodes: 500 (2D), 300 (10D), 200 (40D)
- Reward: Based on feasibility and improvement
- Exploration: Uncertainty bonuses in reward function

## Troubleshooting

### COCO Installation Issues

If `cocoex` fails to install:

```bash
# Try installing from source
git clone https://github.com/numbbo/coco.git
cd coco
python do.py install-postprocessing
```

### Memory Issues (40D)

For 40D problems, reduce:
- Number of GP restarts (in kernel fitting)
- Number of MC samples
- Or run on GPU if available

### Numerical Stability

If you encounter GP fitting errors:
- Check for duplicate points
- Increase noise level (`NOISE_SE`)
- Use fixed kernel hyperparameters (set `optimizer=None`)

## Citations

### BoTorch Tutorial
```
@misc{botorch_tutorial,
  title={Closed-loop batch, constrained BO in BoTorch with qLogEI and qLogNEI},
  url={https://botorch.org/docs/tutorials/closed_loop_botorch_only/},
  publisher={BoTorch}
}
```

### COCO Benchmark
```
@inproceedings{hansen2016coco,
  title={COCO: A platform for comparing continuous optimizers in a black-box setting},
  author={Hansen, Nikolaus and Auger, Anne and Ros, Raymond and Mersmann, Olaf and Tu{\v{s}}ar, Tea and Brockhoff, Dimo},
  booktitle={Optimization Methods and Software},
  year={2016}
}
```

## Assignment Requirements Checklist

- [x] Implementation fully reproducible (fixed seeds)
- [x] README.md with installation and run instructions
- [x] requirements.txt for dependencies
- [x] Baseline: qLogEI with q=1 from BoTorch tutorial
- [x] Benchmarking on COCO constrained BBOB
- [x] Functions: F2, F4, F6, F50, F52, F54
- [x] Instances: 0, 1, 2
- [x] Repetitions: 5 per instance
- [x] Dimensions: 2, 10, 40
- [x] Budget: ≥ 10*D (using 30*D)
- [ ] Convergence plots (generated by compare_methods.py)
- [ ] Poster and presentation

## License

MIT License - Educational project for Bayesian Optimization Course 2025

## Authors

[Your group members here]

## Contact

[Your contact information here]
