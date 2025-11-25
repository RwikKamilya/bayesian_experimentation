"""
Comparison script for RL-enhanced BO vs qLogEI baseline

This script runs both methods and generates comparison plots including:
- Convergence plots
- Performance comparison tables
- Statistical analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import both implementations
from baseline_qlogei import run_baseline_experiments as run_qlogei
from bayesian_optimization import main as run_rl_bo


def plot_convergence(histories_baseline, histories_rl, save_dir="plots"):
    """
    Plot convergence curves comparing baseline and RL methods

    Args:
        histories_baseline: Dict of baseline convergence histories
        histories_rl: Dict of RL convergence histories
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)

    # Get all problem keys
    all_keys = set(histories_baseline.keys()) | set(histories_rl.keys())

    for key in sorted(all_keys):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot baseline
        if key in histories_baseline:
            baseline_data = np.array(histories_baseline[key])
            mean_baseline = baseline_data.mean(axis=0)
            std_baseline = baseline_data.std(axis=0)
            iterations = np.arange(len(mean_baseline))

            ax.plot(iterations, mean_baseline, 'b-', label='qLogEI Baseline (q=1)', linewidth=2)
            ax.fill_between(
                iterations,
                mean_baseline - std_baseline,
                mean_baseline + std_baseline,
                alpha=0.3,
                color='blue'
            )

        # Plot RL
        if key in histories_rl:
            rl_data = np.array(histories_rl[key])
            mean_rl = rl_data.mean(axis=0)
            std_rl = rl_data.std(axis=0)
            iterations = np.arange(len(mean_rl))

            ax.plot(iterations, mean_rl, 'r-', label='RL-Enhanced BO', linewidth=2)
            ax.fill_between(
                iterations,
                mean_rl - std_rl,
                mean_rl + std_rl,
                alpha=0.3,
                color='red'
            )

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Feasible Objective', fontsize=12)
        ax.set_title(f'Convergence: {key}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/convergence_{key}.png", dpi=300)
        plt.close()

    print(f"Convergence plots saved to {save_dir}/")


def plot_aggregate_performance(results_baseline, results_rl, save_dir="plots"):
    """
    Plot aggregate performance across all problems

    Args:
        results_baseline: Dict of baseline results
        results_rl: Dict of RL results
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)

    # Group by dimension
    dimensions = [2, 10, 40]

    for dim in dimensions:
        baseline_means = []
        baseline_stds = []
        rl_means = []
        rl_stds = []
        labels = []

        for key in sorted(results_baseline.keys()):
            if f"_d{dim}" in key:
                labels.append(key.replace(f"_d{dim}", ""))
                baseline_means.append(results_baseline[key]['mean'])
                baseline_stds.append(results_baseline[key]['std'])

                if key in results_rl:
                    rl_means.append(results_rl[key]['mean'])
                    rl_stds.append(results_rl[key]['std'])
                else:
                    rl_means.append(0)
                    rl_stds.append(0)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
               label='qLogEI Baseline', capsize=5)
        ax.bar(x + width/2, rl_means, width, yerr=rl_stds,
               label='RL-Enhanced BO', capsize=5)

        ax.set_xlabel('Problem', fontsize=12)
        ax.set_ylabel('Best Feasible Objective', fontsize=12)
        ax.set_title(f'Performance Comparison (Dimension {dim})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/comparison_d{dim}.png", dpi=300)
        plt.close()

    print(f"Comparison plots saved to {save_dir}/")


def generate_results_table(results_baseline, results_rl, save_file="results_table.txt"):
    """Generate a formatted results table"""
    with open(save_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTS COMPARISON: qLogEI Baseline vs RL-Enhanced BO\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Problem':<20} {'Baseline Mean':<15} {'Baseline Std':<15} "
                f"{'RL Mean':<15} {'RL Std':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")

        for key in sorted(results_baseline.keys()):
            baseline_mean = results_baseline[key]['mean']
            baseline_std = results_baseline[key]['std']

            if key in results_rl:
                rl_mean = results_rl[key]['mean']
                rl_std = results_rl[key]['std']
                improvement = ((rl_mean - baseline_mean) / abs(baseline_mean) * 100
                              if baseline_mean != 0 else 0)
            else:
                rl_mean = 0
                rl_std = 0
                improvement = 0

            f.write(f"{key:<20} {baseline_mean:>14.4f} {baseline_std:>14.4f} "
                   f"{rl_mean:>14.4f} {rl_std:>14.4f} {improvement:>13.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Results table saved to {save_file}")


def statistical_comparison(results_baseline, results_rl):
    """Perform statistical comparison between methods"""
    from scipy import stats

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (Paired t-test)")
    print("=" * 80)

    baseline_values = []
    rl_values = []

    for key in results_baseline.keys():
        if key in results_rl:
            baseline_values.extend(results_baseline[key]['all'])
            rl_values.extend(results_rl[key]['all'])

    if len(baseline_values) > 0 and len(rl_values) > 0:
        t_stat, p_value = stats.ttest_rel(rl_values, baseline_values)
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            if t_stat > 0:
                print("Result: RL-Enhanced BO is significantly BETTER (p < 0.05)")
            else:
                print("Result: RL-Enhanced BO is significantly WORSE (p < 0.05)")
        else:
            print("Result: No significant difference (p >= 0.05)")
    else:
        print("Insufficient data for statistical comparison")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare RL-BO and qLogEI baseline")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Run baseline experiments")
    parser.add_argument("--run-rl", action="store_true",
                        help="Run RL experiments")
    parser.add_argument("--baseline-results", type=str, default="baseline_results.npy",
                        help="Baseline results file")
    parser.add_argument("--rl-results", type=str, default="rl_results.npy",
                        help="RL results file")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 10],
                        help="Dimensions to test")
    parser.add_argument("--functions", type=int, nargs="+", default=[2, 4, 6, 50, 52, 54],
                        help="COCO function IDs")
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2],
                        help="Instance IDs")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of repetitions")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results")
    parser.add_argument("--save-dir", type=str, default="plots",
                        help="Directory to save plots")

    args = parser.parse_args()

    results_baseline = None
    results_rl = None
    histories_baseline = None
    histories_rl = None

    # Run experiments or load existing results
    if args.run_baseline:
        print("Running baseline experiments...")
        results_baseline, histories_baseline = run_qlogei(
            functions=args.functions,
            instances=args.instances,
            dimensions=args.dimensions,
            repetitions=args.repetitions,
            budget_multiplier=30,
            verbose=False
        )
        np.save(args.baseline_results, {
            "results": results_baseline,
            "histories": histories_baseline
        })
    elif Path(args.baseline_results).exists():
        print(f"Loading baseline results from {args.baseline_results}")
        data = np.load(args.baseline_results, allow_pickle=True).item()
        results_baseline = data['results']
        histories_baseline = data.get('histories', {})

    if args.run_rl:
        print("Running RL experiments...")
        # Note: This would need to be adapted based on your RL implementation
        # For now, we'll just note that this should be run separately
        print("Please run bayesian_optimization.py separately and save results")
    elif Path(args.rl_results).exists():
        print(f"Loading RL results from {args.rl_results}")
        data = np.load(args.rl_results, allow_pickle=True).item()
        results_rl = data['results']
        histories_rl = data.get('histories', {})

    # Generate comparison plots and tables
    if results_baseline is not None:
        print("\nGenerating results table for baseline...")
        if results_rl is not None:
            generate_results_table(results_baseline, results_rl)
            statistical_comparison(results_baseline, results_rl)
        else:
            print("RL results not available for comparison")

        if histories_baseline is not None and results_rl is not None and histories_rl is not None:
            print("\nGenerating convergence plots...")
            plot_convergence(histories_baseline, histories_rl, args.save_dir)
            plot_aggregate_performance(results_baseline, results_rl, args.save_dir)
        elif histories_baseline is not None:
            print("\nGenerating baseline-only plots...")
            # Could plot baseline only if needed


if __name__ == "__main__":
    main()
