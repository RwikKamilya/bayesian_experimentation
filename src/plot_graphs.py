#!/usr/bin/env python3
"""
Quick plotting script for:
- Training logs: training_logs/training_dim{dim}.csv
- Benchmark logs: results_coco_transformer_vs_qlogei.csv

Creates PNGs in plots/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_LOG_DIR = Path("../training_logs")
BENCH_CSV = Path("../results_coco_transformer_vs_qlogei.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def plot_training(dim: int):
    csv_path = TRAIN_LOG_DIR / f"training_dim{dim}.csv"
    if not csv_path.exists():
        print(f"[plot_training] No training CSV for dim={dim} at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["epoch"], df["train_mse"], label="train MSE")
    plt.plot(df["epoch"], df["val_mse"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training loss (dim={dim})")
    plt.legend()
    out_path = PLOTS_DIR / f"training_dim{dim}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_training] Saved {out_path}")


def plot_benchmark():
    if not BENCH_CSV.exists():
        print(f"[plot_benchmark] {BENCH_CSV} not found.")
        return

    df = pd.read_csv(BENCH_CSV)

    for dim in sorted(df["dim"].unique()):
        df_d = df[df["dim"] == dim]
        methods = sorted(df_d["method"].unique())
        max_eval = df_d["eval"].max()

        plt.figure()
        for m in methods:
            df_m = df_d[df_d["method"] == m]
            # Aggregate over all functions / instances / repetitions
            grp = df_m.groupby("eval")["best_feasible"]
            mean = grp.mean()
            std = grp.std()

            evals = mean.index.values
            plt.plot(evals, mean.values, label=m)
            plt.fill_between(
                evals,
                (mean - std).values,
                (mean + std).values,
                alpha=0.2,
            )

        plt.xlabel("Evaluation")
        plt.ylabel("Best feasible (lower is better)")
        plt.title(f"COCO constrained suite – dim={dim}")
        plt.legend()
        out_path = PLOTS_DIR / f"benchmark_dim{dim}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plot_benchmark] Saved {out_path}")


def main():
    # training plots
    for dim in [2, 10]:
        plot_training(dim)

    # benchmark plots
    plot_benchmark()


if __name__ == "__main__":
    main()
