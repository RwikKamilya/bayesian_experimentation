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


TRAIN_LOG_DIR = Path("training_logs")
BENCH_CSV = Path("results/results_coco_transformer_vs_qlogei.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def plot_training(dim: int):
    csv_path = TRAIN_LOG_DIR / f"training_dim{dim}.csv"
    if not csv_path.exists():
        print(f"[plot_training] No training CSV for dim={dim} at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # ---------------- Linear scale plot ----------------
    plt.figure()
    plt.plot(df["epoch"], df["train_mse"], label="train MSE")
    plt.plot(df["epoch"], df["val_mse"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training loss (dim={dim}) – linear scale")
    plt.legend()
    out_path = PLOTS_DIR / f"training_dim{dim}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_training] Saved {out_path}")

    # ---------------- Log scale plot ----------------
    # Clip to avoid log(0); change eps if your losses get very small
    eps = 1e-10
    train_mse = np.clip(df["train_mse"].values, eps, None)
    val_mse = np.clip(df["val_mse"].values, eps, None)

    plt.figure()
    plt.plot(df["epoch"], train_mse, label="train MSE")
    plt.plot(df["epoch"], val_mse, label="val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title(f"Training loss (dim={dim}) – log scale")
    plt.legend()
    out_path_log = PLOTS_DIR / f"training_dim{dim}_log.png"
    plt.savefig(out_path_log, bbox_inches="tight")
    plt.close()
    print(f"[plot_training] Saved {out_path_log}")


def plot_benchmark():
    if not BENCH_CSV.exists():
        print(f"[plot_benchmark] {BENCH_CSV} not found.")
        return

    df = pd.read_csv(BENCH_CSV)

    # Ensure numeric, let pandas parse 'inf' → np.inf
    df["best_feasible"] = pd.to_numeric(df["best_feasible"], errors="coerce")

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    for dim in sorted(df["dim"].unique()):
        df_d = df[df["dim"] == dim]
        methods = sorted(df_d["method"].unique())

        # ---------- Plot mean best-feasible vs eval ----------
        plt.figure(figsize=(8, 5))

        for m in methods:
            df_m = df_d[df_d["method"] == m]

            # Group by eval
            grp = df_m.groupby("eval")["best_feasible"]

            # Replace inf by NaN for aggregation (we only average over finite values)
            mean = grp.apply(lambda s: s.replace([np.inf, -np.inf], np.nan).mean())
            std = grp.apply(lambda s: s.replace([np.inf, -np.inf], np.nan).std())

            evals = mean.index.values
            mean_vals = mean.values
            std_vals = std.values

            # Skip if everything is NaN (e.g. method never finds feasible points)
            if np.all(np.isnan(mean_vals)):
                print(f"[plot_benchmark] dim={dim}, method={m}: no finite values to plot.")
                continue

            # For plotting, it's fine to have some NaNs; matplotlib will break the line there.
            plt.plot(evals, mean_vals, label=m)
            plt.fill_between(
                evals,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2,
                )

        plt.xlabel("Evaluation")
        plt.ylabel("Best feasible objective (lower is better)")
        plt.title(f"COCO constrained suite – mean best-feasible vs budget (dim={dim})")
        plt.legend()
        plt.grid(alpha=0.3)
        out_path = PLOTS_DIR / f"benchmark_dim{dim}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plot_benchmark] Saved {out_path}")

        # ---------- OPTIONAL: Plot feasibility vs eval ----------
        # This is nice to "prove" that Transformer hurts feasibility.
        plt.figure(figsize=(8, 5))
        for m in methods:
            df_m = df_d[df_d["method"] == m]
            grp = df_m.groupby("eval")["best_feasible"]

            # Feasibility rate = fraction of finite values
            feas = grp.apply(lambda s: np.isfinite(s).mean())

            evals = feas.index.values
            feas_vals = feas.values

            plt.plot(evals, feas_vals, label=m)

        plt.xlabel("Evaluation")
        plt.ylabel("Feasibility rate")
        plt.ylim(0, 1.05)
        plt.title(f"COCO constrained suite – feasibility vs budget (dim={dim})")
        plt.legend()
        plt.grid(alpha=0.3)
        out_path_feas = PLOTS_DIR / f"benchmark_feasibility_dim{dim}.png"
        plt.savefig(out_path_feas, bbox_inches="tight")
        plt.close()
        print(f"[plot_benchmark] Saved {out_path_feas}")


def main():
    # training plots
    for dim in [2, 10]:
        plot_training(dim)

    # benchmark plots
    plot_benchmark()


if __name__ == "__main__":
    main()
