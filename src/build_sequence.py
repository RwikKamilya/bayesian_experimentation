#!/usr/bin/env python3
"""
Build teacher sequences (COCO + GP cEI) once and save to disk.

This script is CPU-heavy and does NOT train the Transformer.
"""

import numpy as np
from pathlib import Path

from train_transformer import build_sequences_for_dim  # import from your file


def main():
    TARGET_FIDS = [2, 4, 6, 50, 52, 54]
    TARGET_INSTANCES = [1, 2, 3]
    dims_to_build = [2, 10]

    N_OFFLINE_POINTS = 10_000
    EPISODES_PER_PAIR = 50
    T_MAX = 50
    SEED_BASE = 42

    out_dir = Path("teacher_seqs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in dims_to_build:
        print("=" * 80)
        print(f"[Build] Starting teacher sequences for dim={dim}")
        print("=" * 80)

        X_seqs, y_seqs, c_seqs = build_sequences_for_dim(
            dim=dim,
            function_ids=TARGET_FIDS,
            instance_ids=TARGET_INSTANCES,
            n_offline_points=N_OFFLINE_POINTS,
            episodes_per_pair=EPISODES_PER_PAIR,
            T_max=T_MAX,
            seed=SEED_BASE + dim,
        )

        # Sanity checks before saving
        for name, arr in [("X_seqs", X_seqs), ("y_seqs", y_seqs), ("c_seqs", c_seqs)]:
            print(
                f"[Build dim={dim}] {name}: "
                f"shape={arr.shape}, "
                f"min={arr.min()}, max={arr.max()}, "
                f"finite={np.isfinite(arr).all()}"
            )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN/Inf for dim={dim}")

        out_path = out_dir / f"teacher_seqs_dim{dim}.npz"
        np.savez(out_path, X_seqs=X_seqs, y_seqs=y_seqs, c_seqs=c_seqs)
        print(f"[Build dim={dim}] Saved sequences to: {out_path}")


if __name__ == "__main__":
    main()
