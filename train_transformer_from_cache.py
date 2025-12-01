#!/usr/bin/env python3
"""
Train next-config Transformer from precomputed teacher sequences (GPU-only training).

Assumes teacher_seqs/teacher_seqs_dim{2,10}.npz exist.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_transformer import (
    NextConfigDataset,
    NextConfigTransformer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_next_config_model_for_dim_from_cache(
        dim: int,
        X_seqs: np.ndarray,
        y_seqs: np.ndarray,
        c_seqs: np.ndarray,
        T_max: int = 50,
        batch_size: int = 128,
        n_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 0,
        model_path: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[Train-from-cache dim={dim}] Using device: {DEVICE}")
    print(
        f"[Train-from-cache dim={dim}] X_seqs shape={X_seqs.shape}, "
        f"y_seqs shape={y_seqs.shape}, c_seqs shape={c_seqs.shape}"
    )

    # Extra NaN / Inf guard
    for name, arr in [("X_seqs", X_seqs), ("y_seqs", y_seqs), ("c_seqs", c_seqs)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf before training for dim={dim}")

    # 1) split episodes into train/val
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

    tokens, attn_mask, target_x = next(iter(train_loader))
    print("tokens finite?", torch.isfinite(tokens).all().item())
    print("target_x finite?", torch.isfinite(target_x).all().item())

    # 2) model + optimizer
    model = NextConfigTransformer(dim=dim, max_len=T_max).to(DEVICE)
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
            tokens = tokens.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            target_x = target_x.to(DEVICE)

            optimizer.zero_grad()
            pred_x = model(tokens, attn_mask)
            loss = criterion(pred_x, target_x)
            if torch.isnan(loss):
                print("[WARN] NaN loss encountered during training; skipping batch.")
                continue

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
                tokens = tokens.to(DEVICE)
                attn_mask = attn_mask.to(DEVICE)
                target_x = target_x.to(DEVICE)

                pred_x = model(tokens, attn_mask)
                loss = criterion(pred_x, target_x)
                if torch.isnan(loss):
                    print("[WARN] NaN loss encountered during validation; skipping batch.")
                    continue

                bs = tokens.size(0)
                val_loss += loss.item() * bs
                num_val += bs

        val_loss /= max(num_val, 1)

        print(
            f"[Epoch {epoch:03d} dim={dim}] "
            f"train MSE={train_loss:.4e} | val MSE={val_loss:.4e}"
        )

        if val_loss < best_val_loss and not math.isnan(val_loss):
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
        print(f"[Train-from-cache dim={dim}] Saved best model to: {model_path}")
    else:
        print(f"[Train-from-cache dim={dim}] WARNING: no valid best state recorded!")


def main():
    dims_to_train = [2, 10]
    # dims_to_train = [2]
    T_MAX = 50
    seq_dir = Path("teacher_seqs")

    for dim in dims_to_train:
        seq_path = seq_dir / f"teacher_seqs_dim{dim}.npz"
        print("=" * 80)
        print(f"[Main] Loading teacher sequences for dim={dim} from {seq_path}")
        print("=" * 80)

        data = np.load(seq_path)
        X_seqs = data["X_seqs"].astype(np.float32)
        y_seqs = data["y_seqs"].astype(np.float32)
        c_seqs = data["c_seqs"].astype(np.float32)

        mask = (
                np.isfinite(X_seqs).all(axis=(1, 2))
                & np.isfinite(y_seqs).all(axis=1)
                & np.isfinite(c_seqs).all(axis=1)
                )
        print(f"Keeping {mask.sum()} / {mask.shape[0]} episodes after NaN/Inf filter.")
        X_seqs = X_seqs[mask]
        y_seqs = y_seqs[mask]
        c_seqs = c_seqs[mask]


        train_next_config_model_for_dim_from_cache(
            dim=dim,
            X_seqs=X_seqs,
            y_seqs=y_seqs,
            c_seqs=c_seqs,
            T_max=T_MAX,
            batch_size=128,
            n_epochs=500,
            lr=1e-3,
            weight_decay=1e-5,
            seed=42 + dim,
            model_path=f"next_config_transformer_dim{dim}.pt",
        )


if __name__ == "__main__":
    main()
