#!/usr/bin/env python3
"""
Train NextConfigTransformer from cached teacher sequences.

- Loads teacher_seqs/teacher_seqs_dim{dim}.npz
  with arrays:
    X_seqs: [num_episodes, T_max, dim]
    y_seqs: [num_episodes, T_max]
    c_seqs: [num_episodes, T_max]

  X_seqs should already be normalised into [0,1]^dim (unit hypercube).
  y_seqs and c_seqs are raw objective / aggregated constraint values.

- Trains a Transformer to predict the next configuration given the sequence.

- Logs train/val MSE for each epoch to:
    training_logs/training_dim{dim}.csv

- Saves the best checkpoint to:
    next_config_transformer_dim{dim}.pt

  The checkpoint contains:
    - dim, T_max
    - model_state_dict
    - y_mean, y_std, c_mean, c_std
      (for consistent normalisation in the benchmark)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

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
            pred_x: [batch, dim]  (next config in normalized [0,1]^dim)
        """
        B, L, _ = tokens.shape
        x = self.input_proj(tokens)          # [B, L, d_model]
        x = self.pos_enc(x)                  # [B, L, d_model]

        # Transformer expects [L, B, d_model]
        x = x.transpose(0, 1)                # [L, B, d_model]

        # key_padding_mask: True at PAD positions
        key_padding_mask = (attn_mask == 0)  # [B, L]
        enc_out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [L, B, d_model]

        enc_out = enc_out.transpose(0, 1)    # [B, L, d_model]

        # get last *real* token hidden state for each batch element
        lengths = attn_mask.sum(dim=1).long() - 1  # [B]
        lengths = torch.clamp(lengths, min=0)

        idx = lengths.view(B, 1, 1).expand(-1, 1, enc_out.size(-1))  # [B, 1, d_model]
        last_hidden = enc_out.gather(1, idx).squeeze(1)  # [B, d_model]

        pred_x = self.head(last_hidden)  # [B, dim]
        return pred_x


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TeacherSequenceDataset(Dataset):
    """
    Each episode provides a full sequence of tokens:
        (x_0, y_0, c_0), ..., (x_{T-1}, y_{T-1}, c_{T-1})
    and the target is the last x_{T-1}.
    """

    def __init__(self, X_seqs: np.ndarray, y_seqs: np.ndarray, c_seqs: np.ndarray):
        """
        X_seqs: [N, T, dim]  (normalised into [0,1])
        y_seqs: [N, T]       (raw objective)
        c_seqs: [N, T]       (raw aggregated constraint)
        """
        assert X_seqs.shape[0] == y_seqs.shape[0] == c_seqs.shape[0]
        assert X_seqs.shape[1] == y_seqs.shape[1] == c_seqs.shape[1]
        self.X_seqs = X_seqs.astype(np.float32)
        self.y_seqs = y_seqs.astype(np.float32)
        self.c_seqs = c_seqs.astype(np.float32)
        self.N, self.T, self.dim = self.X_seqs.shape

        # --- Normalise y and c for stability (per dim & T) ---
        # Standardise then clip to [-5,5] to avoid outliers.
        self.y_mean = float(self.y_seqs.mean())
        self.y_std = float(self.y_seqs.std() + 1e-8)
        self.c_mean = float(self.c_seqs.mean())
        self.c_std = float(self.c_seqs.std() + 1e-8)

        self.y_seqs = (self.y_seqs - self.y_mean) / self.y_std
        self.c_seqs = (self.c_seqs - self.c_mean) / self.c_std

        self.y_seqs = np.clip(self.y_seqs, -5.0, 5.0)
        self.c_seqs = np.clip(self.c_seqs, -5.0, 5.0)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x_seq = self.X_seqs[idx]          # [T, dim], already in [0,1]
        y_seq = self.y_seqs[idx]          # [T], normalised and clipped
        c_seq = self.c_seqs[idx]          # [T], normalised and clipped

        tokens = np.concatenate(
            [x_seq, y_seq[:, None], c_seq[:, None]],
            axis=-1
        ).astype(np.float32)              # [T, dim+2]

        # No padding within cached sequences: all length=T
        attn_mask = np.ones(self.T, dtype=np.float32)

        target = x_seq[-1].astype(np.float32)  # predict final config

        return (
            torch.from_numpy(tokens),       # [T, dim+2]
            torch.from_numpy(attn_mask),    # [T]
            torch.from_numpy(target),       # [dim]
        )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_next_config_model_for_dim_from_cache(
        dim: int,
        X_seqs: np.ndarray,
        y_seqs: np.ndarray,
        c_seqs: np.ndarray,
        T_max: int,
        batch_size: int = 128,
        n_epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 42,
        model_path: Path | None = None,
        log_csv_path: Path | None = None,
        device: str | torch.device = "cuda",
):
    """
    Train transformer for a given dimension from cached teacher sequences.

    Saves:
      - best checkpoint (lowest val MSE)
      - CSV log with columns [epoch, train_mse, val_mse]
    """
    set_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Basic sanity check on shapes
    assert X_seqs.ndim == 3 and X_seqs.shape[2] == dim
    assert y_seqs.shape[:2] == X_seqs.shape[:2]
    assert c_seqs.shape[:2] == X_seqs.shape[:2]
    N, T, dim_check = X_seqs.shape
    if T_max is not None and T != T_max:
        print(f"[WARN] Teacher sequences T={T} but T_max={T_max}. Using T={T} from data.")
        T_max = T

    # Construct dataset (also computes and applies y/c normalisation)
    dataset = TeacherSequenceDataset(X_seqs, y_seqs, c_seqs)
    y_mean, y_std = dataset.y_mean, dataset.y_std
    c_mean, c_std = dataset.c_mean, dataset.c_std

    # Train/val split
    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)
    split = int(0.8 * len(dataset))
    train_idxs = idxs[:split]
    val_idxs = idxs[split:]

    train_subset = torch.utils.data.Subset(dataset, train_idxs)
    val_subset = torch.utils.data.Subset(dataset, val_idxs)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Model
    model = NextConfigTransformer(dim=dim, max_len=T_max)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Logging
    if model_path is None:
        model_path = Path(f"next_config_transformer_dim{dim}.pt")
    model_path = Path(model_path)

    if log_csv_path is None:
        log_csv_path = Path("training_logs") / f"training_dim{dim}.csv"
    log_csv_path = Path(log_csv_path)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with log_csv_path.open("w") as f_log:
        f_log.write("epoch,train_mse,val_mse\n")

        best_val = float("inf")
        best_state = None

        print(model)
        print(f"[Train-from-cache dim={dim}] Using device: {device}")
        print(
            f"[Train-from-cache dim={dim}] X_seqs shape={X_seqs.shape}, "
            f"y_seqs shape={y_seqs.shape}, c_seqs shape={c_seqs.shape}"
        )
        print(
            f"[Train-from-cache dim={dim}] y_mean={y_mean:.3e}, y_std={y_std:.3e}, "
            f"c_mean={c_mean:.3e}, c_std={c_std:.3e}"
        )

        for epoch in range(1, n_epochs + 1):
            # ---- Train ----
            model.train()
            train_losses = []
            for tokens, attn_mask, target_x in train_loader:
                tokens = tokens.to(device)          # [B, T, dim+2]
                attn_mask = attn_mask.to(device)    # [B, T]
                target_x = target_x.to(device)      # [B, dim]

                optimizer.zero_grad()
                pred_x = model(tokens, attn_mask)   # [B, dim]
                loss = criterion(pred_x, target_x)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                loss.backward()
                # Gradient clipping for extra safety
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            train_mse = float(np.mean(train_losses))

            # ---- Validation ----
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tokens, attn_mask, target_x in val_loader:
                    tokens = tokens.to(device)
                    attn_mask = attn_mask.to(device)
                    target_x = target_x.to(device)
                    pred_x = model(tokens, attn_mask)
                    loss = criterion(pred_x, target_x)
                    val_losses.append(loss.item())

            val_mse = float(np.mean(val_losses))

            # Logging
            print(
                f"[Epoch {epoch:03d} dim={dim}] train MSE={train_mse:.4e} | "
                f"val MSE={val_mse:.4e}"
            )
            f_log.write(f"{epoch},{train_mse},{val_mse}\n")
            f_log.flush()

            # Track best
            if val_mse < best_val:
                best_val = val_mse
                best_state = {
                    "dim": dim,
                    "T_max": T_max,
                    "model_state_dict": model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "c_mean": c_mean,
                    "c_std": c_std,
                }

        # Save best model
        if best_state is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, model_path)
            print(
                f"[Train-from-cache dim={dim}] Saved best model "
                f"(val MSE={best_val:.4e}) to {model_path}"
            )
        else:
            print(f"[Train-from-cache dim={dim}] WARNING: no best_state saved!")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 10],
        help="Dimensions to train transformer on (must have teacher_seqs_dim{d}.npz).",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="teacher_seqs",
        help="Directory where teacher_seqs_dim{dim}.npz live.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs per dimension.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=50,
        help="Maximum sequence length; should match what was used to build teacher sequences.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed base; actual seed is seed + dim.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="training_logs",
        help="Where to store training CSV logs.",
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    log_dir = Path(args.log_dir)

    for dim in args.dims:
        seq_path = seq_dir / f"teacher_seqs_dim{dim}.npz"
        print("=" * 80)
        print(f"[Main] Loading teacher sequences for dim={dim} from {seq_path}")
        print("=" * 80)
        if not seq_path.exists():
            print(f"[Main] Skipping dim={dim}: {seq_path} does not exist.")
            continue

        data = np.load(seq_path)
        X_seqs = data["X_seqs"]      # [N, T, dim]
        y_seqs = data["y_seqs"]      # [N, T]
        c_seqs = data["c_seqs"]      # [N, T]

        # Nan/Inf filter at episode level
        finite_mask = (
                np.isfinite(X_seqs).all(axis=(1, 2))
                & np.isfinite(y_seqs).all(axis=1)
                & np.isfinite(c_seqs).all(axis=1)
        )
        kept = finite_mask.sum()
        total = len(finite_mask)
        print(f"Keeping {kept} / {total} episodes after NaN/Inf filter.")
        X_seqs = X_seqs[finite_mask]
        y_seqs = y_seqs[finite_mask]
        c_seqs = c_seqs[finite_mask]

        # Quick sanity check
        tokens = np.concatenate(
            [X_seqs, y_seqs[..., None], c_seqs[..., None]],
            axis=-1,
        )
        print("tokens finite?", np.isfinite(tokens).all())
        print("target_x finite?", np.isfinite(X_seqs[:, -1, :]).all())

        train_next_config_model_for_dim_from_cache(
            dim=dim,
            X_seqs=X_seqs,
            y_seqs=y_seqs,
            c_seqs=c_seqs,
            T_max=args.T_max,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + dim,
            model_path=Path(f"next_config_transformer_dim{dim}.pt"),
            log_csv_path=log_dir / f"training_dim{dim}.csv",
        )


if __name__ == "__main__":
    main()
