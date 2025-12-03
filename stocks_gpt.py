
"""
Train a GPT-style Transformer on weekly stock features
to predict next-week return (ret_1w).

- Reads all CSV files: root/data/llm_out/[TICKER]_LLM_Data.csv
- Builds time-series windows per ticker
- Trains a Transformer (adapted from Karpathy's nanoGPT)
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
#                        Hyperparameters & Paths
# =====================================================================

# Training hyperparameters
batch_size    = 64     # how many independent sequences in one batch
block_size    = 24     # how many weeks of history per sample (sequence length)
max_iters     = 6000   # training iterations
eval_interval = 250    # how often we print train/val loss
learning_rate = 3e-4
eval_iters    = 100
n_embd        = 192    # model (embedding) dimension
n_head        = 6      # attention heads
n_layer       = 6      # Transformer blocks
dropout       = 0.2
train_split_ratio = 0.85  # per-ticker time-based split: 80% train, 20% val

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Data path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "llm_out"
SECTOR_MAP_DIR = ROOT / "Params"
SECTOR_MAP_PATH = SECTOR_MAP_DIR / "sector_id_map.json"

# =====================================================================
#                             Data Loading
# =====================================================================


def save_sector_map(sector_to_id: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sector_to_id, f, indent=4)

def load_sector_map(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None  


def load_all_ticker_data(data_dir: Path) -> pd.DataFrame:
    """
    Load all [TICKER]_LLM_Data.csv files from data_dir into a single DataFrame.

    Assumptions:
    - Files are named like "AAPL_LLM_Data.csv"
    - Each file contains:
        sector, week_end_date, weekly_avg_close, ...
    - We add a 'ticker' column based on the filename
    """
    csv_files = list(data_dir.glob("*_LLM_Data.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_LLM_Data.csv files found in {data_dir}")

    dfs: List[pd.DataFrame] = []

    for file in csv_files:
        # Extract ticker symbol from filename
        # e.g. "AAPL_LLM_Data.csv" -> "AAPL"
        ticker = file.name.split("_")[0]

        df = pd.read_csv(file)

        # Enforce expected columns (sanity check / helpful error)
        if "week_end_date" not in df.columns:
            raise ValueError(f"'week_end_date' column missing in {file}")
        if "sector" not in df.columns:
            raise ValueError(f"'sector' column missing in {file}")

        df["ticker"] = ticker
        # Sort by date to ensure chronological order
        df = df.sort_values("week_end_date")
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


# =====================================================================
#                           Preprocessing
# =====================================================================

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert raw DataFrame into model-ready arrays:

    X_all: (N_rows, D_features) float32
    y_all: (N_rows,) float32          next-week ret_1w
    tickers_all: (N_rows,)
    df_clean: aligned DataFrame
    """
    df = df.copy()
    
    # ============================ STANDARDIZED SECTOR ID ============================
    # From string sector to integer codes

    # Load existing mapping if available
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    sector_map = load_sector_map(SECTOR_MAP_PATH)

    if sector_map is None:
        # First run -> create mapping from scratch
        unique_sectors = sorted(df["sector"].unique().tolist())  # all strings now
        sector_map = {sector: i for i, sector in enumerate(unique_sectors)}
        save_sector_map(sector_map, SECTOR_MAP_PATH)
        print(f"Created new sector_id map with {len(sector_map)} sectors:")
        print(sector_map)
    else:
        print(f"Loaded existing sector_id map with {len(sector_map)} sectors.")

        # Find any NEW sector strings not yet in the map
        current_sectors = set(df["sector"].unique().tolist())
        known_sectors   = set(sector_map.keys())
        new_sectors     = sorted(current_sectors - known_sectors)

        if new_sectors:
            print("Found new sectors, adding:", new_sectors)
            next_id = max(sector_map.values()) + 1
            for s in new_sectors:
                sector_map[s] = next_id
                next_id += 1
            save_sector_map(sector_map, SECTOR_MAP_PATH)
            print("Updated sector_id map saved.")

    # Map all sectors to IDs (includes "Unknown" and any new ones)
    df["sector_id"] = df["sector"].map(sector_map).astype(int)

    tickers_all = df["ticker"].values

    # =============================== Target ===============================
    if "ret_1w" not in df.columns:
        raise ValueError("Expected column 'ret_1w' for the prediction target.")

    # next-week target
    # df["target_ret_1w_next"] = df["ret_1w"].shift(-1)
    df["target_ret_1w_next"] = (
    df.groupby("ticker")["ret_1w"].shift(-1)
    )

    # =============================== Numeric conversion ===============================
    non_numeric_cols = ["ticker", "sector", "week_end_date"]
    numeric_candidate_cols = [c for c in df.columns if c not in non_numeric_cols]

    for c in numeric_candidate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =============================== Clip obviously insane values ===============================
    # Tune these thresholds later
    big_cols = [
        "revenue_ttm", "fcf_ttm", "net_income_ttm",
        "market_cap"
    ]
    for c in big_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=-1e12, upper=1e12)

    # =============================== Feature set ===============================
    drop_cols = [
        "ticker",                   #grouping key only
        "sector",                   #use sector_id instead
        "week_end_date",            #rely on ordering
        "target_ret_1w_next",       #goes into y_all, not X_all
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # =============================== Build X_all / y_all ===============================
    X_all = df[feature_cols].to_numpy(dtype=np.float64)             #Features
    y_all = df["target_ret_1w_next"].to_numpy(dtype=np.float64)     #Target

    # Drop rows where target is NaN/Inf
    target_mask = np.isfinite(y_all)
    X_all = X_all[target_mask]
    y_all = y_all[target_mask]
    tickers_all = tickers_all[target_mask]
    df = df.iloc[target_mask.nonzero()[0]].reset_index(drop=True)

    # =============================== Inspect target outliers ===============================
    print("Target ret_1w stats BEFORE clipping:")
    q = np.percentile(y_all, [0, 1, 5, 50, 95, 99, 100])
    print("  min,1,5,50,95,99,max =", q)

    # Clip target to reasonable range, e.g. [-0.5, 0.5] weekly
    # (tighten later)
    y_all = np.clip(y_all, -0.5, 0.5)

    # =============================== Handle NaNs in features ===============================
    nan_counts = np.isnan(X_all).sum(axis=0)
    if np.any(nan_counts > 0):
        print("NaNs per feature BEFORE filling:")
        for col, cnt in zip(feature_cols, nan_counts):
            if cnt > 0:
                print(f"  {col}: {cnt}")

    # Means in float64; guard against all-NaN columns & infinities
    col_means = np.nanmean(X_all, axis=0)
    col_means = np.where(~np.isfinite(col_means), 0.0, col_means)

    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(col_means, inds[1])

    # Sanity check: all finite now
    assert np.isfinite(X_all).all(), "Non-finite in X_all after fill!"
    assert np.isfinite(y_all).all(), "Non-finite in y_all after clipping!"

    # =============================== Standardize features ===============================
    means = X_all.mean(axis=0, keepdims=True)
    stds = X_all.std(axis=0, keepdims=True) + 1e-8  #avoid division by zero

    # Save/overwrite the scaler so inference can use it later
    FEATURE_SCALER_DIR = ROOT / "Params"
    FEATURE_SCALER_PATH = FEATURE_SCALER_DIR / "feature_scaler.json"
    FEATURE_SCALER_DIR.mkdir(parents=True, exist_ok=True)

    scaler = {
        "means": means.flatten().tolist(),
        "stds":  stds.flatten().tolist(),
    }

    with open(FEATURE_SCALER_PATH, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=4)

    X_all = (X_all - means) / stds

    print(f"Preprocessed data: {X_all.shape[0]} rows, {X_all.shape[1]} features")

    # Finally cast to float32 for PyTorch
    X_all = X_all.astype("float32")
    y_all = y_all.astype("float32")

    df_clean = df
    return X_all, y_all, tickers_all, df_clean


# =====================================================================
#                       Build Time-Series Sequences
# =====================================================================

def build_sequences_per_ticker(                                                                                                                     # Karpathy’s get_batch:
    X_all: np.ndarray,                                                                                                                              # ix = torch.randint(len(data) - block_size, (batch_size,))
    y_all: np.ndarray,                                                                                                                              # x = torch.stack([data[i:i+block_size] for i in ix])
    tickers_all: np.ndarray,                                                                                                                        # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Build Transformer-ready sequences of length 'block_size' per ticker.

    For each ticker:
        - Take its rows in chronological order
        - Slide a window of size 'block_size' over X
        - For each window, target is y at the *next* step after the window

    Returns:
        X_seq: (N_samples, block_size, D_features) float32
        y_seq: (N_samples,) float32
        ticker_seq: list of ticker for each sample (same length as X_seq)
    """
    X_seq_list: List[np.ndarray] = []
    y_seq_list: List[float] = []
    ticker_seq: List[str] = []

    unique_tickers = np.unique(tickers_all)
    print(f"Building sequences for {len(unique_tickers)} tickers...")

    for ticker in unique_tickers:                           #Select rows belonging to a single ticker via idx
        idx = np.where(tickers_all == ticker)[0]            
        X_t = X_all[idx]                                    #X_t is that ticker’s feature history in chronological order
        y_t = y_all[idx]                                    #y_all[t] = ret_1w of week t+1

        if len(X_t) <= block_size:
            # Not enough history for this ticker; skip it
            continue

        # Slide window: [i : i+block_size] for features, target at i+block_size = "For each window of block_size weeks, predict the target at the next time step."
        for i in range(len(X_t) - block_size+1):
            X_window = X_t[i:i + block_size]      # shape (block_size, D)
            y_target = y_t[i + block_size-1]      # scalar

            X_seq_list.append(X_window)
            y_seq_list.append(y_target)
            ticker_seq.append(ticker)

    X_seq = torch.tensor(np.stack(X_seq_list), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq_list), dtype=torch.float32)

    print(f"Built sequences: X_seq shape = {X_seq.shape}, y_seq shape = {y_seq.shape}")
    return X_seq, y_seq, ticker_seq


# =====================================================================
#                     Train / Validation Split (Time-based)
# =====================================================================

def split_train_val_time_based(
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    ticker_seq: List[str],
    train_ratio: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Time-based split within each ticker.

    For each ticker:
        - Group its samples
        - Use first train_ratio fraction as train
        - Last part as validation

    Returns:
        X_train, y_train, X_val, y_val as tensors
    """
    ticker_seq = np.array(ticker_seq)
    X_train_list: List[torch.Tensor] = []
    y_train_list: List[torch.Tensor] = []
    X_val_list: List[torch.Tensor] = []
    y_val_list: List[torch.Tensor] = []

    unique_tickers = np.unique(ticker_seq)
    for ticker in unique_tickers:
        idx = np.where(ticker_seq == ticker)[0]     #positions (indices in X_seq) that belong to a single ticker
        if len(idx) == 0:
            continue

        split_idx = int(len(idx) * train_ratio)     #Compute per-ticker train/val split
        if split_idx == 0 or split_idx == len(idx):
            # Not enough samples to split; skip or handle differently if needed
            continue

        train_idx = idx[:split_idx]
        val_idx = idx[split_idx:]

        X_train_list.append(X_seq[train_idx])
        y_train_list.append(y_seq[train_idx])
        X_val_list.append(X_seq[val_idx])
        y_val_list.append(y_seq[val_idx])

    X_train = torch.cat(X_train_list, dim=0)        #All per-ticker train chunks are concatenated into a single train set
    y_train = torch.cat(y_train_list, dim=0)
    X_val = torch.cat(X_val_list, dim=0)            #Same for val
    y_val = torch.cat(y_val_list, dim=0)

    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    return X_train, y_train, X_val, y_val


# =====================================================================
#                          Data Loader Function
# =====================================================================

def get_batch(
    split: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch from train or val set.

    Returns:
        x: (batch_size, block_size, D_features)
        y: (batch_size,)
    """
    if split == "train":
        X = X_train
        y = y_train
    else:
        X = X_val
        y = y_val

    idx = torch.randint(0, X.shape[0], (batch_size,))
    xb = X[idx].to(device)
    yb = y[idx].to(device)
    return xb, yb


# =====================================================================
#                       Transformer Model Components
# =====================================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask so attention can't look into the future                           A lower-triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # Compute attention weights
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Aggregate values
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple MLP: Linear -> ReLU -> Linear with dropout."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: self-attention + feed-forward."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))   # residual
        x = x + self.ffwd(self.ln2(x)) # residual
        return x


# =====================================================================
#                    Time-Series GPT-style Model
# =====================================================================

class GPTTimeSeriesModel(nn.Module):
    """
    GPT-like Transformer for numeric time-series.

    Input:
        x: (B, T, D_features) float32

    Output:
        preds: (B, 1) predicted next-week ret_1w
    """

    def __init__(self, feature_dim: int, out_dim: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim

        # Project numeric features to embedding dimension
        # Stock data is numeric, not discrete tokens like text.
        self.input_linear = nn.Linear(feature_dim, n_embd)

        # Positional embeddings over time steps
        # We must inject information about *which week* each row
        # corresponds to. 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        # This is the actual GPT core
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

        # Regression head: predict out_dim from the last time step
        self.head = nn.Linear(n_embd, out_dim)

        # Standard GPT initialization (normal distribution)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """
        x: (B, T, D_features)
        targets: (B,) or (B, 1) or None
        """
        B, T, D = x.shape
        assert T <= block_size, "Sequence length exceeds block_size"

        # Project numeric input features
        tok_emb = self.input_linear(x)  # (B,T,n_embd)

        # Positional embeddings (same for all batches)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding_table(pos)  # (T,n_embd)

        # Combine token + position
        h = tok_emb + pos_emb  # (B,T,n_embd)

        # Transformer blocks
        h = self.blocks(h)
        h = self.ln_f(h)       # (B,T,n_embd)

        # Use only the last time step for prediction
        h_last = h[:, -1, :]   # (B,n_embd)
        preds = self.head(h_last)  # (B,out_dim)

        loss = None
        if targets is not None:
            targets = targets.view(B, self.out_dim)
            loss = F.mse_loss(preds, targets)

        return preds, loss


# =====================================================================
#                       Loss Estimation Helper
# =====================================================================

@torch.no_grad()
def estimate_loss(
    model: GPTTimeSeriesModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict:
    """Compute mean train/val loss over eval_iters batches."""

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, X_train, y_train, X_val, y_val)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

"""
Assume loss = 0.0025
loss = Mean-Squared-Error = 0.0025
RMSE = sqrt(0.0025) = 0.05 = 5%
anything below 0.0017 should be good
====================================

If both go down = learning is happening
If train goes down but val does not = overfitting
If val goes up sharply = model memorizing noise
"""


# =====================================================================
#                               Main
# =====================================================================

def main():
    # ============================ Load & preprocess data ============================
    print("Loading CSV files from:", DATA_DIR)
    df = load_all_ticker_data(DATA_DIR)

    print("Preprocessing DataFrame...")
    X_all, y_all, tickers_all, df_clean = preprocess_dataframe(df)

    feature_dim = X_all.shape[1]

    # ============================ Build sequences per ticker ============================
    X_seq, y_seq, ticker_seq = build_sequences_per_ticker(
        X_all, y_all, tickers_all, block_size=block_size
    )

    # ============================ Train / validation split ============================
    X_train, y_train, X_val, y_val = split_train_val_time_based(
        X_seq, y_seq, ticker_seq, train_ratio=train_split_ratio
    )

    # ============================ Create model ============================
    model = GPTTimeSeriesModel(feature_dim=feature_dim, out_dim=1)
    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ============================ Training loop ============================
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, X_train, y_train, X_val, y_val)
            print(
                f"Step {iter}: "
                f"train loss {losses['train']:.6f}, "
                f"val loss {losses['val']:.6f}"
            )

        # Sample a batch of data
        xb, yb = get_batch("train", X_train, y_train, X_val, y_val)

        # Forward pass & backprop
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # (Optional) Save model
    out_path = ROOT / "models" / "gpt_timeseries.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Training complete. Model saved to {out_path}")


if __name__ == "__main__":
    main()
