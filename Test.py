import torch
import numpy as np
import pandas as pd
from pathlib import Path

from stocks_gpt import (
    GPTTimeSeriesModel,
    preprocess_dataframe,
    build_sequences_per_ticker,
    block_size,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = Path(r"")
MODEL_PATH = Path(r"")


def load_all_test_csvs(test_dir: Path) -> pd.DataFrame:
    """
    Load all *_LLM_Data.csv files from test_dir into a single DataFrame.

    Each file is assumed to be named like 'TICKER_LLM_Data.csv'.
    We add the 'ticker' column from the filename and sort by week_end_date.
    """
    csv_paths = sorted(test_dir.glob("*_LLM_Data.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_LLM_Data.csv files found in {test_dir}")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        ticker = p.name.split("_")[0]
        df["ticker"] = ticker
        df = df.sort_values("week_end_date")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def evaluate_on_test_dir(model_path: Path, test_dir: Path) -> None:
    print("\n=== Evaluating Model on Test Directory ===\n")
    print(f"Loading test CSVs from: {test_dir}")

    # ============================ Load and preprocess test data ============================
    df_test_raw = load_all_test_csvs(test_dir)

    # IMPORTANT: we call preprocess_dataframe ONCE and use its X_all for
    # both feature_dim detection and sequence building.
    X_all, y_all, tickers_all, df_clean = preprocess_dataframe(df_test_raw)

    feature_dim = X_all.shape[1]

    # ============================ Load model ============================
    model = GPTTimeSeriesModel(feature_dim=feature_dim)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # ============================ Build sequences ============================
    X_seq, y_seq, ticker_seq = build_sequences_per_ticker(
        X_all, y_all, tickers_all, block_size=block_size
    )

    n_tickers = len(np.unique(tickers_all))
    print(f"Preprocessed data: {len(df_clean)} rows, {n_tickers} tickers.")
    print(
        f"Built sequences: X_seq shape = {X_seq.shape}, "
        f"y_seq shape = {y_seq.shape}"
    )
    print(f"Test sequences: {X_seq.shape[0]} samples\n")

    # ============================ Run predictions ============================
    preds_list: list[np.ndarray] = []
    reals_list: list[np.ndarray] = []
    batch_size_eval = 256

    with torch.no_grad():
        for i in range(0, X_seq.shape[0], batch_size_eval):
            xb = X_seq[i : i + batch_size_eval].to(device)
            yb = y_seq[i : i + batch_size_eval].to(device)

            out, _ = model(xb)  # out: (B, 1)
            preds_list.append(out.cpu().numpy().reshape(-1))
            reals_list.append(yb.cpu().numpy().reshape(-1))

    preds = np.concatenate(preds_list)
    reals = np.concatenate(reals_list)

    # ============================ Metrics ============================
    print("--- Evaluation Metrics on Test Set ---")

    # 1. Directional accuracy (up vs down)
    # The model predicts a numeric weekly return (e.g., +0.01, -0.03, etc.)
    # The easiest and most important financial question is:
    #     "Does the model correctly predict UP or DOWN?"
    #
    # Convert predictions and real returns to binary labels:
    #       1 = price goes up (return > 0)
    #       0 = price goes down or is flat (return <= 0)
    pred_up = (preds > 0).astype(int)
    real_up = (reals > 0).astype(int)
    direction_acc = (pred_up == real_up).mean()
    print(f"Directional Accuracy: {direction_acc * 100:.2f}%")

    # 2. Pearson correlation (linear)
    # Measures how "linearly related" predictions are to real returns.
    # Correlation r is between -1 and +1:
    #     +1  = perfect positive linear relationship
    #      0  = no linear relationship
    #     -1  = perfect negative linear relationship
    #
    # If predictions or real returns have zero variance (std = 0),
    # correlation cannot be computed.
    if np.std(preds) == 0 or np.std(reals) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(preds, reals)[0, 1]
    print(f"Prediction–Real Correlation: {corr:.4f}")

    # 3. Mean Absolute 
    # MAE = average size of prediction error (absolute difference)
    # Example: preds = [0.05, -0.02], reals = [0.06, -0.01]
    #          abs errors = [0.01, 0.01] -> MAE = 0.01
    #
    # This is a measure of HOW FAR the prediction is from reality.
    # For weekly returns, MAE of 0.015 means 1.5% average error.
    mae = np.mean(np.abs(preds - reals))
    print(f"MAE: {mae:.5f}")

    # 4. Ranking performance (top/bottom 10 by predicted return)
    # The model outputs one number per stock per week.
    # We sort stocks by predicted returns (highest -> lowest).
    #
    # The idea:
    #   If the model is good, the stocks it predicts as BEST should
    #   have higher REAL returns. And the worst should have lower REAL returns.
    idx = np.argsort(preds)[::-1]  # descending
    if len(reals) >= 10:
        top10_mean = reals[idx[:10]].mean()
        bottom10_mean = reals[idx[-10:]].mean()
    else:
        top10_mean = np.nan
        bottom10_mean = np.nan

    print("\nRanking-based metrics:")
    print(f"Top-10 predicted avg return:    {top10_mean * 100:.2f}%")
    print(f"Bottom-10 predicted avg return: {bottom10_mean * 100:.2f}%")

    if not np.isnan(top10_mean) and not np.isnan(bottom10_mean):
        spread = (top10_mean - bottom10_mean) * 100
        print(f"Long–Short Spread:              {spread:.2f}%")

    print("\n=== Evaluation complete ===\n")


if __name__ == "__main__":
    evaluate_on_test_dir(MODEL_PATH, TEST_DIR)
