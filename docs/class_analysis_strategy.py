import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from pprint import pprint

def create_windows(X, y=None, window_size=10, step=1):
    """
    Slide windows over X (n_samples, n_features) → X_win (n_windows, window_size*n_features).
    If y is provided, also return y_win (n_windows,).
    """
    n_samples, _ = X.shape
    Xw, yw = [], []
    for i in range(0, n_samples - window_size + 1, step):
        Xw.append(X[i : i + window_size].reshape(-1))
        if y is not None:
            yw.append(y[i + window_size - 1])
    return np.stack(Xw, axis=0), (np.array(yw) if y is not None else None)

def _map_power(props, gamma):
    p = np.power(props, gamma)
    return p / p.sum()

def _map_softmax(props, T):
    logp = np.log(props + 1e-12) / T
    q = np.exp(logp)
    return q / q.sum()

def _map_linear(props, alpha):
    K = len(props)
    return (1 - alpha) * props + alpha * (1.0 / K)

def window_and_balance(X, y, window_size=10, step_size=1,
                       strategy="power", gamma=0.7, T=2.0, alpha=0.2,
                       sigma=0.01, random_state=42):
    """
    Window slicing + distribution mapping + resampling.
    strategy: "power"|"softmax"|"linear"
    gamma/T/alpha: mapping hyperparameters
    sigma: jitter noise std for oversampling minority
    Returns X_res, y_res, summary_df with columns:
      "windowed_count", "resampled_count"
    """
    rng = np.random.default_rng(random_state)

    # 1) Window slicing
    X_win, y_win = create_windows(X, y, window_size, step_size)
    total = len(y_win)

    # 2) Original window-level counts
    classes, counts = np.unique(y_win, return_counts=True)
    props = counts / counts.sum()

    # 3) Compute new props
    if strategy == "power":
        new_props = _map_power(props, gamma)
    elif strategy == "softmax":
        new_props = _map_softmax(props, T)
    elif strategy == "linear":
        new_props = _map_linear(props, alpha)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 4) Compute target counts
    target = np.floor(new_props * total).astype(int)
    diff = total - target.sum()
    if diff != 0:
        target[np.argmax(target)] += diff

    window_count = dict(zip(classes, counts))
    target_count = dict(zip(classes, target))

    # 5) Resample per class
    X_parts, y_parts = [], []
    for cls in classes:
        idx = np.where(y_win == cls)[0]
        orig_n = len(idx)
        tgt_n  = target_count[cls]
        if orig_n >= tgt_n:
            sel = rng.choice(idx, size=tgt_n, replace=False)
            X_parts.append(X_win[sel])
            y_parts.append(np.full(tgt_n, cls))
        else:
            X_parts.append(X_win[idx])
            y_parts.append(np.full(orig_n, cls))
            needed = tgt_n - orig_n
            pick   = rng.choice(idx, size=needed, replace=True)
            noise  = rng.normal(0, sigma, (needed, X_win.shape[1]))
            X_parts.append(X_win[pick] + noise)
            y_parts.append(np.full(needed, cls))

    X_res = np.vstack(X_parts)
    y_res = np.hstack(y_parts)
    resampled_count = dict(Counter(y_res))

    # 6) Build summary DataFrame
    summary_df = pd.DataFrame({
        "windowed_count": pd.Series(window_count),
        "resampled_count": pd.Series(resampled_count)
    }).fillna(0).astype(int)

    return X_res, y_res, summary_df

if __name__ == "__main__":
    # --- Load and split data ---
    data_dir = '../data/surgical_trajectory'
    exp_files = [f for f in os.listdir(data_dir) if f.endswith('_Exp.csv')]
    nov_files = [f for f in os.listdir(data_dir) if f.endswith('_Nov.csv')]

    exp_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in exp_files], ignore_index=True)
    nov_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in nov_files], ignore_index=True)

    X_nov = nov_data.iloc[:, :-1].values
    y_nov = nov_data.iloc[:, -1].values
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_nov, y_nov, test_size=0.5, random_state=42)

    X_train = np.vstack([exp_data.iloc[:, :-1].values, Xn_tr])
    y_train = np.hstack([exp_data.iloc[:, -1].values, yn_tr])

    print("=== Frame-level counts (train) ===")
    pprint(Counter(y_train))

    # --- Compare strategies and plot ---
    ws, st = 10, 1
    strategies = ["power", "softmax", "linear"]
    for strat in strategies:
        print(f"\n==== Strategy: {strat} ====")
        _, _, df = window_and_balance(
            X_train, y_train,
            window_size=ws, step_size=st,
            strategy=strat,
            gamma=0.7, T=2.0, alpha=0.2,
            sigma=0.01, random_state=42
        )
        print(df)

        # Plot side by side with identical scales
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), tight_layout=True)
        ymax = max(df["windowed_count"].max(), df["resampled_count"].max()) * 1.05

        df["windowed_count"].plot.bar(ax=axes[0], title=f"{strat} strategy - Windowed Count")
        axes[0].set_ylim(0, ymax)
        axes[0].set_xlabel("Category")
        axes[0].set_ylabel("Count")

        df["resampled_count"].plot.bar(ax=axes[1], title=f"{strat} strategy - Resampled Count")
        axes[1].set_ylim(0, ymax)
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Count")

        fig.suptitle(f"Sample Distribution Comparison for {strat.capitalize()} Strategy", fontsize=16)
        plt.savefig(f"distribution_{strat}.png")
        plt.close(fig)
