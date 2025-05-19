import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

def create_windows(X, y=None, window_size=10, step=1):
    """
    Slide a window over X (n_samples, n_features), optionally y (n_samples,).
    Returns:
      X_win: np.ndarray of shape (n_windows, window_size * n_features)
      y_win: np.ndarray of shape (n_windows,) or None
    """
    n_samples, _ = X.shape
    Xw, yw = [], []
    for i in range(0, n_samples - window_size + 1, step):
        win = X[i : i + window_size].reshape(-1)
        Xw.append(win)
        if y is not None:
            yw.append(y[i + window_size - 1])
    Xw = np.stack(Xw, axis=0)
    yw = np.array(yw) if y is not None else None
    return Xw, yw

def augment_jitter(X, y, sigma=0.02, random_state=42):
    """
    Add Gaussian noise to frames of the minority class (frame-level jitter).
    Returns augmented X and y.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    mask = (y == minority)
    X_min = X[mask]
    noise = rng.normal(0, sigma, X_min.shape)
    X_aug = np.vstack([X, X_min + noise])
    y_aug = np.hstack([y, y[mask]])
    return X_aug, y_aug

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



def window_and_balance(
    X, y,
    window_size, step_size,
    strategy="softmax",    # mapping strategy
    gamma=0.7, T=2.0, alpha=0.2,
    sigma=0.02,
    random_state=42
):
    """
    1) Frame-level jitter augmentation on minority
    2) Window slicing
    3) Distribution mapping (power, softmax, or linear)
    4) Per-class over/under-sampling to match mapped proportions
       - When filling minority class, you can switch internally between 'replication + jitter' or 'SMOTE oversampling'
    """
    rng = np.random.default_rng(random_state)

    # === Internal switch: use SMOTE or replication + jitter for minority class oversampling ===
    USE_SMOTE_OVERSAMPLING = False

    # Control whether to apply jitter to SMOTE oversampled samples
    USE_JITTER_OVERSAMPLING = False

    BALANCE = True

    # --- Sliding windows only, no balancing ---
    if not BALANCE:
        print("Sliding windows only")
        X_win, y_win = create_windows(X, y, window_size, step_size)
        return X_win, y_win

    # 1) augment minority frames
    X_aug, y_aug = augment_jitter(X, y, sigma=sigma, random_state=random_state)

    # print("Test")

    # 2) window slicing
    X_win, y_win = create_windows(X_aug, y_aug, window_size, step_size)
    N = len(y_win)

    # 3) compute original proportions & map
    classes, counts = np.unique(y_win, return_counts=True)
    props = counts / counts.sum()
    if strategy == "power":
        # print("power")
        new_props = _map_power(props, gamma)
    elif strategy == "softmax":
        # print("softmax")
        new_props = _map_softmax(props, T)
    elif strategy == "linear":
        # print("linear")
        new_props = _map_linear(props, alpha)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 4) determine target counts
    target = np.floor(new_props * N).astype(int)
    diff = N - target.sum()
    if diff > 0:
        target[np.argmax(target)] += diff

    # 5) per-class resampling
    X_parts, y_parts = [], []
    for cls, orig_n, tgt_n in zip(classes, counts, target):
        idx = np.where(y_win == cls)[0]

        # 5a) More than target, random undersampling
        if orig_n >= tgt_n:
            sel = rng.choice(idx, size=tgt_n, replace=False)
            X_parts.append(X_win[sel])
            y_parts.append(np.full(tgt_n, cls))

        # 5b) Less than target, choose internal strategy
        else:
            # -- Option A: replication + jitter --
            if not USE_SMOTE_OVERSAMPLING:
                # print("Not SMOTE")
                # First keep all
                X_parts.append(X_win[idx])
                y_parts.append(np.full(orig_n, cls))
                # Then replicate windows from idx and add noise to make up
                needed = tgt_n - orig_n
                pick   = rng.choice(idx, size=needed, replace=True)
                noise  = rng.normal(0, sigma, (needed, X_win.shape[1]))
                X_parts.append(X_win[pick] + noise)
                y_parts.append(np.full(needed, cls))

            # -- Option B: SMOTE oversampling --
            else:
                # For classes below target count, use SMOTE to augment
                from imblearn.over_sampling import SMOTE
                print(f"SMOTE oversampling for class {cls}")

                # Prepare current minority class window data
                X_minority = X_win[idx]
                y_minority = np.full(orig_n, cls)

                # SMOTE sampling target count = tgt_n - orig_n
                sm = SMOTE(
                    sampling_strategy={cls: tgt_n},
                    random_state=random_state
                )

                # SMOTE requires all windows and labels as input: here we use full X_win and y_win
                X_res, y_res = sm.fit_resample(X_win, y_win)

                # Extract all windows of this class after SMOTE
                mask_res = (y_res == cls)
                X_cls_all = X_res[mask_res]

                # Keep only the first tgt_n windows
                X_cls_sel = X_cls_all[:tgt_n]

                # If jitter switch is on, add small Gaussian jitter to these synthetic samples
                if USE_JITTER_OVERSAMPLING:
                    print(f"[JITTER] Applying Gaussian jitter to SMOTE samples for class {cls}, shape={X_cls_sel.shape}")
                    # rng is defined at the start of the function: rng = np.random.default_rng(random_state)
                    noise = rng.normal(0, sigma, size=X_cls_sel.shape)
                    X_cls_sel = X_cls_sel + noise

                # Add selected windows to the final list
                X_parts.append(X_cls_sel)
                y_parts.append(np.full(tgt_n, cls))


    # 6) Concatenate all class parts
    X_final = np.vstack(X_parts)
    y_final = np.hstack(y_parts)

    return X_final, y_final
