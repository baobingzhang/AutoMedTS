import numpy as np
from sklearn.utils import resample
from collections import defaultdict


def apply_sliding_window(X, y=None, window_size: int = 10, step_size: int = 1):
    """
    Applies a sliding window transformation to tabular data, with optional majority voting for labels.
    
    Parameters:
        X (array-like): Input features with shape (n_samples, n_features).
        y (array-like or None): Input labels with shape (n_samples,). If None, only X is transformed.
        window_size (int): The number of consecutive rows to group as one sample.
        step_size (int): The sliding step size.
    
    Returns:
        new_X (np.ndarray): Transformed features with shape (n_windows, window_size * n_features).
        new_y (np.ndarray or None): Transformed labels, or None if y is None.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    n_windows = (n_samples - window_size) // step_size + 1

    new_X = []
    new_y = [] if y is not None else None

    # Convert y to numpy array only if provided
    if y is not None:
        y = np.asarray(y)

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = X[start:end]
        new_X.append(window.flatten())

        if y is not None:
            window_labels = y[start:end]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            mode_label = unique_labels[np.argmax(counts)]
            new_y.append(mode_label)

    return (
        np.array(new_X),
        np.array(new_y) if y is not None else None
    )




def apply_balanced_sliding_window(X, y=None, window_size: int = 10, step_size: int = 1, max_per_class: int = 1000):
    """
    Applies a class-balanced sliding window transformation.
    Ensures each class contributes a balanced number of windows to reduce class imbalance.

    Parameters:
        X (array-like): Input features with shape (n_samples, n_features).
        y (array-like or None): Input labels with shape (n_samples,). If None, only X is transformed.
        window_size (int): Number of time steps in each window.
        step_size (int): Step size between windows.
        max_per_class (int): Maximum number of sliding windows to sample per class.

    Returns:
        new_X (np.ndarray): Transformed feature vectors with shape (n_windows, window_size * n_features).
        new_y (np.ndarray or None): Majority-voted labels for each window, or None if y is None.
    """
    X = np.asarray(X)
    if y is None:
        # 无标签模式：整体滑窗（与原函数一致）
        n_samples, n_features = X.shape
        new_X = []
        for start in range(0, n_samples - window_size + 1, step_size):
            window = X[start:start+window_size]
            new_X.append(window.flatten())
        return np.array(new_X), None

    y = np.asarray(y)
    unique_classes = np.unique(y)
    new_X, new_y = [], []

    for cls in unique_classes:
        # 当前类别对应的所有索引
        cls_indices = np.where(y == cls)[0]

        # 要求窗口必须连续，所以截取连续片段
        cls_X = X[cls_indices]
        cls_y = y[cls_indices]

        n_samples = cls_X.shape[0]
        if n_samples < window_size:
            continue  # 跳过过短类别

        # 类内滑窗
        class_X_windows = []
        class_y_windows = []
        for start in range(0, n_samples - window_size + 1, step_size):
            window_X = cls_X[start:start + window_size]
            window_y = cls_y[start:start + window_size]

            class_X_windows.append(window_X.flatten())
            # 多数投票
            unique_labels, counts = np.unique(window_y, return_counts=True)
            class_y_windows.append(unique_labels[np.argmax(counts)])

        # 类别窗口采样（平衡）
        if len(class_X_windows) > max_per_class:
            X_sampled, y_sampled = resample(
                class_X_windows, class_y_windows,
                n_samples=max_per_class, replace=False, random_state=42
            )
        else:
            X_sampled, y_sampled = resample(
                class_X_windows, class_y_windows,
                n_samples=max_per_class, replace=True, random_state=42
            )

        new_X.extend(X_sampled)
        new_y.extend(y_sampled)

    return np.array(new_X), np.array(new_y)



