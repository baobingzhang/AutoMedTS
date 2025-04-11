import numpy as np

# def apply_sliding_window(X, y, window_size: int = 10, step_size: int = 1):
#     """
#     Applies a sliding window transformation to tabular data with majority voting for labels.
    
#     Parameters:
#         X (array-like): Input features with shape (n_samples, n_features).
#         y (array-like): Input labels with shape (n_samples,).
#         window_size (int): The number of consecutive rows to group as one sample.
#         step_size (int): The sliding step size.
    
#     Returns:
#         new_X (np.ndarray): Transformed features with shape (n_windows, window_size * n_features).
#         new_y (np.ndarray): Transformed labels, where each label is determined via majority voting 
#                             among the labels in the window.
    
#     Example:
#         If X has 100 samples and window_size=10, step_size=1, then new_X will have 91 samples,
#         and each sample is a vector of size (10 * n_features). The label for each window is the label 
#         that occurs most frequently in the window.
#     """
#     # Convert inputs to numpy arrays
#     X = np.asarray(X)
#     y = np.asarray(y)
    
#     n_samples, n_features = X.shape
#     n_windows = (n_samples - window_size) // step_size + 1

#     new_X = []
#     new_y = []

#     # Iterate over sliding windows
#     for start in range(0, n_samples - window_size + 1, step_size):
#         end = start + window_size
#         window = X[start:end]
#         # Flatten the window into a single 1D vector
#         new_X.append(window.flatten())

#         # Instead of taking the last label, use majority voting over all labels in the window.
#         window_labels = y[start:end]
#         # Count unique tags and their occurrences
#         unique_labels, counts = np.unique(window_labels, return_counts=True)
#         # Select the label with the most occurrences as the final label for the window (majority vote)
#         mode_label = unique_labels[np.argmax(counts)]
#         new_y.append(mode_label)
    
#     return np.array(new_X), np.array(new_y)


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

