import os
import re
import numpy as np
import matplotlib.pyplot as plt

# === FOLDER PATH === (defaults to the SESSION set in config.py)
from config import DATAFOLDER
FOLDER_PATH = DATAFOLDER


def load_indexed_emg_csvs(n_files, folder_path=FOLDER_PATH, delimiter=",", dtype=float):
    """
    Loads files that END with '#_emg.csv' where # is in [0, n_files-1].
    Each file is expected to contain a 2D array (rows=samples, cols=channels/features).

    Returns
    -------
    arrays : list[np.ndarray]
        List of 2D arrays ordered by index.
    """
    if n_files <= 0:
        raise ValueError("n_files must be > 0")

    arrays = [None] * n_files
    pattern = re.compile(r"(\d+)_emg\.csv$")

    for filename in os.listdir(folder_path):
        m = pattern.search(filename)
        if not m:
            continue

        idx = int(m.group(1))
        if 0 <= idx < n_files:
            filepath = os.path.join(folder_path, filename)
            data = np.loadtxt(filepath, delimiter=delimiter, dtype=dtype)

            # Ensure 2D (in case a file has 1 row or 1 col and collapses)
            data = np.atleast_2d(data)

            arrays[idx] = data

    # Validate all found
    missing = [i for i, a in enumerate(arrays) if a is None]
    if missing:
        raise FileNotFoundError(
            "Missing file(s) ending with: " + ", ".join([f"{i}_emg.csv" for i in missing])
        )

    # Validate consistent number of columns across files
    n_cols = arrays[0].shape[1]
    for i, a in enumerate(arrays):
        if a.shape[1] != n_cols:
            raise ValueError(
                f"Column mismatch: file index {i} has {a.shape[1]} cols, expected {n_cols}."
            )

    return arrays


def stack_emg_files(arrays):
    """
    Vertically concatenates a list of 2D arrays (same #columns) into one big 2D array.

    Returns
    -------
    big : np.ndarray
        Shape = (sum_i rows_i, n_cols)
    starts : np.ndarray
        Start sample index (in big array) for each segment
    lengths : np.ndarray
        Number of rows per segment
    """
    lengths = np.array([a.shape[0] for a in arrays], dtype=int)
    starts = np.cumsum(np.r_[0, lengths[:-1]])
    big = np.vstack(arrays)
    return big, starts, lengths


def plot_emg_column_time_series(big, starts, lengths, col=0, colors=("C0", "C1"), lw=1.0, connect=True):
    """
    Plots one column of the big concatenated EMG array as a continuous time series,
    optionally alternating colors per file segment.

    Parameters
    ----------
    big : np.ndarray
        Concatenated 2D array (N_total, n_cols)
    starts, lengths : array-like
        Segment layout as returned by stack_emg_files
    col : int
        Column index to plot
    colors : tuple
        Two matplotlib colors to alternate between segments
    connect : bool
        If True, draws a connector line between segments to avoid visual breaks.
    """
    if big.ndim != 2:
        raise ValueError("big must be a 2D array")
    if not (0 <= col < big.shape[1]):
        raise ValueError(f"col must be in [0, {big.shape[1]-1}]")

    starts = np.asarray(starts, dtype=int)
    lengths = np.asarray(lengths, dtype=int)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    y = big[:, col]

    for i in range(len(lengths)):
        s = starts[i]
        e = s + lengths[i]
        c = colors[i % 2]

        x = np.arange(s, e)
        ax.plot(x, y[s:e], color=c, lw=lw)

        if connect and i < len(lengths) - 1:
            s2 = starts[i + 1]
            # connector from last sample of seg i to first sample of seg i+1
            x_conn = np.array([e - 1, e])
            y_conn = np.array([y[e - 1], y[s2]])
            ax.plot(x_conn, y_conn, color=colors[(i + 1) % 2], lw=lw)

    ax.set_title(f"EMG column {col} across {len(lengths)} files (concatenated)")
    ax.set_xlabel("Sample index (concatenated)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax

def plot_emg_column_alternating(big, starts, lengths, col=0, colors=("C0", "C1"), lw=1.0):

    if not (0 <= col < big.shape[1]):
        raise ValueError(f"Column must be between 0 and {big.shape[1]-1}")

    fig, ax = plt.subplots(figsize=(12, 4))
    y = big[:, col]

    for i in range(len(lengths)):
        s = starts[i]
        e = s + lengths[i]
        c = colors[i % 2]

        x = np.arange(s, e)
        ax.plot(x, y[s:e], color=c, lw=lw)

        # Connector to avoid visual break
        if i < len(lengths) - 1:
            x_conn = np.array([e - 1, e])
            y_conn = np.array([y[e - 1], y[e]])
            ax.plot(x_conn, y_conn, color=colors[(i + 1) % 2], lw=lw)

    ax.set_title(f"EMG column {col} across {len(lengths)} files (alternating colors)")
    ax.set_xlabel("Sample index (concatenated)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    N_FILES = 60  # <-- set this

    emg_arrays = load_indexed_emg_csvs(N_FILES)
    big_emg, starts, lengths = stack_emg_files(emg_arrays)

    print("Loaded EMG arrays:")
    for i, a in enumerate(emg_arrays):
        print(f"  {i}: {a.shape}")

    print("Big EMG shape:", big_emg.shape)

    plot_emg_column_alternating(
        big_emg,
        starts,
        lengths,
        col=63,
        colors=("C0", "C1"),
        lw=0.8
    )

    plt.show()
