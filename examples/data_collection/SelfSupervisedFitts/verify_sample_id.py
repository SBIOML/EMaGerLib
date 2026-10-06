import os
import re
import numpy as np
import matplotlib.pyplot as plt

# === FOLDER PATH === (defaults to the SESSION set in config.py)
from config import DATAFOLDER
FOLDER_PATH = DATAFOLDER


def load_sample_csvs(n_files):
    """
    Loads files that END with '#_sample_id.csv'
    where # is between 0 and n_files-1.

    Parameters
    ----------
    n_files : int
        Expected number of files (indices 0 to n_files-1)

    Returns
    -------
    List of numpy arrays ordered from 0 to n_files-1.
    """

    if n_files <= 0:
        raise ValueError("n_files must be > 0")

    # Prepare container
    data_list = [None] * n_files

    pattern = re.compile(r"(\d+)_sample_id\.csv$")

    for filename in os.listdir(FOLDER_PATH):
        match = pattern.search(filename)
        if match:
            index = int(match.group(1))

            if 0 <= index < n_files:
                filepath = os.path.join(FOLDER_PATH, filename)
                data = np.loadtxt(filepath, delimiter=",")
                data_list[index] = data

    # Check that all required files were found
    for i, item in enumerate(data_list):
        if item is None:
            raise FileNotFoundError(
                f"No file found ending with '{i}_sample_id.csv'"
            )

    return data_list



def plot_concatenated_connected(arr_list, colors=(None, None), lw=1.2):

    if not arr_list:
        raise ValueError("arr_list is empty.")

    segs = []
    lengths = []
    for k, a in enumerate(arr_list):
        a = np.asarray(a).ravel()
        if a.size == 0:
            raise ValueError(f"Segment {k} is empty.")
        segs.append(a)
        lengths.append(a.size)

    starts = np.cumsum([0] + lengths[:-1])
    ends = starts + np.array(lengths)
    total_len = np.sum(lengths)

    fig, (ax_sig, ax_d, ax_bar) = plt.subplots(
        3, 1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 2]}
    )

    # =====================================
    # 1) Concatenated signal
    # =====================================
    for i, a in enumerate(segs):
        c = colors[i % 2]
        x = np.arange(starts[i], ends[i])
        ax_sig.plot(x, a, color=c, lw=lw)

        if i < len(segs) - 1:
            next_a = segs[i + 1]
            next_c = colors[(i + 1) % 2]
            x_conn = np.array([ends[i] - 1, ends[i]])
            y_conn = np.array([a[-1], next_a[0]])
            ax_sig.plot(x_conn, y_conn, color=next_c, lw=lw)

    ax_sig.set_ylabel("Value")
    ax_sig.set_title("Concatenated signal (connected)")
    ax_sig.grid(True, alpha=0.3)

    # =====================================
    # 2) Delta
    # =====================================
    full = np.concatenate(segs)
    dfull = np.diff(full)
    x_d = np.arange(1, full.size)

    for i in range(len(segs)):
        c = colors[i % 2]

        j0 = max(starts[i], 1)
        j1 = ends[i]

        if j1 <= j0:
            continue

        idx0 = j0 - 1
        idx1 = j1 - 1
        ax_d.plot(x_d[idx0:idx1], dfull[idx0:idx1], color=c, lw=lw)

    ax_d.set_ylabel("Δ value")
    ax_d.set_title("Delta (first difference)")
    ax_d.grid(True, alpha=0.3)

    # =====================================
    # 3) Segment size bars
    # Width = length
    # Height = length
    # =====================================
    for i in range(len(segs)):
        c = colors[i % 2]

        ax_bar.bar(
            x=starts[i],
            height=lengths[i],
            width=lengths[i],
            align='edge',
            color=c
        )

    ax_bar.set_xlim(0, total_len)
    ax_bar.set_ylabel("Samples per file")
    ax_bar.set_title("Segment duration (width=time, height=sample count)")
    ax_bar.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax_sig, ax_d, ax_bar)



# Example:
# fig, ax1, ax2 = plot_concatenated_connected(arrays, colors=("C0","C1"))
# plt.show()






if __name__ == "__main__":

    N_FILES = 60   # <-- change here only

    arrays = load_sample_csvs(N_FILES)

    print(f"Loaded {len(arrays)} files.")
    for i in range(N_FILES):
        print(f"Shape of index {i}: {arrays[i].shape}")

    plot_concatenated_connected(arrays, colors=("C0", "C1"))
    plt.show()

