"""
eval_fewshot_loso.py -- Leave-one-SESSION-out few-shot evaluation.

Answers the question the rep-LOO benchmark cannot: *does on-device few-shot
calibration actually work, and is the gradient-free prototype rule competitive
with the obvious alternatives at the SAME calibration budget?*

Why a separate script from train_cnn_benchmark.py
--------------------------------------------------
That harness does leave-one-REP-out within a single session. Reps of one
session are nearly identically distributed, so calibration has nothing to
recover (before approx= after, every model approx 99%). The few-shot idea only
earns its keep under distribution shift.

This script instead trains on N-1 SESSIONS and treats the held-out session as a
new one: it draws `k` labeled windows per class (the "shots") for calibration
and evaluates on the rest. Cross-session shift (electrode shift, days apart) is
exactly what on-device calibration is supposed to fix.

Fairness
--------
Every method is scored on the SAME query set, and every *calibrated* method gets
the SAME `k` support shots. So a difference between methods isolates the
*mechanism*, not the data each one happened to see. The proto table in
models.md compares proto (which secretly gets 5 calibration shots) against
plain CNNs (which don't) -- that gap is mostly the free calibration data, not
the prototype rule. This script removes that confound.

Methods
-------
Floors (no calibration, k-independent):
    CNN (zero-shot)        Base CNN trained on the train sessions, used as-is.
    Proto-* (generic)      Prototypes = mean embedding over the train sessions.

Calibrated (swept over k shots/class):
    CNN + fine-tune        Base CNN, backprop-fine-tuned on the k shots.
    Proto-* (k-shot)       Prototypes = mean embedding of the k shots (NO backprop).

The headline is Proto-*(k-shot) vs CNN+fine-tune across k: can a gradient-free
prototype match/beat backprop fine-tuning, especially at small k?
The number to watch is the *calibration lift* (after - the method's own floor):
that is what a quick per-session calibration buys.

What a "shot" is
----------------
1 shot = 1 REPETITION. k-shot calibration = the user performs each gesture k
times. We hold out one whole rep of the session as the query (test) set, and
draw the k support reps from the OTHER reps; support = every window of those
reps, query = every window of the held-out rep.

This is the only meaningful unit for EMG. A "shot = one window" (0.1 s of
signal) is both unintuitive and leaky: with 95%-overlapping windows a support
window can be a near-duplicate of a query window. Splitting by rep makes k an
instruction you can actually give a user ("do each gesture k times") and is
leakage-free -- support and query come from different contractions.

k can range up to (reps_per_session - 1) since one rep is always reserved for
query. K_VALUES below is trimmed per-fold to whatever the held-out session
actually has (see `k_values_fold` in `main()`), so mixing session groups with
different rep counts is safe.

Caching
-------
Offline training (the expensive part) depends ONLY on the train sessions, not on
k or the shot definition. Trained models are cached under benchmark_results/
model_cache/, so re-running with a different k range / split is minutes, not the
full offline cost. Pass --fresh to ignore the cache and retrain.

Usage
-----
    python examples/training/eval_fewshot_loso.py
    python examples/training/eval_fewshot_loso.py --quick      # 1 seed, k in {1,2}
    python examples/training/eval_fewshot_loso.py --plot       # also save a PNG curve
    python examples/training/eval_fewshot_loso.py --fresh      # ignore cached models
"""

import sys
import copy
import time
import logging
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import lightning
from torch.utils.data import DataLoader, TensorDataset

# Reuse the canonical data pipeline (filters, dataset-info reader) and paths from
# the rep-LOO harness. That module has a __main__ guard, so importing it here is
# side-effect-free (no training kicked off).
sys.path.insert(0, str(Path(__file__).parent))
from train_cnn_benchmark import load_and_filter, read_dataset_info, BASE_PATH, fmt_duration  # noqa: E402

import emagerlib.models.new_emager_cnn as new_models  # noqa: E402

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------------

# Sessions of the SAME subject, leave-one-session-out. All must share classes/reps.
SESSIONS = [
    "Felix_5sessions/S_0",   # 2026-03-09 14:52
    "Felix_5sessions/S_1",   # 2026-03-10 22:54  (~1.3 days later)
    "Felix_5sessions/S_2",   # 2026-03-12 11:46  (~1.5 days later)
    "Felix_5sessions/S_3",   # 2026-03-14 19:49  (~2.3 days later)
    "Felix_5sessions/S_4",   # 2026-03-17 11:28  (~2.9 days later)
]
# Prior run used the EM subject's 3 sessions (see FEWSHOT_LOSO_LOG.md, 2026-06-10):
#   ["Test_EM_C7_R5", "Test_EM_C7_R5_02", "Test_EM_C7_R5_03"]

SEEDS    = [42, 123, 456]                  # offline-training seed AND which reps are support/query
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]     # calibration REPETITIONS per class (1 rep reserved for query;
                                            # Felix sessions have 10 reps each vs EM's 5, hence the wider sweep)

CNN_MODEL    = "EmagerCNNBase"
PROTO_MODELS = ["EmagerCNNProtoEpisodic", "EmagerCNNProtoCE"]

MAX_EPOCHS = 10                # offline embedding / CNN training
FT_EPOCHS  = 30                # CNN fine-tune steps on the k shots (full-batch)
FT_LR      = 1e-3

WINDOW_SIZE      = 200
WINDOW_INCREMENT = 10
SAMPLING         = 2000
BATCH_TRAIN      = 64
BATCH_TEST       = 256

INPUT_SHAPE = (4, 16)

# Rough wall-time per offline training, for the upfront estimate only.
EST_SECONDS_PER_TRAIN = 45

RESULTS_DIR = Path(__file__).parent / "benchmark_results"
LOG_PATH    = RESULTS_DIR / "FEWSHOT_LOSO_LOG.md"
CACHE_DIR   = RESULTS_DIR / "model_cache"

# ----------------------------------------------------------------------------
#  Pretty names for the output tables
# ----------------------------------------------------------------------------

def _short(model_name: str) -> str:
    """EmagerCNNProtoEpisodic -> protoEp ; EmagerCNNProtoCE -> protoCE."""
    return "proto" + model_name.replace("EmagerCNNProto", "")[:2]

LABELS = {
    "cnn_zeroshot": "CNN (zero-shot)",
    "cnn_finetune": "CNN + fine-tune",
}
for _m in PROTO_MODELS:
    _variant = _m.replace("EmagerCNNProto", "")          # "Episodic" / "CE"
    LABELS[f"{_short(_m)}_before"] = f"Proto-{_variant} (generic)"
    LABELS[f"{_short(_m)}_after"]  = f"Proto-{_variant} (k-shot)"


# ----------------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------------

def load_session_mav(name: str):
    """
    Load one session, window it across ALL reps.
    Returns (mav (N,64), labels (N,), reps (N,), num_classes). The per-window rep
    id is what lets us split calibration vs test by repetition rather than window.
    """
    path = BASE_PATH / name
    num_classes, num_reps = read_dataset_info(path)
    odh = load_and_filter(path, num_classes, num_reps, SAMPLING)
    windows, meta = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    mav    = np.mean(np.abs(windows), axis=2).astype(np.float32)   # collapse time axis -> (N, 64)
    labels = np.asarray(meta["classes"]).astype(np.int64)
    reps   = np.asarray(meta["reps"]).astype(np.int64)
    return mav, labels, reps, num_classes


def make_loader(mav, labels, batch, shuffle, drop_last):
    ds = TensorDataset(torch.from_numpy(mav), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last)


def build_rep_split(reps, rng):
    """
    Split the held-out session's reps into a fixed query rep + an ordered support
    pool of the remaining reps. The query rep is FIXED across all k, so accuracies
    at different k are directly comparable; k-shot uses the first k support reps.
    Returns (query_rep, support_pool: list of rep ids).
    """
    unique = np.unique(reps)
    perm   = rng.permutation(unique)
    return int(perm[0]), [int(r) for r in perm[1:]]


def support_indices(reps, support_pool, k):
    """Window indices whose rep is among the first k support reps."""
    return np.where(np.isin(reps, support_pool[:k]))[0]


# ----------------------------------------------------------------------------
#  Training / evaluation primitives
# ----------------------------------------------------------------------------

def train_offline(model_name, train_mav, train_labels, num_classes, seed):
    """Reseed, build a fresh shuffled loader, train one model on the train sessions."""
    lightning.seed_everything(seed, verbose=False)
    dl = make_loader(train_mav, train_labels, BATCH_TRAIN, shuffle=True, drop_last=True)
    model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
    model.fit(dl, max_epochs=MAX_EPOCHS)     # no test loader -> trains and returns
    model.eval()
    return model


def _cache_path(model_name, held_out, seed, train_names):
    """Cache key = everything that determines the offline weights. Shot/k are NOT
    in it: offline training is independent of the calibration protocol."""
    import hashlib
    tr = hashlib.md5("+".join(sorted(train_names)).encode()).hexdigest()[:8]
    # Session names may contain "/" (e.g. "Felix_5sessions/S_0") when a dataset
    # groups sessions in subfolders -- sanitize so the cache key stays a flat filename.
    ho = held_out.replace("/", "_").replace("\\", "_")
    fn = (f"{model_name}__ho-{ho}__seed{seed}__ep{MAX_EPOCHS}"
          f"__w{WINDOW_SIZE}-{WINDOW_INCREMENT}__sr{SAMPLING}__tr{tr}.pt")
    return CACHE_DIR / fn


def get_offline_model(model_name, train_mav, train_labels, num_classes, seed,
                      held_out, train_names, use_cache):
    """Train the offline model, or load it from cache if a matching one exists.
    Returns (model, was_cached)."""
    path = _cache_path(model_name, held_out, seed, train_names)
    if use_cache and path.exists():
        model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
        if hasattr(model, "_ce_head"):
            model._ce_head = None   # match the post-fit state (head dropped) so keys align
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        model.eval()
        return model, True
    model = train_offline(model_name, train_mav, train_labels, num_classes, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model, False


def finetune_cnn(base_model, sx, sy, epochs, lr):
    """Deep-copy the offline CNN and backprop-fine-tune it on the k shots (full-batch)."""
    m = copy.deepcopy(base_model)
    m.train()
    x = torch.as_tensor(sx, dtype=torch.float32)
    y = torch.as_tensor(sy, dtype=torch.long)
    opt  = torch.optim.AdamW(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        crit(m(x), y).backward()
        opt.step()
    m.eval()
    return m


def _accuracy(preds, labels):
    return float(np.mean(np.asarray(preds) == np.asarray(labels)))


def predict_acc(model, qx, qy):
    """Works for CNN and proto alike -- both expose the LibEMG predict()."""
    return _accuracy(model.predict(qx), qy)


def set_protos_and_acc(proto_model, proto_source, qx, qy):
    """Set prototypes from an iterable of (x, y) batches, then score the query set."""
    proto_model.prototypes = proto_model._prototypes_from(proto_source)
    return predict_acc(proto_model, qx, qy)


# ----------------------------------------------------------------------------
#  Reporting
# ----------------------------------------------------------------------------

def _agg(vals):
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _cell(vals):
    m, s = _agg(vals)
    if np.isnan(m):
        return "n/a"
    return f"{m:.1%} ± {s:.1%}"


def build_markdown(floors, swept, k_values, n_folds, elapsed):
    L = []
    L.append(f"## {datetime.now():%Y-%m-%d %H:%M:%S}  — Few-shot LOSO\n")
    L.append(f"- **Sessions (leave-one-out):** {', '.join(SESSIONS)}")
    L.append(f"- **Seeds:** {SEEDS}  |  **k (calibration reps/class):** {k_values}  "
             f"(1 shot = 1 repetition; 1 rep held out as query)")
    ft = f"{FT_EPOCHS} epochs @ lr {FT_LR} (full model)"
    L.append(f"- **Fine-tune:** {ft}")
    L.append(f"- **Offline:** {MAX_EPOCHS} epochs  |  Window {WINDOW_SIZE}/{WINDOW_INCREMENT}  "
             f"|  Sampling {SAMPLING}  |  Batch {BATCH_TRAIN}/{BATCH_TEST}")
    L.append(f"- **Folds:** {len(SESSIONS)} sessions × {len(SEEDS)} seeds = {n_folds} evals/method")
    host = f"cpu · torch {torch.__version__} · py {sys.version.split()[0]} · {platform.platform()}"
    L.append(f"- **Host:** {host}")
    L.append(f"- **Elapsed:** {fmt_duration(elapsed)}\n")

    # -- Floors --
    L.append("**Floors (no calibration, k-independent)**\n")
    L.append("| Method | Acc |")
    L.append("|---|---|")
    for key in ["cnn_zeroshot"] + [f"{_short(m)}_before" for m in PROTO_MODELS]:
        L.append(f"| {LABELS[key]} | {_cell(floors[key])} |")
    L.append("")

    # -- Calibrated vs k --
    L.append("**Calibrated (query accuracy vs k calibration reps/class)**\n")
    header = "| Method | " + " | ".join(f"k={k} rep" for k in k_values) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(k_values) + 1))
    cal_keys = ["cnn_finetune"] + [f"{_short(m)}_after" for m in PROTO_MODELS]
    for key in cal_keys:
        cells = " | ".join(_cell(swept[key][k]) for k in k_values)
        L.append(f"| {LABELS[key]} | {cells} |")
    L.append("")

    # -- Calibration lift (mean after - mean floor) --
    L.append("**Calibration lift (Δ accuracy vs each method's own floor, mean)**\n")
    L.append(header.replace("Method", "Method (vs floor)"))
    L.append("|" + "---|" * (len(k_values) + 1))
    lift_pairs = [("cnn_finetune", "cnn_zeroshot")]
    for m in PROTO_MODELS:
        lift_pairs.append((f"{_short(m)}_after", f"{_short(m)}_before"))
    for after_key, floor_key in lift_pairs:
        floor_mean = _agg(floors[floor_key])[0]
        cells = " | ".join(f"{_agg(swept[after_key][k])[0] - floor_mean:+.1%}" for k in k_values)
        L.append(f"| {LABELS[after_key]} | {cells} |")
    L.append("")
    return "\n".join(L)


def save_plot(floors, swept, k_values, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping plot (it was removed from deps).")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ["cnn_finetune"] + [f"{_short(m)}_after" for m in PROTO_MODELS]:
        means = [_agg(swept[key][k])[0] for k in k_values]
        ax.plot(k_values, means, marker="o", label=LABELS[key])
    for key in ["cnn_zeroshot"] + [f"{_short(m)}_before" for m in PROTO_MODELS]:
        ax.axhline(_agg(floors[key])[0], ls="--", alpha=0.6, label=LABELS[key])
    ax.set_xlabel("calibration repetitions per class (k)")
    ax.set_ylabel("query accuracy")
    ax.set_title("Few-shot calibration vs k reps (leave-one-session-out)")
    ax.set_xticks(k_values)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main(make_plot=False, use_cache=True):
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # so ± / · render on Windows consoles
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    k_values = sorted(set(K_VALUES))
    n_folds  = len(SESSIONS) * len(SEEDS)
    n_trains = n_folds * (1 + len(PROTO_MODELS))
    logger.info(f"Planned: {n_folds} folds (LOSO) × {1 + len(PROTO_MODELS)} models "
                f"= {n_trains} offline trainings; k sweep {k_values} reps/class")
    logger.info(f"Caching: {'on' if use_cache else 'OFF (--fresh)'} -> {CACHE_DIR}")
    logger.info(f"Estimated wall time (uncached): ~{fmt_duration(n_trains * EST_SECONDS_PER_TRAIN)}")

    start = time.perf_counter()

    # -- Load every session once (filter + window + MAV) ----------------------
    logger.info("Loading sessions ...")
    sessions = {}
    num_classes = None
    for name in SESSIONS:
        mav, labels, reps, C = load_session_mav(name)
        sessions[name] = (mav, labels, reps)
        num_classes = C
        logger.info(f"  {name}: {mav.shape[0]:,} windows, {C} classes, {len(np.unique(reps))} reps")

    # -- Result accumulators --------------------------------------------------
    floors = {"cnn_zeroshot": []}
    for m in PROTO_MODELS:
        floors[f"{_short(m)}_before"] = []
    swept = {"cnn_finetune": {k: [] for k in k_values}}
    for m in PROTO_MODELS:
        swept[f"{_short(m)}_after"] = {k: [] for k in k_values}

    # -- Leave-one-session-out ------------------------------------------------
    fold_i = 0
    for held_out in SESSIONS:
        train_names = [s for s in SESSIONS if s != held_out]
        train_mav   = np.concatenate([sessions[s][0] for s in train_names])
        train_labels = np.concatenate([sessions[s][1] for s in train_names])
        test_mav, test_labels, test_reps = sessions[held_out]

        for seed in SEEDS:
            fold_i += 1
            tag = f"[fold {fold_i}/{n_folds}] held-out={held_out} seed={seed}"
            logger.info("=" * 60)
            logger.info(f"{tag}  (train on {train_names})")

            # Pick a query rep + support-rep order for this fold (shared by EVERY
            # method). The query rep is fixed across k; k-shot uses k support reps.
            rng = np.random.default_rng(seed)
            query_rep, support_pool = build_rep_split(test_reps, rng)
            query_idx = np.where(test_reps == query_rep)[0]
            qx, qy = test_mav[query_idx], test_labels[query_idx]
            logger.info(f"  query = rep {query_rep} ({len(query_idx):,} windows)  "
                        f"support reps (ordered): {support_pool}")

            # -- Offline models (cached; reused across all k) -----------------
            cnn, c0 = get_offline_model(CNN_MODEL, train_mav, train_labels,
                                        num_classes, seed, held_out, train_names, use_cache)
            floors["cnn_zeroshot"].append(predict_acc(cnn, qx, qy))

            gen_dl = make_loader(train_mav, train_labels, BATCH_TEST, shuffle=False, drop_last=False)
            proto_trained, cached_flags = {}, [c0]
            for name in PROTO_MODELS:
                pm, ci = get_offline_model(name, train_mav, train_labels,
                                           num_classes, seed, held_out, train_names, use_cache)
                # generic prototypes (the "factory" floor) from the train sessions
                before = set_protos_and_acc(pm, gen_dl, qx, qy)
                floors[f"{_short(name)}_before"].append(before)
                proto_trained[name] = pm
                cached_flags.append(ci)

            src = "cached" if all(cached_flags) else ("trained" if not any(cached_flags) else "mixed")
            logger.info(f"  offline models: {src}  |  floors -> "
                        f"CNN {floors['cnn_zeroshot'][-1]:.1%}  "
                        + "  ".join(f"{_short(n)} {floors[_short(n)+'_before'][-1]:.1%}"
                                    for n in PROTO_MODELS))

            # -- k sweep (calibration), k = number of support reps -----------
            k_values_fold = [k for k in k_values if k <= len(support_pool)]
            for k in k_values_fold:
                s_idx  = support_indices(test_reps, support_pool, k)
                sx, sy = test_mav[s_idx], test_labels[s_idx]

                # CNN backprop fine-tune on the k support reps
                ft = finetune_cnn(cnn, sx, sy, FT_EPOCHS, FT_LR)
                swept["cnn_finetune"][k].append(predict_acc(ft, qx, qy))

                # Prototypes = mean embedding of the k support reps (no backprop)
                support_batch = [(torch.from_numpy(sx), torch.from_numpy(sy))]
                for name, pm in proto_trained.items():
                    acc = set_protos_and_acc(pm, support_batch, qx, qy)
                    swept[f"{_short(name)}_after"][k].append(acc)

                line = (f"  k={k}rep ({len(s_idx):,} win)  CNN-ft {swept['cnn_finetune'][k][-1]:.1%}  "
                        + "  ".join(f"{_short(n)} {swept[_short(n)+'_after'][k][-1]:.1%}"
                                    for n in PROTO_MODELS))
                logger.info(line)

    elapsed = time.perf_counter() - start

    # -- Report ---------------------------------------------------------------
    md = build_markdown(floors, swept, k_values, n_folds, elapsed)
    print("\n" + md)

    RESULTS_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Few-shot leave-one-session-out log\n\n"
                            "Most recent runs are appended below.\n\n", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(md + "\n")
    logger.info(f"Appended summary to: {LOG_PATH}")

    if make_plot:
        png = save_plot(floors, swept, k_values, RESULTS_DIR / "fewshot_loso_curve.png")
        if png:
            logger.info(f"Saved curve to: {png}")

    logger.info(f"Total elapsed: {fmt_duration(elapsed)}")
    return floors, swept


if __name__ == "__main__":
    if "--quick" in sys.argv:
        SEEDS    = [42]
        K_VALUES = [1, 2]
    main(make_plot="--plot" in sys.argv, use_cache="--fresh" not in sys.argv)
