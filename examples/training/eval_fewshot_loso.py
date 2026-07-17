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

Cost, and how to keep it down
-----------------------------
MEASURED on Felix (2026-07-16, CPU): ~9-12 min per offline training, i.e. ~35 min
for one fold of three models. The bill is dominated by OFFLINE TRAINING, not by the
calibration sweep:

    offline training: models x sessions x seeds, ~9-12 min each on Felix. <- bottleneck
    CNN + fine-tune : FT_EPOCHS full-batch passes per k per fold.  ~40 min total.
    proto k-shot    : one forward pass over the support windows + a mean. Negligible.

Felix is ~166k training windows per fold (4 sessions x ~41.6k) against ~24k for the
EM datasets -- ~7x the data, which is why EST_SECONDS_PER_TRAIN (tuned on EM) badly
under-predicts here. The 2026-07-15 Felix run's 6h47m was ~45 uncached trainings,
NOT the fine-tune sweep.

Three levers, in order of value:

1. STAGE1_SOURCE (automatic, exact). The PTQ variants share their FP32 stage-1
   weights bit-for-bit with EmagerCNNProtoRingStrided, so the harness trains it once
   per (fold, seed) and derives the rest in seconds. Without this the 5-variant sweep
   is ~15h; with it, ~4h. Verified bit-identical -- see STAGE1_SOURCE.
2. The cache. Offline training depends only on the train sessions, so re-running with
   a different k range or split costs minutes. Only NEW models pay training cost.
3. --skip-cnn. Worth ~40 min (the CNN's own training + its fine-tune sweep). Real but
   secondary -- it does NOT touch the bottleneck. Use it when the CNN rows can be read
   off an earlier run of the same sessions, since that baseline does not move when a
   proto variant is added.

    --models EmagerCNNProtoRingStridedPTQ --skip-cnn --k 1,3,5

Re-run the CNN baseline only when the sessions, the window/MAV pipeline, or
MAX_EPOCHS change.

Usage
-----
    python examples/training/eval_fewshot_loso.py                    # everything
    python examples/training/eval_fewshot_loso.py --quick            # plumbing check
    python examples/training/eval_fewshot_loso.py --models A,B --skip-cnn --k 1,3,5
    python examples/training/eval_fewshot_loso.py --plot             # also save a PNG curve
    python examples/training/eval_fewshot_loso.py --fresh            # ignore cached models
    python examples/training/eval_fewshot_loso.py --help
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
# Default set. Override per-run with --models; the deployment-line variants below
# are the RingStrided (smallest arch) branch, PTQ/QAT being the INT8 leaves.
PROTO_MODELS = [
    "EmagerCNNProtoEpisodic",           # Base arch, episodic       — measured (EM + Felix)
    "EmagerCNNProtoCE",                 # Base arch, cross-entropy  — measured (EM + Felix)
    "EmagerCNNProtoRingStrided",        # RingStrided arch, FP32     — pending
    "EmagerCNNProtoRingStridedPTQ",     # RingStrided arch, INT8 PTQ — pending
    "EmagerCNNProtoRingStridedQAT",     # RingStrided arch, INT8 QAT — pending
]

# Variants whose FP32 stage-1 training is BIT-IDENTICAL to another model's, so it can
# be trained once and shared instead of repeated.
#
# Why it is exact, not an approximation: these subclasses add only QuantStub/DeQuantStub
# (no parameters -> no RNG draws during __init__) on top of an unchanged backbone and an
# unchanged episodic loss. After seed_everything(seed) both get the same init, the same
# shuffled batches and the same forward numerics -- the stubs are no-ops until prepare().
# So the shared model's weights ARE each variant's post-stage-1 weights; the harness
# loads them and calls quantize_and_eval() to run only the INT8 stage (seconds).
#
# QAT is deliberately absent: it fine-tunes with fake quant, so its weights genuinely
# differ and it must train on its own.
#
# This is the difference between ~4h and ~15h for the 5-variant sweep on Felix, where
# one offline training is ~9-12 min (166k windows/fold, CPU).
# QAT is included too: its FP32 *warm start* is the same training, and it then pays
# only its fake-quant fine-tuning on top (qat_epochs, ~1/3 the cost of a full train).
STAGE1_SOURCE = {
    "EmagerCNNProtoRingStridedPTQ":             "EmagerCNNProtoRingStrided",
    "EmagerCNNProtoRingStridedPTQFp32Protos":   "EmagerCNNProtoRingStrided",
    "EmagerCNNProtoRingStridedPTQSupportCalib": "EmagerCNNProtoRingStrided",
    "EmagerCNNProtoRingStridedQAT":             "EmagerCNNProtoRingStrided",
}

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

def _variant(model_name: str) -> str:
    """EmagerCNNProtoRingStridedPTQ -> RingStridedPTQ."""
    return model_name.replace("EmagerCNNProto", "")


def _short(model_name: str) -> str:
    """
    Result-dict key for a proto model: EmagerCNNProtoEpisodic -> protoEpisodic.

    Must keep the variant name whole. An earlier version truncated it to 2 chars
    ("protoEp" / "protoCE"), which silently collided once the RingStrided family
    arrived -- RingStrided, RingStridedPTQ and RingStridedQAT all mapped to
    "protoRi" and would have merged into each other's results.
    """
    return "proto" + _variant(model_name)


CNN_LABELS = {
    "cnn_zeroshot": "CNN (zero-shot)",
    "cnn_finetune": "CNN + fine-tune",
}


def label_for(key: str) -> str:
    """Pretty table label for a result key."""
    if key in CNN_LABELS:
        return CNN_LABELS[key]
    body, _, suffix = key.rpartition("_")
    kind = "generic" if suffix == "before" else "k-shot"
    return f"Proto-{body.replace('proto', '', 1)} ({kind})"


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


def _is_quantized_family(model_name) -> bool:
    """True for the INT8 variants (anything carrying EmagerCNNQuantizedPTQ's qbackend)."""
    return hasattr(getattr(new_models, model_name), "qbackend")


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


def needs_per_k_quant(model_name) -> bool:
    """
    True for variants whose quantization depends on the support set, so it cannot be
    done once per fold and reused across the k sweep.

    Only calib_source="support" does this: its observers see the user's k shots, and k
    changes inside the sweep. Cheap to honour anyway -- with STAGE1_SOURCE the FP32 base
    is already in memory, so re-quantizing is fuse+calibrate+convert (seconds), not a
    retrain.
    """
    return getattr(getattr(new_models, model_name), "calib_source", "train") == "support"


def uses_fp32_protos(model_name) -> bool:
    """True for the proto_precision="fp32" ablation: prototypes come from the
    pre-quantization embedding while classification still runs through INT8."""
    return getattr(getattr(new_models, model_name), "proto_precision", "int8") == "fp32"


def _is_qat(model_name) -> bool:
    """True for QAT variants, which fine-tune instead of calibrating observers."""
    return hasattr(getattr(new_models, model_name), "qat_epochs")


def derive_quantized(model_name, base, num_classes, calib_batches=None, train_dl=None):
    """
    Build a quantized variant from an already-trained FP32 stage-1 model, skipping
    stage 1 entirely (see STAGE1_SOURCE for why the weights are interchangeable).

    `base` is left untouched -- its weights are copied out, so the caller can keep using
    it (e.g. as the FP32 prototype source for the Fp32Protos ablation, or as the stage-1
    for the next variant).

    PTQ variants get `calib_batches` (explicit observer calibration); QAT ignores that
    and fine-tunes on `train_dl` instead. Both are reached through the same
    quantize_from_pretrained() entry point.
    """
    model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    # The FP32 base and its quantized subclasses have identical parameter trees before
    # convert(); anything else means STAGE1_SOURCE claims an equivalence that does not hold.
    if missing or unexpected:
        raise RuntimeError(
            f"STAGE1_SOURCE[{model_name!r}] weights do not match: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys. "
            f"The shared-stage-1 assumption is broken -- check the two classes' __init__."
        )
    model.eval()
    # test_dataloader=None keeps this to the quantization stage -- prototypes/scoring are
    # the harness's job (set_protos_and_acc), per k.
    if _is_qat(model_name):
        model.quantize_from_pretrained(train_dl, None)
    else:
        # Calibration data is passed explicitly: this harness owns the support/query
        # split, so the model must not try to re-derive it from a test_dataloader it
        # never sees.
        model.quantize_from_pretrained(None, None, calib_loader=calib_batches)
    return model


def get_offline_model(model_name, train_mav, train_labels, num_classes, seed,
                      held_out, train_names, use_cache):
    """
    Train an FP32 offline model, or load it from cache if a matching one exists.
    Returns (model, was_cached).

    FP32 only, by design. A converted model does not round-trip in either direction:
    its state_dict will not load into a fresh instance (convert() restructured the
    module tree), and pickling the object whole produces fused modules missing their
    `_modules` dict -- the reload succeeds but .eval() and forward both raise
    AttributeError. Quantized variants are therefore never cached; they are derived
    from a cached FP32 stage-1 via STAGE1_SOURCE + derive_quantized(), which is exact
    and takes seconds.
    """
    if _is_quantized_family(model_name):
        raise RuntimeError(
            f"{model_name} is a quantized variant and cannot be trained or cached "
            f"directly (converted models do not survive save/load). Add a STAGE1_SOURCE "
            f"entry so it is derived from a shared FP32 stage-1 instead."
        )

    path = _cache_path(model_name, held_out, seed, train_names)
    if use_cache and path.exists():
        model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
        if hasattr(model, "_ce_head"):
            model._ce_head = None   # match the post-fit state (head dropped) so keys align
        incompatible = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        # strict=False is needed for the dropped CE head, but it also means a
        # structurally mismatched cache would load NOTHING and silently hand back a
        # randomly initialised model. Fail loudly instead.
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Stale/incompatible cached model for {model_name} at {path.name}: "
                f"{len(incompatible.missing_keys)} missing / "
                f"{len(incompatible.unexpected_keys)} unexpected keys. "
                f"Delete it or re-run with --fresh."
            )
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


def set_protos_and_acc(model, proto_source, qx, qy, proto_src=None):
    """
    Set prototypes from an iterable of (x, y) batches, then score the query set.

    proto_src: the model whose embedding COMPUTES the prototypes; defaults to `model`.
        The proto_precision="fp32" ablation passes the pre-quantization model here, so
        prototypes come from the FP32 embedding while classification still runs through
        `model` (INT8). Prototypes are plain (C, D) FP32 tensors either way, so they
        transfer between the two.
    """
    model.prototypes = (proto_src or model)._prototypes_from(proto_source)
    return predict_acc(model, qx, qy)


# ----------------------------------------------------------------------------
#  Reporting
# ----------------------------------------------------------------------------

def _pct(v):
    """Percent for log lines; NaN (an undefined metric, not a zero) renders as n/a."""
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


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


def build_markdown(floors, swept, models, seeds, k_values, n_folds, elapsed, with_cnn):
    L = []
    L.append(f"## {datetime.now():%Y-%m-%d %H:%M:%S}  — Few-shot LOSO\n")
    L.append(f"- **Sessions (leave-one-out):** {', '.join(SESSIONS)}")
    L.append(f"- **Models:** {', '.join(models)}")
    L.append(f"- **Seeds:** {seeds}  |  **k (calibration reps/class):** {k_values}  "
             f"(1 shot = 1 repetition; 1 rep held out as query)")
    if with_cnn:
        L.append(f"- **Fine-tune:** {FT_EPOCHS} epochs @ lr {FT_LR} (full model)")
    else:
        L.append("- **Fine-tune:** _CNN baseline skipped (--skip-cnn) — see an earlier "
                 "run of the same sessions for the CNN rows_")
    L.append(f"- **Offline:** {MAX_EPOCHS} epochs  |  Window {WINDOW_SIZE}/{WINDOW_INCREMENT}  "
             f"|  Sampling {SAMPLING}  |  Batch {BATCH_TRAIN}/{BATCH_TEST}")
    L.append(f"- **Folds:** {len(SESSIONS)} sessions × {len(seeds)} seeds = {n_folds} evals/method")
    host = f"cpu · torch {torch.__version__} · py {sys.version.split()[0]} · {platform.platform()}"
    L.append(f"- **Host:** {host}")
    L.append(f"- **Elapsed:** {fmt_duration(elapsed)}\n")

    floor_keys = (["cnn_zeroshot"] if with_cnn else []) + [f"{_short(m)}_before" for m in models]
    cal_keys   = (["cnn_finetune"] if with_cnn else []) + [f"{_short(m)}_after"  for m in models]

    # -- Floors --
    L.append("**Floors (no calibration, k-independent)**\n")
    L.append("| Method | Acc |")
    L.append("|---|---|")
    for key in floor_keys:
        L.append(f"| {label_for(key)} | {_cell(floors[key])} |")
    L.append("")

    # -- Calibrated vs k --
    L.append("**Calibrated (query accuracy vs k calibration reps/class)**\n")
    header = "| Method | " + " | ".join(f"k={k} rep" for k in k_values) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(k_values) + 1))
    for key in cal_keys:
        cells = " | ".join(_cell(swept[key][k]) for k in k_values)
        L.append(f"| {label_for(key)} | {cells} |")
    L.append("")

    # -- Calibration lift (mean after - mean floor) --
    L.append("**Calibration lift (Δ accuracy vs each method's own floor, mean)**\n")
    L.append(header.replace("Method", "Method (vs floor)"))
    L.append("|" + "---|" * (len(k_values) + 1))
    lift_pairs = ([("cnn_finetune", "cnn_zeroshot")] if with_cnn else [])
    lift_pairs += [(f"{_short(m)}_after", f"{_short(m)}_before") for m in models]
    for after_key, floor_key in lift_pairs:
        floor_mean = _agg(floors[floor_key])[0]
        # A method with no defined floor (support-calibrated: no factory state exists)
        # has no defined lift either -- render n/a rather than "+nan%".
        cells = " | ".join(
            "n/a" if np.isnan(floor_mean) else f"{_agg(swept[after_key][k])[0] - floor_mean:+.1%}"
            for k in k_values
        )
        L.append(f"| {label_for(after_key)} | {cells} |")
    L.append("")
    return "\n".join(L)


def save_plot(floors, swept, models, k_values, path, with_cnn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping plot (it was removed from deps).")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for key in (["cnn_finetune"] if with_cnn else []) + [f"{_short(m)}_after" for m in models]:
        means = [_agg(swept[key][k])[0] for k in k_values]
        ax.plot(k_values, means, marker="o", label=label_for(key))
    for key in (["cnn_zeroshot"] if with_cnn else []) + [f"{_short(m)}_before" for m in models]:
        ax.axhline(_agg(floors[key])[0], ls="--", alpha=0.6, label=label_for(key))
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

def main(models=None, seeds=None, k_values=None, with_cnn=True,
         make_plot=False, use_cache=True):
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # so ± / · render on Windows consoles
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    models   = list(models) if models else list(PROTO_MODELS)
    seeds    = list(seeds)  if seeds  else list(SEEDS)
    k_values = sorted(set(k_values if k_values else K_VALUES))

    n_folds  = len(SESSIONS) * len(seeds)
    n_trains = n_folds * (len(models) + (1 if with_cnn else 0))
    logger.info(f"Planned: {n_folds} folds (LOSO) × {len(models) + (1 if with_cnn else 0)} models "
                f"= {n_trains} offline trainings; k sweep {k_values} reps/class")
    logger.info(f"Proto models: {', '.join(models)}")
    if with_cnn:
        logger.info(f"CNN baseline: on ({CNN_MODEL}; zero-shot floor + {FT_EPOCHS}-epoch fine-tune per k)")
    else:
        logger.info("CNN baseline: OFF (--skip-cnn) — proto models only")
    logger.info(f"Caching: {'on' if use_cache else 'OFF (--fresh)'} -> {CACHE_DIR}")
    logger.info(f"Estimated wall time (uncached): ~{fmt_duration(n_trains * EST_SECONDS_PER_TRAIN)}"
                + ("" if with_cnn else "  (+ no CNN fine-tune sweep, the usual bottleneck)"))

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
    floors = {"cnn_zeroshot": []} if with_cnn else {}
    for m in models:
        floors[f"{_short(m)}_before"] = []
    swept = {"cnn_finetune": {k: [] for k in k_values}} if with_cnn else {}
    for m in models:
        swept[f"{_short(m)}_after"] = {k: [] for k in k_values}

    # -- Leave-one-session-out ------------------------------------------------
    fold_i = 0
    for held_out in SESSIONS:
        train_names = [s for s in SESSIONS if s != held_out]
        train_mav   = np.concatenate([sessions[s][0] for s in train_names])
        train_labels = np.concatenate([sessions[s][1] for s in train_names])
        test_mav, test_labels, test_reps = sessions[held_out]

        for seed in seeds:
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
            cnn, cached_flags = None, []
            if with_cnn:
                cnn, c0 = get_offline_model(CNN_MODEL, train_mav, train_labels,
                                            num_classes, seed, held_out, train_names, use_cache)
                floors["cnn_zeroshot"].append(predict_acc(cnn, qx, qy))
                cached_flags.append(c0)

            gen_dl = make_loader(train_mav, train_labels, BATCH_TEST, shuffle=False, drop_last=False)
            # Observer-calibration batches for PTQ variants derived from a shared stage-1.
            calib_dl = make_loader(train_mav, train_labels, BATCH_TRAIN, shuffle=False, drop_last=False)
            # Shuffled loader for QAT's fine-tuning, matching train_offline's setup.
            qat_dl = make_loader(train_mav, train_labels, BATCH_TRAIN, shuffle=True, drop_last=True)
            proto_trained = {}
            for name in models:
                src_name = STAGE1_SOURCE.get(name)
                if src_name is None:
                    pm, ci = get_offline_model(name, train_mav, train_labels, num_classes,
                                               seed, held_out, train_names, use_cache)
                    base = pm
                else:
                    # Train (or load) the shared FP32 stage-1 ONCE, then derive.
                    base, ci = get_offline_model(src_name, train_mav, train_labels, num_classes,
                                                 seed, held_out, train_names, use_cache)
                    pm = None if needs_per_k_quant(name) else derive_quantized(
                        name, base, num_classes, calib_batches=calib_dl, train_dl=qat_dl)

                # Prototype source: the FP32 base for the proto_precision="fp32"
                # ablation, otherwise the (quantized) model itself.
                proto_src = base if uses_fp32_protos(name) else pm

                if pm is None:
                    # calib_source="support": quantization depends on k, so it happens
                    # inside the sweep. A "generic/factory floor" is not defined for it
                    # -- calibrating observers on the user's own shots means there is no
                    # factory state to measure. Recorded as NaN, rendered "n/a".
                    floors[f"{_short(name)}_before"].append(float("nan"))
                else:
                    # generic prototypes (the "factory" floor) from the train sessions
                    before = set_protos_and_acc(pm, gen_dl, qx, qy, proto_src=proto_src)
                    floors[f"{_short(name)}_before"].append(before)

                proto_trained[name] = (pm, base)
                cached_flags.append(ci)

            src = "cached" if all(cached_flags) else ("trained" if not any(cached_flags) else "mixed")
            floor_bits = ([f"CNN {_pct(floors['cnn_zeroshot'][-1])}"] if with_cnn else [])
            floor_bits += [f"{_variant(n)} {_pct(floors[_short(n)+'_before'][-1])}" for n in models]
            logger.info(f"  offline models: {src}  |  floors -> " + "  ".join(floor_bits))

            # -- k sweep (calibration), k = number of support reps -----------
            k_values_fold = [k for k in k_values if k <= len(support_pool)]
            for k in k_values_fold:
                s_idx  = support_indices(test_reps, support_pool, k)
                sx, sy = test_mav[s_idx], test_labels[s_idx]

                bits = []
                if with_cnn:
                    # CNN backprop fine-tune on the k support reps. This is the
                    # bottleneck of the whole harness (FT_EPOCHS full-batch passes
                    # over every support window, per k, per fold) -- --skip-cnn
                    # exists because it is a fixed baseline that does not change
                    # when a new proto variant is added.
                    ft = finetune_cnn(cnn, sx, sy, FT_EPOCHS, FT_LR)
                    swept["cnn_finetune"][k].append(predict_acc(ft, qx, qy))
                    bits.append(f"CNN-ft {swept['cnn_finetune'][k][-1]:.1%}")

                # Prototypes = mean embedding of the k support reps (no backprop)
                support_batch = [(torch.from_numpy(sx), torch.from_numpy(sy))]
                for name, (pm, base) in proto_trained.items():
                    model = pm
                    if model is None:
                        # calib_source="support": re-quantize from the shared FP32 base
                        # using THIS k's support set as the observer calibration data.
                        # Cheap (fuse+calibrate+convert), because stage 1 is already done.
                        model = derive_quantized(name, base, num_classes,
                                                 calib_batches=support_batch)
                    proto_src = base if uses_fp32_protos(name) else model
                    acc = set_protos_and_acc(model, support_batch, qx, qy, proto_src=proto_src)
                    swept[f"{_short(name)}_after"][k].append(acc)
                    bits.append(f"{_variant(name)} {acc:.1%}")

                logger.info(f"  k={k}rep ({len(s_idx):,} win)  " + "  ".join(bits))

    elapsed = time.perf_counter() - start

    # -- Report ---------------------------------------------------------------
    md = build_markdown(floors, swept, models, seeds, k_values, n_folds, elapsed, with_cnn)
    print("\n" + md)

    RESULTS_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Few-shot leave-one-session-out log\n\n"
                            "Most recent runs are appended below.\n\n", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(md + "\n")
    logger.info(f"Appended summary to: {LOG_PATH}")

    if make_plot:
        png = save_plot(floors, swept, models, k_values,
                        RESULTS_DIR / "fewshot_loso_curve.png", with_cnn)
        if png:
            logger.info(f"Saved curve to: {png}")

    logger.info(f"Total elapsed: {fmt_duration(elapsed)}")
    return floors, swept


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Leave-one-session-out few-shot evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s\n"
            "      full run: every proto model + the CNN baseline, all seeds, full k sweep\n"
            "  %(prog)s --models EmagerCNNProtoRingStridedPTQ --skip-cnn --k 1,3,5\n"
            "      score one new variant against the existing CNN numbers (~15 min)\n"
            "  %(prog)s --quick\n"
            "      1 seed, k in {1,2}, no CNN baseline -- a plumbing check\n"
        ),
    )
    p.add_argument("--models", type=lambda s: [m for m in s.split(",") if m],
                   default=None, metavar="A,B",
                   help=f"comma-separated proto models to evaluate (default: {', '.join(PROTO_MODELS)})")
    p.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",") if v],
                   default=None, metavar="42,123",
                   help=f"comma-separated seeds (default: {SEEDS})")
    p.add_argument("--k", dest="k_values", type=lambda s: [int(v) for v in s.split(",") if v],
                   default=None, metavar="1,3,5",
                   help=f"comma-separated calibration reps/class to sweep (default: {K_VALUES})")
    p.add_argument("--skip-cnn", action="store_true",
                   help="skip the CNN zero-shot floor and fine-tune sweep. This is the "
                        "bulk of the wall time and it is a fixed baseline, so skip it when "
                        "measuring a new proto variant against an earlier run's CNN rows.")
    p.add_argument("--quick", action="store_true",
                   help="plumbing check: 1 seed, k in {1,2}, implies --skip-cnn")
    p.add_argument("--fresh", action="store_true", help="ignore cached offline models and retrain")
    p.add_argument("--plot", action="store_true", help="also save a PNG of the k curve")

    args = p.parse_args(argv)
    if args.quick:
        args.seeds    = args.seeds or [42]
        args.k_values = args.k_values or [1, 2]
        args.skip_cnn = True

    unknown = [m for m in (args.models or []) if not hasattr(new_models, m)]
    if unknown:
        p.error(f"unknown model(s): {', '.join(unknown)}")
    return args


if __name__ == "__main__":
    a = parse_args()
    main(models=a.models, seeds=a.seeds, k_values=a.k_values, with_cnn=not a.skip_cnn,
         make_plot=a.plot, use_cache=not a.fresh)
