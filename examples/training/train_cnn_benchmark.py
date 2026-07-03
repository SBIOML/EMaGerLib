from pathlib import Path
from datetime import datetime
import json
import logging
import platform
import random
import time

import lightning
import numpy as np

logger = logging.getLogger(__name__)

# Every run is persisted here: one timestamped .json per run (full fidelity)
# plus an appended section in RESULTS_LOG.md (human-readable history).
RESULTS_DIR = Path(__file__).parent / "benchmark_results"

# -- Configuration -----------------------------------------------------------
# Each dataset folder must contain a collection_details.json.
# num_classes and num_reps are read from that file automatically.
# Leave-one-out cross-validation across reps: for each held-out rep r,
# train on all other reps and test on r. Accuracy is averaged over
# (num_reps x num_seeds) runs per (model, dataset).

BASE_PATH = Path(__file__).parent.parent.parent / "Datasets"

DATASETS = [
    "Test_EM_C7_R5",
    "Test_EM_C7_R5_02",
    "Test_EM_C7_R5_03",
]

# Each model is trained once per seed. Results reported as mean +/- std.
# Set to a single value (e.g. [42]) for a quick one-shot run.
SEEDS = [42, 123, 456]

# Comment / uncomment to select which variants to include.
MODELS_TO_TEST = [
    # "EmagerCNNBase",
    # "EmagerCNNWide",
    # "EmagerCNNDeep",
    # "EmagerCNNLight",
    # "EmagerCNNStrided",
    "EmagerCNNCircular",          # pending re-measure (rewritten to W-only ring pad)
    "EmagerCNNRingStrided",       # pending: never benchmarked
    # "EmagerCNNGAP",
    "EmagerCNNQuantizedPTQ",      # pending: only 1-dataset/1-seed smoke run so far
    "EmagerCNNQuantizedQAT",      # pending: only smoke-tested so far
    "EmagerCNNRingStridedQAT",    # pending: only smoke-tested so far
    # "EmagerCNNProtoEpisodic",     # few-shot: episodic prototypical training
    # "EmagerCNNProtoCE",           # few-shot: cross-entropy pretrain + computed prototypes
]

WINDOW_SIZE      = 200
WINDOW_INCREMENT = 10
MAX_EPOCHS       = 10 # 10 usually
BATCH_TRAIN      = 64
BATCH_TEST       = 256
SAMPLING         = 2000

# Rough wall-time per single train+eval fit, used only for the upfront time
# estimate printed before the run. Tune to your hardware (~20s on a midrange
# GPU, ~60-90s on CPU for ~10 epochs).
EST_SECONDS_PER_FIT = 45

# ----------------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def read_dataset_info(dataset_path: Path):
    cj = dataset_path / "collection_details.json"
    if not cj.exists():
        raise FileNotFoundError(f"No collection_details.json in {dataset_path}")
    d = json.loads(cj.read_text())
    return d["num_motions"], d["num_reps"]


def load_and_filter(dataset_path: Path, num_classes: int, num_reps: int, sampling: int):
    """Load raw EMG from disk and apply notch + bandpass filters."""
    from libemg.data_handler import OfflineDataHandler, RegexFilter
    from libemg.filtering import Filter

    odh = OfflineDataHandler()
    odh.get_data(
        folder_location=str(dataset_path),
        regex_filters=[
            RegexFilter(left_bound="C_", right_bound="_",        values=[str(i) for i in range(num_classes)], description="classes"),
            RegexFilter(left_bound="R_", right_bound="_emg.csv", values=[str(i) for i in range(num_reps)],    description="reps"),
        ],
    )
    filt = Filter(sampling)
    filt.install_filters({"name": "notch",    "cutoff": 60,        "bandwidth": 3})
    filt.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    filt.filter(odh)
    return odh


def rep_data_stats(odh, sampling: int):
    """
    Recording size per rep. Each entry in odh.data is one (class, rep) recording
    shaped (n_samples, n_channels) -- i.e. one repetition of one gesture. The mean
    length over all recordings gives the datapoints per rep, and dividing by the
    sampling rate gives the seconds of recording per rep.

    Returns (avg_samples_per_rep, avg_seconds_per_rep), or (None, None) if empty.
    """
    counts = [len(arr) for arr in odh.data]
    if not counts:
        return None, None
    avg_samples = float(np.mean(counts))
    return avg_samples, avg_samples / sampling


def _fmt_macs(n):
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def compute_macs(model, input_shape) -> int:
    """
    Count multiply-accumulates (MACs) for one forward pass at batch=1.
    Only Conv2d and Linear are counted -- they dominate compute for these
    models (BN/ReLU/Flatten contribute <1%). Must run BEFORE any quantization:
    once nn.Conv2d / nn.Linear are replaced with their quantized variants, the
    hooks below stop firing.
    """
    import torch
    import torch.nn as nn
    macs = [0]
    handles = []

    def conv_hook(m, inp, out):
        _, c_out, h_out, w_out = out.shape
        k_h, k_w = m.kernel_size
        c_in_per_group = m.in_channels // m.groups
        macs[0] += h_out * w_out * c_out * c_in_per_group * k_h * k_w

    def linear_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, int(np.prod(input_shape)))
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return macs[0]


def compute_latency_ms(model, input_shape, num_threads: int = 1) -> float:
    """
    Median single-sample forward latency in milliseconds, measured single-threaded.
    Quantized models will use fbgemm / qnnpack INT8 kernels here -- this is the
    column that reflects the quantization speedup. (FLOPs/MACs do not.)
    Host-CPU x86 timings; absolute numbers won't match an ARM deployment, but
    the relative ordering between models is informative.
    """
    import torch
    from torch.utils import benchmark
    model.eval()
    x = torch.zeros(1, int(np.prod(input_shape)))
    with torch.no_grad():
        for _ in range(10):
            model(x)
    t = benchmark.Timer(
        stmt="with torch.no_grad():\n    model(x)",
        globals={"model": model, "x": x, "torch": torch},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=0.5)
    return t.median * 1000.0


def print_results_table(results: dict, datasets: list, models: list, seeds: list, model_stats: dict):
    col_w   = 22
    ds_w    = 20
    ov_w    = 14
    param_w = 14
    size_w  = 12
    macs_w  = 12
    lat_w   = 12

    overall = {}
    for m in models:
        all_accs = [a for d in datasets for a in results[m].get(d, [])]
        overall[m] = np.mean(all_accs) if all_accs else float("nan")

    header = (
        "Model".ljust(col_w)
        + "".join(d[:ds_w].center(ds_w) for d in datasets)
        + "Overall".center(ov_w)
        + "Params".rjust(param_w)
        + "Size (KB)".rjust(size_w)
        + "MACs".rjust(macs_w)
        + "Lat (ms)".rjust(lat_w)
    )
    sep    = "-" * len(header)

    logger.info("=" * len(header))
    logger.info(f"Benchmark results  (LOO across reps, seeds: {seeds})")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)

    for m in sorted(models, key=lambda x: overall[x], reverse=True):
        row = m.ljust(col_w)
        for d in datasets:
            accs = results[m].get(d, [])
            cell = f"{np.mean(accs):.1%}+/-{np.std(accs):.1%}" if accs else "N/A"
            row += cell.center(ds_w)
        ov_str = f"{overall[m]:.1%}" if not np.isnan(overall[m]) else "N/A"
        row += ov_str.center(ov_w)
        params, size_kb, macs, latency = model_stats.get(m, (None, None, None, None))
        row += (f"{params:,}".rjust(param_w)   if params  is not None else "N/A".rjust(param_w))
        row += (f"{size_kb:,.1f}".rjust(size_w) if size_kb is not None else "N/A".rjust(size_w))
        row += (_fmt_macs(macs).rjust(macs_w)   if macs    is not None else "N/A".rjust(macs_w))
        row += (f"{latency:.2f}".rjust(lat_w)   if latency is not None else "N/A".rjust(lat_w))
        logger.info(row)

    logger.info(sep)


def _build_markdown_section(record: dict) -> str:
    """Render one run as a markdown section (config block + sorted results table)."""
    datasets = record["datasets"]
    models   = record["models"]
    cfg      = record["config"]
    host     = record["host"]

    # sort by overall accuracy, unknowns last
    order = sorted(models, key=lambda m: (models[m]["overall_acc"] is None,
                                          -(models[m]["overall_acc"] or 0.0)))

    lines = [f"\n## {record['timestamp']}\n"]

    # Datasets with their class/rep counts (older records may lack this).
    ds_info = record.get("dataset_info", {})
    if ds_info:
        lines.append("- **Datasets** (leave-one-out across reps):")
        sr = cfg.get("sampling")
        for d in datasets:
            info = ds_info.get(d, {})
            bullet = (
                f"    - `{d}` — {info.get('num_classes', '?')} classes, "
                f"{info.get('num_reps', '?')} reps"
            )
            spr = info.get("samples_per_rep")
            sec = info.get("seconds_per_rep")
            if spr is not None:
                bullet += f" · ~{spr:,.0f} datapoints/rep"
                if sec is not None:
                    bullet += f" (~{sec:.1f}s" + (f" @ {sr} Hz)" if sr else ")")
            lines.append(bullet)
    else:
        lines.append(f"- **Datasets:** {', '.join(datasets) or 'none'}")

    lines.append(f"- **Seeds:** {record['seeds']}")
    lines.append(
        f"- **Epochs:** {cfg.get('max_epochs')}  |  "
        f"**Window:** {cfg.get('window_size')}/{cfg.get('window_increment')}  |  "
        f"**Sampling:** {cfg.get('sampling')}  |  "
        f"**Batch:** {cfg.get('batch_train')}/{cfg.get('batch_test')}"
    )

    # Fit count + average wall time per fit (one fit = one rep/fold × model × seed).
    total_fits = record.get("total_fits")
    if total_fits is not None:
        fit_line = f"- **Fits:** {total_fits} total"
        avg_fit = record.get("avg_fit_s")
        if avg_fit is not None:
            fit_line += f"  ·  {avg_fit:.1f}s avg/fit (incl. data load)"
        lines.append(fit_line)

    lines.append(f"- **Config file:** {cfg.get('config_file') or '(constants in script)'}")
    lines.append(f"- **Host:** {host['device']} · torch {host['torch']} · py {host['python']} · {host['platform']}")
    lines.append(f"- **Elapsed:** {fmt_duration(record['elapsed_s'])}\n")

    cols = ["Model"] + datasets + ["Overall", "Params", "Size (KB)", "MACs", "Lat (ms)"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for m in order:
        md  = models[m]
        row = [m]
        for d in datasets:
            pd = md["per_dataset"][d]
            row.append(f"{pd['mean']:.1%} ± {pd['std']:.1%}" if pd["mean"] is not None else "N/A")
        row.append(f"{md['overall_acc']:.1%}" if md["overall_acc"] is not None else "N/A")
        row.append(f"{md['params']:,}"        if md["params"]      is not None else "N/A")
        row.append(f"{md['size_kb']:,.1f}"     if md["size_kb"]     is not None else "N/A")
        row.append(_fmt_macs(md["macs"]))
        row.append(f"{md['latency_ms']:.2f}"   if md["latency_ms"]  is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def save_results(results, datasets, models, seeds, model_stats, hyperparams, elapsed_s,
                 dataset_meta=None, total_fits=None):
    """
    Persist a run to disk:
      - benchmark_results/<timestamp>.json  -- complete structured record
      - benchmark_results/RESULTS_LOG.md    -- appended human-readable section

    dataset_meta: {name: {"num_classes": int, "num_reps": int}} for the per-dataset
    breakdown. total_fits: number of train+eval runs in this benchmark.

    Returns (json_path, md_path).
    """
    import torch

    RESULTS_DIR.mkdir(exist_ok=True)
    now   = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")

    def _stats(m):
        return model_stats.get(m, (None, None, None, None))

    def _overall(m):
        accs = [a for d in datasets for a in results[m].get(d, [])]
        return float(np.mean(accs)) if accs else None

    record = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed_s, 1),
        "total_fits": total_fits,
        "avg_fit_s": round(elapsed_s / total_fits, 1) if total_fits else None,
        "host": {
            "platform": platform.platform(),
            "python":   platform.python_version(),
            "torch":    torch.__version__,
            "device":   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "config":   hyperparams,
        "datasets": datasets,
        "dataset_info": dataset_meta or {},
        "seeds":    seeds,
        "models": {
            m: {
                "params":      _stats(m)[0],
                "size_kb":     _stats(m)[1],
                "macs":        _stats(m)[2],
                "latency_ms":  _stats(m)[3],
                "overall_acc": _overall(m),
                "per_dataset": {
                    d: {
                        "accs": [float(a) for a in results[m].get(d, [])],
                        "mean": float(np.mean(results[m][d])) if results[m].get(d) else None,
                        "std":  float(np.std(results[m][d]))  if results[m].get(d) else None,
                    }
                    for d in datasets
                },
            }
            for m in models
        },
    }

    json_path = RESULTS_DIR / f"{stamp}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    md_path = RESULTS_DIR / "RESULTS_LOG.md"
    if not md_path.exists():
        md_path.write_text("# Benchmark results log\n\nMost recent runs are appended below.\n", encoding="utf-8")
    with md_path.open("a", encoding="utf-8") as f:
        f.write(_build_markdown_section(record))

    return json_path, md_path


def show_results(n: int = 3):
    """
    Print the most recent N saved runs to the terminal (newest last), reusing the
    same table format as RESULTS_LOG.md. Lets you review past results without
    re-running anything or opening the log file. Use: `... --show [N]`.
    """
    if not RESULTS_DIR.exists():
        print(f"No results yet -- {RESULTS_DIR} does not exist. Run the benchmark first.")
        return
    files = sorted(RESULTS_DIR.glob("*.json"))  # timestamped names sort chronologically
    if not files:
        print(f"No saved runs found in {RESULTS_DIR}.")
        return
    selected = files[-n:]
    print(f"Showing {len(selected)} of {len(files)} saved run(s) from {RESULTS_DIR}:")
    for path in selected:
        try:
            print(_build_markdown_section(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  (skipped {path.name}: {e})")


def main(datasets=None, seeds=None, models=None, config=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import emagerlib.models.new_emager_cnn as new_models

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # If a config file is provided, pull training hyperparams from it.
    # Dataset selection is always controlled by the datasets param / DATASETS constant.
    if config is not None:
        from emagerlib.config.load_config import load_config as _load_config
        _cfg             = _load_config(config)
        _window_size      = _cfg.WINDOW_SIZE
        _window_increment = _cfg.WINDOW_INCREMENT
        _max_epochs       = _cfg.EPOCH
        _sampling         = _cfg.SAMPLING
    else:
        _window_size      = WINDOW_SIZE
        _window_increment = WINDOW_INCREMENT
        _max_epochs       = MAX_EPOCHS
        _sampling         = SAMPLING

    _datasets = datasets if datasets is not None else DATASETS
    _seeds    = seeds    if seeds    is not None else SEEDS
    _models   = models   if models   is not None else MODELS_TO_TEST

    start_time = time.perf_counter()

    # Pre-scan dataset info so we can quote a total fit count + time estimate.
    dataset_info = []
    for dataset_name in _datasets:
        dataset_path = BASE_PATH / dataset_name
        try:
            num_classes, num_reps = read_dataset_info(dataset_path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset_name}: {e}")
            continue
        dataset_info.append((dataset_name, dataset_path, num_classes, num_reps))

    total_fits = sum(nr for _, _, _, nr in dataset_info) * len(_models) * len(_seeds)
    est_seconds = total_fits * EST_SECONDS_PER_FIT
    logger.info(
        f"Planned: {total_fits} fits "
        f"({len(dataset_info)} dataset(s) x reps x {len(_models)} model(s) x {len(_seeds)} seed(s))"
    )
    logger.info(f"Estimated wall time: ~{fmt_duration(est_seconds)}  (assuming {EST_SECONDS_PER_FIT}s/fit)")

    # results[model][dataset] = [acc_fold0_seed0, acc_fold0_seed1, ...]
    results = {m: {} for m in _models}
    # model_stats[model] = (param_count, state_dict_size_kb, macs, latency_ms)
    model_stats: dict = {}

    fit_idx = 0  # global counter across all datasets/models/seeds/folds
    dataset_meta: dict = {}  # name -> {num_classes, num_reps, samples_per_rep, seconds_per_rep}

    for di, (dataset_name, dataset_path, num_classes, num_reps) in enumerate(dataset_info, 1):
        logger.info("=" * 60)
        logger.info(f"Dataset {di}/{len(dataset_info)}: {dataset_name}")
        logger.info("=" * 60)
        logger.info(f"  {num_classes} classes, {num_reps} reps -- leave-one-out across reps")

        # -- 1. Load & filter raw EMG (once per dataset) ----------------------
        odh = load_and_filter(dataset_path, num_classes, num_reps, _sampling)

        # Raw signal volume per rep (datapoints + seconds), recorded in the log.
        samples_per_rep, seconds_per_rep = rep_data_stats(odh, _sampling)
        dataset_meta[dataset_name] = {
            "num_classes":     num_classes,
            "num_reps":        num_reps,
            "samples_per_rep": samples_per_rep,
            "seconds_per_rep": seconds_per_rep,
        }
        if samples_per_rep is not None:
            logger.info(
                f"  ~{samples_per_rep:,.0f} datapoints/rep "
                f"(~{seconds_per_rep:.1f}s @ {_sampling} Hz)"
            )

        # -- 2. Pre-compute MAV features for each LOO fold --------------------
        # MAV collapses the time axis; the CNN sees a 4x16 map of signal energy.
        folds = []
        for held_out in range(num_reps):
            # -- 2a. Isolate train/test data for this fold (held-out rep) --
            train_reps = [r for r in range(num_reps) if r != held_out]
            train_odh = odh.isolate_data("reps", train_reps)
            test_odh  = odh.isolate_data("reps", [held_out])

            # -- 2b. Parse windows and compute MAV features (rebuilt per fold for deterministic shuffle) --
            train_windows, train_meta = train_odh.parse_windows(_window_size, _window_increment)
            test_windows,  test_meta  = test_odh.parse_windows( _window_size, _window_increment)

            # -- 2c. Sample windows data dict for this fold (rebuilt per fold for deterministic shuffle) --
            folds.append({
                "held_out":     held_out,
                "train_mav":    np.mean(np.abs(train_windows), axis=2),   # (N, 64)
                "train_labels": train_meta["classes"],
                "test_mav":     np.mean(np.abs(test_windows),  axis=2),
                "test_labels":  test_meta["classes"],
                "train_shape":  train_windows.shape,
                "test_shape":   test_windows.shape,
            })
        logger.info(
            f"  Built {len(folds)} LOO folds "
            f"(fold 0: train {folds[0]['train_shape']}, test {folds[0]['test_shape']})"
        )

        for mi, model_name in enumerate(_models, 1):
            logger.info("-" * 50)
            logger.info(f"  >>> MODEL {mi}/{len(_models)}: {model_name}  (dataset {di}/{len(dataset_info)}: {dataset_name})")
            logger.info("-" * 50)
            run_accs = []

            for seed in _seeds:
                for fold in folds:
                    fit_idx += 1
                    # -- 3. Build DataLoaders (rebuilt per seed for deterministic shuffle) --
                    lightning.seed_everything(seed)
                    train_dl = DataLoader(
                        TensorDataset(
                            torch.from_numpy(fold["train_mav"].astype(np.float32)),
                            torch.from_numpy(fold["train_labels"]),
                        ),
                        batch_size=BATCH_TRAIN, shuffle=True, drop_last=True,
                    )
                    test_dl = DataLoader(
                        TensorDataset(
                            torch.from_numpy(fold["test_mav"].astype(np.float32)),
                            torch.from_numpy(fold["test_labels"]),
                        ),
                        batch_size=BATCH_TEST, shuffle=False, drop_last=False,
                    )

                    # -- 4. Train & evaluate ----------------------------------
                    logger.info(
                        f"  [fit {fit_idx}/{total_fits}] >>> {model_name} "
                        f"| {dataset_name} | seed={seed} fold={fold['held_out']} ... training"
                    )
                    model_class = getattr(new_models, model_name)
                    model = model_class((4, 16), num_classes)

                    # MACs and param count are taken on the FP32 model BEFORE fit:
                    # once EmagerCNNQuantizedPTQ swaps in INT8 ops the Conv2d/Linear
                    # hooks stop firing (MACs), and its quantized weights move into
                    # opaque _packed_params that don't surface as nn.Parameters,
                    # which under-reports the true architectural param count.
                    first_time = model_name not in model_stats
                    macs   = compute_macs(model, (4, 16)) if first_time else None
                    params = sum(p.numel() for p in model.parameters()) if first_time else None

                    res = model.fit(train_dl, test_dl, max_epochs=_max_epochs)

                    # Size + latency are measured AFTER fit so the quantized
                    # variant's INT8 state_dict and INT8 kernel speed show up.
                    if first_time:
                        # Real serialized size of the state_dict. Must NOT sum
                        # numel*element_size over tensors: quantized layers store
                        # their INT8 weights in packed _packed_params (not plain
                        # tensors), so that sum silently drops them and reports a
                        # near-constant ~35 KB for every quantized model regardless
                        # of architecture. torch.save captures the packed weights.
                        import io
                        _buf = io.BytesIO()
                        torch.save(model.state_dict(), _buf)
                        size_kb    = _buf.tell() / 1024
                        latency_ms = compute_latency_ms(model, (4, 16))
                        model_stats[model_name] = (params, size_kb, macs, latency_ms)
                        logger.info(
                            f"    params: {params:,}  size: {size_kb:,.1f} KB  "
                            f"MACs: {_fmt_macs(macs)}  latency: {latency_ms:.2f} ms"
                        )
                    acc = res[0]["test_acc"]
                    run_accs.append(acc)
                    logger.info(
                        f"  [fit {fit_idx}/{total_fits}] <<< {model_name} "
                        f"| {dataset_name} | seed={seed} fold={fold['held_out']} -> acc={acc:.1%}"
                    )

            results[model_name][dataset_name] = run_accs
            mean = np.mean(run_accs)
            std  = np.std(run_accs)
            logger.info(f"  {model_name:<22} {mean:.1%} +/- {std:.1%}  (n={len(run_accs)})")

    # -- 5. Print results table -----------------------------------------------
    final_datasets = [d[0] for d in dataset_info]
    print_results_table(results, final_datasets, _models, _seeds, model_stats)

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 60)
    logger.info(f"Total elapsed time:  {fmt_duration(elapsed)}")
    logger.info(f"Estimated was:       ~{fmt_duration(est_seconds)}  ({EST_SECONDS_PER_FIT}s/fit)")
    if total_fits > 0:
        logger.info(f"Actual per fit:      {elapsed/total_fits:.1f}s  ({total_fits} fits)")

    # -- 6. Persist results so runs aren't lost to terminal scrollback --------
    hyperparams = {
        "window_size":      _window_size,
        "window_increment": _window_increment,
        "max_epochs":       _max_epochs,
        "sampling":         _sampling,
        "batch_train":      BATCH_TRAIN,
        "batch_test":       BATCH_TEST,
        "config_file":      config,
    }
    json_path, md_path = save_results(
        results, final_datasets, _models, _seeds, model_stats, hyperparams, elapsed,
        dataset_meta=dataset_meta, total_fits=total_fits,
    )
    logger.info(f"Saved full record:   {json_path}")
    logger.info(f"Appended summary to: {md_path}")

    return results


if __name__ == "__main__":
    import sys

    # --show [N]: print the last N saved runs and exit (default 3). No training.
    if "--show" in sys.argv:
        # The markdown uses non-ASCII glyphs (+/- as ±, separators as ·); force
        # UTF-8 so the Windows console doesn't render them as "?".
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        i = sys.argv.index("--show")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 3
        show_results(n)
        sys.exit(0)

    _config = None
    if "--config" in sys.argv:
        _config = sys.argv[sys.argv.index("--config") + 1]
    main(config=_config)
