# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EMaGerLib is an EMG (electromyography) toolbox for the EMaGer v1/v3 acquisition
devices: real-time gesture recognition and prosthetic-hand control. It is built
**on top of a fork of `libemg`**, not the upstream package (see below). The repo
is both an installable library (`emagerlib/`) and a set of runnable scripts
(`examples/`), both shipped in the same distribution.

## Install & environment

```bash
pip install -e .          # pulls libemg fork @ emager-v3, registers console scripts
```

`pip install -e .` installs `libemg` from `git+https://github.com/Michiboi29/libemg-fork.git@emager-v3`,
**not** PyPI. The fork is mandatory: `emagerlib/utils/streamer_utils.py` and the
v3 visualization scripts call `emagerv3_streamer`, which exists only in the fork
(`emager-v3` and `emager-v3-update` branches). Switching libemg source is a manual
two-step (`pip uninstall libemg -y` then `pip install ...`) — pip extras cannot
override a URL-pinned base dependency, so do not add `[pypi]`/`[latest]` extras back.

## Commands

Everything routes through the unified `emager` CLI. **The source of truth for which
subcommands actually run is the `COMMANDS` dict in [examples/main.py](examples/main.py)**,
not the README table — several commands (hand control, some realtime/visualize
variants) are commented out there and will error as "Unknown command" even though
the README lists them.

```bash
emager                       # list commands / help
emager <command> -c my_config.py   # every command takes -c/--config and --log-level
emager train-cnn
emager realtime-predict --log-level DEBUG
emager gui                   # PyQt desktop launcher that wraps all commands (examples/gui/command_launcher.py)
python examples/training/train_cnn.py   # scripts are also runnable directly
```

All subcommands share one argument parser (`emagerlib/utils/arg_parser.py`):
`-c/--config`, `--log-level`, `--save-config-name`, etc. A subcommand's `main()`
calls `setup_runtime()` → `load_config()` → `setup_logging()`.

## Tests

Plain `unittest` (no pytest configured, no `dev` deps yet).

```bash
emager-run-tests                       # discovers tests/test_*.py
python -m unittest tests.test_utils    # one module
python -m unittest tests.test_utils.TestUtilityFunctions.test_01_get_transform_decimation  # one test
```

Tests cover config load/save, arg parsing, model discovery, gestures/constants, and
a libemg-integration smoke test — not the models, hardware, or realtime paths.

## Architecture (the parts that span multiple files)

**Config system** (`emagerlib/config/`). One `CoreConfig` dataclass is loaded from
`.py`, `.yaml`, or `.json` by `load_config()`. Python configs are imported as
modules; unknown keys are captured as `extra` rather than rejected, and the
`${ROOT_EMAGERLIB}` placeholder + env vars are expanded. `ROOT_EMAGERLIB`
(`emagerlib/__init__.py`) is the repo root — configs use it to build paths.
Start from [config_examples/base_config_example.py](config_examples/base_config_example.py).
`EMAGER_VERSION` in the config picks the hardware streamer.

**Streamer dispatch** (`emagerlib/utils/streamer_utils.py`). `get_emager_streamer(version)`
maps `EMAGER_VERSION` → a libemg streamer: `v1.0`/`v1.1` → `emager_streamer`,
`v3.0` → `emagerv3_streamer` (fork-only). This is the single seam between config
and hardware.

**Models** (`emagerlib/models/`) — three files, deliberately layered:
- `models.py` — **legacy**, original `EmagerCNN`/`EmagerSCNN` (non-standard Conv→ReLU→BN
  order, quantization branches inline). Kept only as reference; note `examples/training/train_cnn.py`
  still imports it. Do not build new work on it.
- `emager_cnn.py` — **stable** clean reference baseline (Conv→BN→ReLU).
- `new_emager_cnn.py` — **experimental** variant zoo. All variants share the
  Lightning + LibEMG interface (`fit` / `predict` / `predict_proba`) via `_EmagerBase`;
  most variants change only `__init__`. There are three subclass families:
  architecture variants (Base/Wide/Deep/Light/Strided/Circular/RingStrided/GAP),
  an INT8 quantization chain `EmagerCNNQuantizedPTQ → …QAT → …RingStridedQAT` (these
  override `fit()` to fuse+quantize), and a few-shot prototypical chain
  `_EmagerProtoBase → EmagerCNNProtoEpisodic / …ProtoCE` (no learned classifier;
  classify by distance to prototypes computed from a support set).

  **[emagerlib/models/models.md](emagerlib/models/models.md) is the authoritative model doc** —
  read it before adding, benchmarking, or reasoning about any variant. It documents
  every architecture, the quantization/few-shot flows, param/size/accuracy tables,
  and how to add a new variant.

**Control** (`emagerlib/control/`). Prosthetic-hand backends behind
`abstract_hand_control.py`: Psyonic direct serial (`psyonic_control.py`), Psyonic
via Teensy UART bridge (`psyonic_teensy_control.py`), BLE (`ble_client.py`),
plus `gesture_decoder.py` mapping predictions → hand commands.

**Data/training flow** (in the `examples/training/` scripts): libemg
`OfflineDataHandler` + `RegexFilter` load CSV reps → notch(60Hz)+bandpass(20–450Hz)
filter → `parse_windows` → MAV feature → `TensorDataset`/`DataLoader` → model `.fit()`.

## Benchmark harnesses (model-development work happens here)

- `examples/training/train_cnn_benchmark.py` — **leave-one-REP-out** within a session,
  swept over seeds `[42,123,456]`. Reports params / serialized size / MACs / latency.
  Results append to `examples/training/benchmark_results/RESULTS_LOG.md`.
- `examples/training/eval_fewshot_loso.py` — **leave-one-SESSION-out** few-shot eval;
  the only harness where the few-shot idea can pay off (rep-LOO has ~no distribution
  shift, so before≈after there). Compares gradient-free k-shot prototypes vs CNN
  fine-tuning at equal calibration budget. Caches offline models under
  `benchmark_results/model_cache/`; `--fresh` retrains, `--quick` shrinks the sweep.
  Results append to `benchmark_results/FEWSHOT_LOSO_LOG.md`.

Both derive their data pipeline from `train_cnn_benchmark.py` (`eval_fewshot_loso.py`
imports `load_and_filter`, `read_dataset_info`, `BASE_PATH` from it).

## Gotchas

- **Windows torch/Qt DLL order.** `import libemg` transitively loads torch; if a Qt
  binding (PyQt5/6) is imported *before* libemg, torch's `c10.dll` throws
  `WinError 1114`. In any script touching both, import libemg first. The GUI launcher
  also scrubs `PyQt6\Qt6\bin` from the subprocess PATH for the same reason.
- **libemg source matters.** Any change to a `emagerv3_streamer` code path breaks
  under PyPI/upstream libemg (options 3/4 in the README). Keep the fork installed.
- Quantized models store weights as packed `_packed_params` after `fit()`, so counting
  params post-`fit()` under-reports — the benchmark counts on the FP32 model before
  `fit()`. See the "size caveat" in models.md before touching size/param reporting.

## Conventions

- Use `logging` (module-level `logger = logging.getLogger(__name__)`), not `print`,
  in library and script code.
- New models go in `new_emager_cnn.py` inheriting `_EmagerBase` (or the PTQ/QAT/Proto
  base for those families), with a matching row + section added to `models.md`.
- `CHANGELOG.md` is kept (Keep-a-Changelog style); the project is on SemVer, currently
  1.2.1 with an `[Unreleased]` section.
