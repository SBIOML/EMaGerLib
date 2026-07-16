# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Few-shot prototypical models on the RingStrided architecture** in `emagerlib/models/new_emager_cnn.py` (experimental) — the MCU-targeted deployment line, putting the gradient-free prototype rule on the smallest architecture in the file:
  - `EmagerCNNProtoRingStrided`: episodic prototypical network on the `EmagerCNNRingStrided` backbone. Subclasses `EmagerCNNProtoEpisodic` and overrides only `__init__`; the strided collapse shrinks the embedding head from `Linear(2048→64)` to `Linear(128→64)`, cutting **167 K → 44 K params (~3.8×)** for the same prototype rule.
  - `EmagerCNNProtoRingStridedPTQ`: INT8 post-training quantization of the embedding. **~58 KB serialized.** The recommended deployment target.
  - `EmagerCNNProtoRingStridedQAT`: INT8 QAT counterpart; the fake-quant fine-tuning uses the episodic loss, so the embedding adapts to INT8 error *while* being optimized for the mean-prototype rule.

  The quantized variants bracket **only the embedding** with Quant/DeQuant stubs and keep the prototype distance in FP32 — it is 448 MACs against ~152 K in the conv stack, and it mirrors the MCU split (INT8 kernel for the convs, float for the nearest-prototype lookup). Prototypes are computed *after* `convert()`, through the INT8 embedding, matching what the device does; a consequence is that calibration is inherently quantization-aware with no extra machinery. Documented in [`models.md`](emagerlib/models/models.md#few-shot-on-the-ringstrided-architecture-deployment-line).
- `--models`, `--seeds`, `--k` and `--skip-cnn` flags on `examples/training/eval_fewshot_loso.py` (now a real `argparse` CLI with `--help`). `--skip-cnn` drops the CNN zero-shot floor and fine-tune sweep, which is where essentially all of the harness's wall time goes (`FT_EPOCHS` full-batch backprop passes over every support window, per k, per fold — the 2026-07-15 Felix run spent ~6h45m of its 6h47m there). The CNN baseline is a fixed reference that does not move when a proto variant is added, so scoring a new variant against an earlier run's CNN rows takes **~15 min instead of ~7h**.

- `EmagerCNNRingStridedPTQ` in `emagerlib/models/new_emager_cnn.py` (experimental): the post-training-quantization counterpart to `EmagerCNNRingStridedQAT` — same RingStrided architecture (strided spatial collapse + column-only ring padding), but INT8 is applied *after* FP32 training (fuse → calibrate → convert) instead of via QAT fine-tuning. Subclasses `EmagerCNNQuantizedPTQ`, completing the {Base, RingStrided} × {PTQ, QAT} quantization grid. Benchmarked (LOO across reps, 3 datasets × 3 seeds, run 2026-07-15): **93.6% overall**, 44 K params, 58 KB INT8 (~3.2× smaller than FP32 RingStrided), 152.5 K MACs — within noise of `EmagerCNNRingStridedQAT` (93.9%) and PTQ needs no fine-tuning phase. Documented in [`models.md`](emagerlib/models/models.md).
- **Microcontroller export tool** `examples/deployment/export_model.py`: converts a trained EMaGer CNN to MCU-ready formats through the pipeline PyTorch → ONNX → TFLite (float32) → TFLite (full INT8) → C array, numerically verifying each stage against the PyTorch reference (top-1 agreement + max logit diff). INT8 is a full-integer quantization (int8 weights + int8 I/O) calibrated on a representative set of real MAV windows via TensorFlow's `TFLiteConverter`, and the tool prints/records the input/output scale & zero-point needed to feed the MCU. Covers all three deployment stacks (STM32 X-CUBE-AI, TFLite Micro / LiteRT, ONNX Runtime). Works on the architecture variants (validated on `EmagerCNNRingStrided`: 55 KiB INT8, ~99.6% agreement) and on the few-shot prototypical models (bakes in generic "factory" prototypes so the distance-based forward is self-contained). New optional-dependency group `deploy` (`pip install -e ".[deploy]"`: onnx, onnxruntime, tensorflow, onnx2tf, tf_keras). See [`examples/deployment/README.md`](examples/deployment/README.md).
- Few-shot prototypical models in `emagerlib/models/new_emager_cnn.py` (experimental): `EmagerCNNProtoEpisodic` (episodic Prototypical Network training) and `EmagerCNNProtoCE` (cross-entropy pretrained embedding), sharing a `_EmagerProtoBase`. Classification is by distance to class prototypes *computed* from a support set rather than a learned classifier, modelling an offline-train → on-device few-shot-calibration deployment flow. `fit()` reports accuracy both before (generic prototypes) and after (5-shot calibration) on a held-out rep. Documented in [`models.md`](emagerlib/models/models.md#few-shot-prototypical-models).

### Changed
- Reordered `new_emager_cnn.py` so `EmagerCNNGAP` sits with the other architecture variants (after `EmagerCNNLight`), keeping all prototype-related code contiguous at the end of the file.
- `_EmagerProtoBase.fit()` split: the calibrate-and-score stage is now `_eval_fewshot()`, so the quantized proto variants can run it *after* `convert()` rather than duplicating it. Behaviour of the existing FP32 proto models is unchanged.
- The exact BN1d→Linear fold used before INT8 `convert()` is now the module-level `_fold_bn1d_into_linear()`, shared by the classifier path (`EmagerCNNQuantizedPTQ._fold_classifier_bn`) and the new embedding path (`_fold_embed_bn`). No behaviour change.

### Fixed
- **Silent cache corruption for quantized models** in `examples/training/eval_fewshot_loso.py`. The offline-model cache reconstructed a fresh FP32 instance and called `load_state_dict(..., strict=False)`. Quantized variants restructure themselves during `fit()` (fuse → fold → `convert()` to packed INT8 modules), so their `state_dict` shares almost no keys with a fresh instance — `strict=False` would have matched nothing and silently returned a **randomly initialised model**, reported as a real result. Quantized variants are now cached as whole pickled objects under a distinct `.full.pt` suffix, and the state_dict path raises on any missing/unexpected key instead of ignoring it.
- **Result-key collision in `eval_fewshot_loso.py`.** `_short()` truncated the variant name to 2 characters (`EmagerCNNProtoEpisodic` → `protoEp`), which was unique for the original two models but collapsed `RingStrided`, `RingStridedPTQ` and `RingStridedQAT` all to `protoRi` — their results would have silently merged into one another. Keys now keep the variant name whole.

## [1.2.1] - 2026-05-18

### Added
- `bleak` (BLE communication) and `PyYAML` (YAML config loading/saving) as direct dependencies.
- Smoke tests for libemg integration and function signatures (`tests/test_libemg_integration.py`).
- Stdin input field and **Send** button in the Command Launcher GUI for forwarding input to running subprocesses (e.g. answering `input()` prompts inside scripts).
- PATH scrub in the Command Launcher: strips `PyQt6\Qt6\bin` from the subprocess's inherited PATH to reduce Qt/torch DLL conflicts on Windows.

### Changed
- The libemg fork is now the default install source, pinned to the stable `emager-v3` branch (`pyproject.toml`, `README.md`, `requirements.txt`).
- README Quick Start now highlights the GUI Command Launcher as the recommended entry point for new users.
- README documents all four available libemg sources (fork `emager-v3`, fork `emager-v3-update`, PyPI, upstream `latest`) and how to switch between them via a manual two-step `uninstall` / `install`.
- Removed `matplotlib` and `pandas` from dependencies (no longer used at the library level).

### Fixed
- Import order in the v3 visualization scripts (`live_64_channel_v3.py`, `live_imu.py`, `live_heatmap.py`): libemg is now imported before any Qt binding, avoiding torch's `c10.dll` `WinError 1114` on Windows that started appearing after libemg PR #130 made `import libemg` transitively load torch.
- Removed the broken `[pypi]` / `[latest]` pip extras from `pyproject.toml` — pip's resolver cannot use them to override a URL-pinned base dependency. Switching libemg sources is now a documented manual two-step.

### Internal
- `.gitignore` updated to exclude `models/` and `.claude/`.

## [1.2.0]

First tagged release on the `main` branch. See git history for details.
