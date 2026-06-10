# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Few-shot prototypical models in `emagerlib/models/new_emager_cnn.py` (experimental): `EmagerCNNProtoEpisodic` (episodic Prototypical Network training) and `EmagerCNNProtoCE` (cross-entropy pretrained embedding), sharing a `_EmagerProtoBase`. Classification is by distance to class prototypes *computed* from a support set rather than a learned classifier, modelling an offline-train → on-device few-shot-calibration deployment flow. `fit()` reports accuracy both before (generic prototypes) and after (5-shot calibration) on a held-out rep. Documented in [`models.md`](emagerlib/models/models.md#few-shot-prototypical-models).

### Changed
- Reordered `new_emager_cnn.py` so `EmagerCNNGAP` sits with the other architecture variants (after `EmagerCNNLight`), keeping all prototype-related code contiguous at the end of the file.

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
