# Installation Guide

Installation instructions for emagerpy with troubleshooting tips.

## Table of Contents

- [Quick Installation](#quick-installation)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Quick Installation

```bash
git clone https://github.com/SBIOML/emagerpy.git
cd emagerpy
pip install -e .
```

This installs emagerpy in editable mode with all dependencies and console commands.

## Prerequisites

### Required

- **Python** >= 3.8
- **pip** (latest version)
- **Git**

### Optional

- **Virtual environment** (recommended)
- **CUDA** (for GPU acceleration)
- **EMaGer hardware** (for real data acquisition)

### Verify

```bash
python --version  # Should be >= 3.8
pip --version
git --version
```

## Installation Methods

### Standard Installation (Recommended)

**Step 1**: Create virtual environment (recommended)

```bash
# Create
python -m venv emagerpy_env

# Activate (Windows)
emagerpy_env\Scripts\activate

# Activate (Linux/Mac)
source emagerpy_env/bin/activate
```

**Step 2**: Clone repository

```bash
git clone https://github.com/SBIOML/emagerpy.git
cd emagerpy
```

**Step 3**: Install

```bash
# Editable mode (recommended for development)
pip install -e .

# Or standard install
pip install .
```

### Alternative: Specific libemg Versions

Install with specific libemg versions for compatibility:

```bash
# Latest development version from GitHub
pip install -e ".[latest]" --force-reinstall --no-deps libemg

# Fork version with EMaGer v3 support
pip install -e ".[fork]" --force-reinstall --no-deps libemg

# Local development version
pip install -e ".[local]" --force-reinstall --no-deps libemg
```

**Note**: If `--force-reinstall --no-deps libemg` doesn't work, try uninstalling first:
```bash
pip uninstall libemg
```

### Alternative: Direct from GitHub

```bash
pip install git+https://github.com/SBIOML/emagerpy.git
```

### Alternative: Requirements Only

```bash
pip install -r requirements.txt
```

Note: This doesn't install console commands. Run scripts directly:
```bash
python examples/training/train_cnn.py
```

## Verification

### Test Console Commands

```bash
# Run test suite
emager-run-tests

# Check help
emager-train-cnn --help
```

### Test Import

```bash
python -c "from emager_tools import config; print('Success!')"
```

### Test EMaGer Connection (if hardware available)

```bash
emager-live-64ch
```

## Troubleshooting

### Commands Not Found

**Problem**: `emager-train-cnn: command not found`

**Solutions**:
```bash
# 1. Reinstall in editable mode
pip install -e .

# 2. Check virtual environment is activated
which python  # Should point to venv

# 3. Use Python directly
python examples/training/train_cnn.py
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'emager_tools'`

**Solutions**:
```bash
# 1. Reinstall
pip uninstall emagerpy
pip install -e .

# 2. Check installation
pip show emagerpy

# 3. Verify Python environment
python -c "import sys; print(sys.path)"
```

### Dependency Conflicts

**Problem**: Dependency version conflicts

**Solutions**:
```bash
# 1. Fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install -e .

# 2. Update pip
pip install --upgrade pip

# 3. Install specific libemg version
pip install git+https://github.com/libemg/libemg.git
pip install -e .
```

### libemg Installation Issues

**Problem**: Can't install libemg

**Solutions**:
```bash
# Install from GitHub directly
pip install git+https://github.com/libemg/libemg.git

# Or clone and install locally
git clone https://github.com/libemg/libemg.git
cd libemg
pip install -e .
cd ../emagerpy
pip install -e .
```

### Permission Errors

**Problem**: Permission denied during installation

**Solutions**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .

# Or use --user flag (not recommended with venv)
pip install --user -e .
```

### EMaGer Not Detected

**Problem**: Device not found

**Solutions**:
- Check USB connection
- Verify device in system:
  - Windows: Device Manager
  - Linux: `lsusb`
- Try different USB port
- Check driver installation
- See [Troubleshooting Guide](TROUBLESHOOTING.md#hardware-issues)

## Platform-Specific Notes

### Windows

- Use Command Prompt or PowerShell
- Activate venv: `venv\Scripts\activate`
- EMaGer serial ports usually `COM3`, `COM4`, etc.

### Linux

- May need sudo for USB permissions:
  ```bash
  sudo usermod -a -G dialout $USER
  # Logout and login for changes to take effect
  ```
- Serial ports: `/dev/ttyUSB0`, `/dev/ttyACM0`, etc.

### macOS

- Install Xcode Command Line Tools if needed:
  ```bash
  xcode-select --install
  ```
- Serial ports: `/dev/cu.usbserial-*`

## Advanced Options

### Install with Specific Dependencies

```bash
# With PyTorch CPU
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

# With PyTorch GPU (CUDA)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

### Development Install

```bash
# Clone with all branches
git clone --recurse-submodules https://github.com/SBIOML/emagerpy.git

# Install in editable mode
pip install -e .

# Install development tools (if needed)
pip install pytest black flake8
```

### Install from Fork

```bash
# Your fork
git clone https://github.com/YOUR_USERNAME/emagerpy.git
cd emagerpy
pip install -e .
```

## Uninstallation

```bash
# Uninstall package
pip uninstall emagerpy

# Remove virtual environment
deactivate
rm -rf emagerpy_env  # or rmdir /s emagerpy_env on Windows
```

## Getting Help

If installation issues persist:

1. Check [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search [GitHub Issues](https://github.com/SBIOML/emagerpy/issues)
3. Open new issue with:
   - Python version (`python --version`)
   - OS and version
   - Complete error message
   - Installation method used

## See Also

- [Quick Start](../README.md#quick-start)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Development Guide](DEVELOPMENT.md)
