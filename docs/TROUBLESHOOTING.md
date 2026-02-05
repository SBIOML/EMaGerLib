# Troubleshooting Guide

Common issues and solutions for installation, hardware, data collection, and development.

## Installation

### Quick Install
```bash
git clone https://github.com/SBIOML/EMaGerLib.git
cd EMaGerLib
pip install -e .
```

**Requirements**: Python >= 3.8, pip, Git

### Commands Not Found

**Problem**: `emager-train-cnn: command not found`

**Solutions**:
```bash
# Reinstall in editable mode
pip install -e .

# Or run scripts directly
python examples/training/train_cnn.py
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'emagerlib'`

**Solutions**:
```bash
pip uninstall emagerlib
pip install -e .
```

### Dependency Conflicts

**Solutions**:
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Or install libemg from GitHub
pip install git+https://github.com/libemg/libemg.git
pip install -e .
```

### Platform-Specific

**Linux USB permissions**:
```bash
sudo usermod -a -G dialout $USER
# Logout and login
```

**macOS**: Install Xcode Command Line Tools if needed:
```bash
xcode-select --install
```

## Hardware Issues

### EMaGer Not Detected

**Checklist**:
- USB cable connected
- Device powered on
- Check Device Manager (Windows) or `lsusb` (Linux)
- Try different USB port

**Serial ports**:
- Windows: `COM3`, `COM4`, etc.
- Linux: `/dev/ttyUSB0`, `/dev/ttyACM0`
- macOS: `/dev/cu.usbserial-*`

## Data Collection Issues

### Screen Not Displaying

**Problem**: GUI doesn't appear or crashes

**Solutions**:
```bash
# Install GUI dependencies
pip install PyQt5

# Try different backend (Linux)
export MPLBACKEND=TkAgg
emager-screen-training
```

### Gesture Images Not Loading

**Problem**: Media files not found

**Solutions**:
```python
# Verify MEDIA_PATH in config
MEDIA_PATH = "./media-test/"

# Use absolute path if needed
from pathlib import Path
MEDIA_PATH = Path("/absolute/path/to/media/")
```

### Data Not Saving

**Problem**: Collected data not written to disk

**Solutions**:
```python
# Check BASE_PATH exists and is writable
BASE_PATH = Path("./Datasets/")
BASE_PATH.mkdir(parents=True, exist_ok=True)
```

## Training Issues

### Out of Memory

**Solution**: Reduce batch size in config:
```python
BATCH_SIZE = 16  # or 8
```

### Data Not Found

**Solution**: Check paths in config:
```python
# Verify these exist
print(BASE_PATH / SESSION)
```

## Real-time Issues

### Real-time Lag

**Solution**: Adjust timing parameters:
```python
# Prediction timing
PREDICTOR_DELAY = 0.01          # Prediction frequency
PREDICTOR_TIMEOUT_DELAY = 0.50  # Timeout for predictions
POLL_SLEEP_DELAY = 0.001        # Prevent busy-waiting

# Smoothing (reduce jitter vs responsiveness)
SMOOTH_WINDOW = 1       # Set to 1 to disable smoothing
SMOOTH_METHOD = 'mode'  # 'mode' for gestures

# Hand communication
HEARTBEAT_INTERVAL = 0.1  # Re-send last gesture (10 Hz)
```

## Development Setup

### Basic Setup

```bash
# Clone repository
git clone https://github.com/SBIOML/EMaGerLib.git
cd EMaGerLib

# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # Windows: dev_env\Scripts\activate

# Install in editable mode
pip install -e .

# Verify
emager-run-tests
```

### Adding New Features

**New hand controller** - Add to `emagerlib/control/`:
```python
from emagerlib.control.abstract_hand_control import AbstractHandControl

class MyHandControl(AbstractHandControl):
    def send_command(self, gesture_id):
        # Implement gesture -> hand command
        pass
```

**Custom model** - Add to `emagerlib/models/`:
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Define architecture
```

**Utility function** - Add to `emagerlib/utils/utils.py`

### Testing

```bash
# Run all tests
emager-run-tests

# Run specific test
python tests/test_config.py
```

### Git Flow Workflow

This project uses Git Flow for branch management:

- **`main`**: Production-ready code
- **`develop`**: Integration branch
- **`feature/*`**: New features (from `develop`)
- **`hotfix/*`**: Urgent fixes (from `main`)

**Basic workflow**:
```bash
# Start feature
git checkout develop
git checkout -b feature/my-feature

# Make changes and commit
git commit -m "feat: add new feature"

# Merge back to develop
git checkout develop
git merge feature/my-feature
git push origin develop
```

**Learn more**: [Git Flow Guide](https://danielkummer.github.io/git-flow-cheatsheet/)

## Getting Help

**Before asking**:
1. Check logs in `./logs/`
2. Run `emager-run-tests`
3. Try with default config

**Report issues**: [GitHub Issues](https://github.com/SBIOML/EMaGerLib/issues)

**Include**:
- Python version (`python --version`)
- OS and version
- Full error message
- Installation method used

**Contact**: etmic6@ulaval.ca