# Troubleshooting Guide

Common issues and quick solutions.

## Installation Issues

### Commands Not Found

**Problem**: `emager-train-cnn: command not found`

**Solution**: Reinstall in editable mode:
```bash
pip install -e .
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'emager_tools'`

**Solution**:
```bash
pip uninstall emagerpy
pip install -e .
```

### Dependency Conflicts

**Solution**: Create fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install -e .
```

## Hardware Issues

### EMaGer Not Detected

**Checklist**:
- USB cable connected
- Device powered on
- Check Device Manager (Windows) or `lsusb` (Linux)

**Linux permissions**:
```bash
sudo usermod -a -G dialout $USER
# Logout and login
```

## Data Collection Issues

### Screen Not Displaying

**Problem**: GUI doesn't appear or crashes

**Solutions**:
```bash
# Install GUI dependencies
pip install PyQt5

# Check display (Linux)
echo $DISPLAY

# Try different backend
export MPLBACKEND=TkAgg
emager-screen-training
```

### Gesture Images Not Loading

**Problem**: Media files not found

**Solutions**:
```python
# Verify MEDIA_PATH in config
MEDIA_PATH = "./media-test/"

# Check files exist
ls -l media-test/

# Use absolute path
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

# Check permissions
ls -ld Datasets/

# Use absolute path
BASE_PATH = Path("/home/user/emg_data/")
```

### Out of Memory During Training

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

### Real-time Lag

**Solution**: Adjust timing parameters:
```python

# Prediction timing (for fine-tuning responsiveness)
PREDICTOR_DELAY = 0.01          # Precision of prediction frequency
PREDICTOR_TIMEOUT_DELAY = 0.50  # Timeout for waiting for predictions
POLL_SLEEP_DELAY = 0.001        # Prevent busy-waiting

# Smoothing (reduce jitter vs responsiveness trade-off)
SMOOTH_WINDOW = 1       # Set to 1 to disable smoothing
SMOOTH_METHOD = 'mode'  # 'mode' for gestures, 'mean' for numeric

# Heartbeat for hand communication
HEARTBEAT_INTERVAL = 0.1  # Re-send last gesture (10 Hz)
```

## Getting Help

**Before asking**:
1. Check logs in `./logs/`
2. Run `emager-run-tests`
3. Try with default config

**Report issues**: [GitHub Issues](https://github.com/SBIOML/emagerpy/issues)

Include:
- Python version (`python --version`)
- OS and version
- Full error message

**Contact**: etmic6@ulaval.ca