# Configuration Guide

Complete guide to configuring EMaGerLib for EMG projects.

## Table of Contents

- [Configuration Formats](#configuration-formats)
- [Core Parameters](#core-parameters)
- [Configuration Priority](#configuration-priority)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Configuration Formats

EMaGerLib supports three formats with automatic detection:

### Python (.py)

**Best for**: Complex configurations, computed values, code reuse

```python
# config.py
from pathlib import Path

BASE_PATH = Path("./Datasets/")
SESSION = "D1"
MEDIA_PATH = "./media-test/"
CLASSES = [2, 3, 30, 14, 18]
NUM_CLASSES = 5
WINDOW_SIZE = 200
SAMPLING = 1008
```

**Advantages**: Execute Python code, use imports, add computed values

### YAML (.yaml)

**Best for**: Version control, readability, team collaboration

```yaml
# config.yaml
BASE_PATH: "./Datasets/"
SESSION: "D1"
MEDIA_PATH: "./media-test/"
CLASSES: [2, 3, 30, 14, 18]
NUM_CLASSES: 5
WINDOW_SIZE: 200
SAMPLING: 1008
```

**Advantages**: Human-readable, easy to edit, good for version control

### JSON (.json)

**Best for**: Automation, saved configurations

```json
{
  "BASE_PATH": "./Datasets/",
  "SESSION": "D1",
  "MEDIA_PATH": "./media-test/",
  "CLASSES": [2, 3, 30, 14, 18],
  "NUM_CLASSES": 5,
  "WINDOW_SIZE": 200,
  "SAMPLING": 1008
}
```

**Advantages**: Machine-readable, used for automatic config saving

## Core Parameters

### Essential Paths

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `BASE_PATH` | Path/str | Root directory for datasets | `Path("./Datasets/")` |
| `SESSION` | str | Session identifier | `"D1"` or `"Participant_01"` |
| `MEDIA_PATH` | Path/str | Path to media files for screen-guided training | `"./media-test/"` |
| `MODEL_NAME` | str/None | Model filename or `None` to auto-find last model | `None` or `"model.ckpt"` |

### Gesture Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `CLASSES` | list[int] | Gesture class IDs | `[2, 3, 30, 14, 18]` |
| `NUM_CLASSES` | int | Number of gesture classes | `5` |
| `NUM_REPS` | int | Repetitions per gesture during collection | `5` |
| `REP_TIME` | int | Duration (seconds) for each gesture | `5` |
| `REST_TIME` | int | Rest duration (seconds) between gestures | `1` |

### Signal Processing

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `WINDOW_SIZE` | int | EMG window size (samples) | `200` |
| `WINDOW_INCREMENT` | int | Window sliding increment (samples) | `10` |
| `SAMPLING` | int | Sampling rate (Hz) - 1010 for v1, 2000 for v3 | `1008` |
| `FILTER` | bool | Enable signal filtering | `False` |
| `MAJORITY_VOTE` | int | Consecutive predictions for majority voting | `30` |

### Model Training

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `EPOCH` | int | Number of training epochs | `10` |

### Hardware

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `VIRTUAL` | bool | Use virtual/simulated device (not implemented) | `False` |
| `PORT` | str/None | Serial port for hardware or `None` for auto-detect | `None` |

### Real-time Control Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `USE_GUI` | bool | Enable graphical interface for real-time control | `False` |
| `POLL_SLEEP_DELAY` | float | Sleep time (seconds) to prevent busy-waiting | `0.001` |
| `PREDICTOR_DELAY` | float | Precision of prediction frequency (seconds) | `0.01` |
| `PREDICTOR_TIMEOUT_DELAY` | float | Timeout for waiting for predictions (seconds) | `0.50` |
| `SMOOTH_WINDOW` | int | Smoothing window size (1 to disable) | `1` |
| `SMOOTH_METHOD` | str | Smoothing method: `'mode'` or `'mean'` | `'mode'` |
| `HEARTBEAT_INTERVAL` | float | Re-send last gesture interval (seconds) | `0.1` |

### Logging

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `LOG_LEVEL` | str | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `"INFO"` |
| `LOG_TO_FILE` | bool | Enable file logging | `False` |
| `LOG_FILE_PATH` | str/None | Log file path or `None` for auto-generated | `None` |
| `LOG_FILE_NAME` | str/None | Custom log filename or `None` for auto-generated | `None` |

### Config Saving

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `SAVE_CONFIG_PATH` | str/None | Where to save config or `None` for default | `None` |
| `SAVE_CONFIG_NAME` | str/None | Custom config name or `None` for auto-generated | `None` |
| `SAVE_CONFIG_FORMAT` | str | Output format: `"json"` or `"yaml"` | `"json"` |

## Configuration Priority

**Command-line args > Config file > System defaults**

Example:
```bash
# Config file has LOG_LEVEL = "WARNING"
emager-train-cnn -c my_config.py --log-level DEBUG
# Result: Uses DEBUG (CLI overrides config)
```

## Usage Examples

### Minimal Configuration

```python
# minimal_config.py
from pathlib import Path

BASE_PATH = Path("./Data/")
SESSION = "S1"
MEDIA_PATH = "./media-test/"
CLASSES = [2, 3, 30]
NUM_CLASSES = 3
```

### Complete Configuration

```python
# full_config.py
from pathlib import Path

# ===== Paths ===== #
BASE_PATH = Path("./Datasets/")
SESSION = "Participant_01"
MEDIA_PATH = "./media-test/"
MODEL_NAME = None  # Auto-find last model

# ===== Gesture Configuration ===== #
CLASSES = [2, 3, 30, 14, 18]  # hand_close, hand_open, index, ok, thumbs_up
NUM_CLASSES = 5
NUM_REPS = 5
REP_TIME = 5
REST_TIME = 1

# ===== Signal Processing ===== #
WINDOW_SIZE = 200
WINDOW_INCREMENT = 10
SAMPLING = 1008
FILTER = False
MAJORITY_VOTE = 30

# ===== Model Training ===== #
EPOCH = 10

# ===== Hardware ===== #
VIRTUAL = False
PORT = None  # Auto-detect

# ===== Real-time Control ===== #
USE_GUI = False
POLL_SLEEP_DELAY = 0.001
PREDICTOR_DELAY = 0.01
PREDICTOR_TIMEOUT_DELAY = 0.50
SMOOTH_WINDOW = 1
SMOOTH_METHOD = 'mode'
HEARTBEAT_INTERVAL = 0.1

# ===== Logging ===== #
LOG_LEVEL = "INFO"
LOG_TO_FILE = False
LOG_FILE_PATH = None
LOG_FILE_NAME = None

# ===== Config Saving ===== #
SAVE_CONFIG_PATH = None
SAVE_CONFIG_NAME = None
SAVE_CONFIG_FORMAT = "json"
```

### Using Configuration

```bash
# Load Python config
emager-train-cnn -c my_config.py

# Load YAML config
emager-screen-training -c sessions/session_D2.yaml

# Load JSON config (typically auto-saved configs)
emager-realtime-predict -c saved_configs/experiment_01.json
```

## Best Practices

- **Use `Path()` for cross-platform paths**: `Path("./data/")` works on Windows/Linux/Mac
- **Document gesture IDs**: Add comments explaining what each class number means
- **Save used configs**: Use `--save-config-name` for reproducibility
- **Start from examples**: Copy [base_config_example.py](../config_examples/base_config_example.py)

### Important: Update .gitignore for Custom Paths

> **⚠️ Warning**: If you change any path-related parameters in your config (especially `BASE_PATH`, `MEDIA_PATH`, log directories, or saved config locations), you should update your `.gitignore` file accordingly to prevent committing large datasets, media files, or logs to version control.

The default `.gitignore` file has a clearly marked configurable section:

```gitignore
### ===============CONFIGURABLE============== ###
### CHANGE THESE IF YOU MODIFY IN CONFIG FILE ###
logs/ 
media-test/ 
Datasets/    
saved_configs/  
### ========================================= ###
```

**Examples of when to update .gitignore**:
- Changed `BASE_PATH` from `"./Datasets/"` to `"./MyData/"` → Add `MyData/` to `.gitignore`
- Changed `MEDIA_PATH` from `"./media-test/"` to `"./gestures/"` → Add `gestures/` to `.gitignore`
- Using custom log directory → Add your log path to `.gitignore`
- Changed saved config location → Add that path to `.gitignore`

This prevents accidentally committing large data files to your repository.

### Validation

The system automatically validates:
- Required parameters are present
- Types are correct (int, str, Path, etc.)
- Values are in valid ranges
- Paths exist (when needed)

Validation errors will show helpful messages explaining what's wrong.

## Common Gesture Class IDs

Reference for common gestures (from LibEMG convention):

| ID | Gesture |
|----|---------|
| 2 | Hand close (fist) |
| 3 | Hand open |
| 14 | OK sign |
| 18 | Thumbs up |
| 30 | Index finger extension (point) |

See your `gestures.json` file for the complete list.

## See Also

- [base_config_example.py](../config_examples/base_config_example.py) - Python template
- [base_config_example.yaml](../config_examples/base_config_example.yaml) - YAML template
- [CLI Guide](CLI.md) - Command-line arguments
