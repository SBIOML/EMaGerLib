# emagerpy

> **Note**: This project replaces the previous `emager-py` repository with improved architecture, enhanced features, and better maintainability.


A comprehensive toolbox for working with the EMaGer v1 and v3 EMG acquisition devices, featuring real-time gesture recognition and prosthetic hand control capabilities.

## Overview

**emagerpy** is a Python-based framework designed for electromyographic (EMG) signal processing, machine learning-based gesture recognition, and prosthetic hand control. Built on top of [libemg](https://github.com/libemg/libemg), it provides a complete pipeline from data collection to real-time control of prosthetic devices.

### Key Features

- **Real-time Gesture Recognition**: Advanced CNN-based models with quantization support for efficient inference
- **Prosthetic Hand Control**: Native support for Psyonic and other prosthetic hands via serial communication
- **Data Collection & Training**: Screen-guided training sessions with configurable gesture sets
- **Visualization Tools**: Real-time 64-channel EMG visualization and monitoring
- **Flexible Configuration**: YAML and Python-based configuration system
- **EMaGer Hardware Support**: Compatible with EMaGer v1 and v3 devices

## Project Structure

```
emagerpy/
├── emager_tools/         # Core library modules
│   ├── config/             # Configuration management
│   ├── control/            # Prosthetic hand control interfaces
│   ├── models/             # Neural network models (EmagerCNN)
│   ├── utils/              # Utility functions and helpers
│   └── visualization/      # Real-time GUI and plotting tools
├── examples/           # Executable scripts
│   ├── data_collection/    # Data collection scripts
│   ├── hand_control/       # Hand control tests scripts
│   ├── realtime/           # Real-time prediction and control
│   ├── training/           # Model training scripts
│   └── visualisation/      # Visualization tools
├── config_examples/      # Example configuration files
├── tests/                # Unit tests
```

## Installation

### Prerequisites

- Python >= 3.8
- Windows/Linux/MacOS
- EMaGer v1/v3 hardware (for real data acquisition)

### Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/SBIOML/emagerpy.git
cd emagerpy

# Install in development mode
pip install -e .
```

### Alternative Installation Methods

Install with specific libemg versions:

```bash
# Latest development version from GitHub
pip install -e ".[latest]" --force-reinstall --no-deps libemg

# Fork version with EMaGer v3 support
pip install -e ".[fork]" --force-reinstall --no-deps libemg

# Local development version
pip install -e ".[local]" --force-reinstall --no-deps libemg
```

### Manual Installation

```bash
pip install -r requirements.txt
```


## Quick Start

> **Note**: After installing with `pip install -e .`, you can use short console commands (e.g., `emager-screen-training`) or run scripts directly with Python (e.g., `python examples/...`). See the [Console Commands](#available-console-commands) table below for the complete list.

### 1. Configure Your Setup

Copy and modify the example configuration:

```python
# config_examples/base_config_example.py
from pathlib import Path

BASE_PATH = Path("./Datasets/")
SESSION = "D1"
MEDIA_PATH = "./media-test/"

# Gesture classes: hand_close, hand_open, index extension, ok, thumbs up
CLASSES = [2, 3, 30, 14, 18]
NUM_CLASSES = 5
WINDOW_SIZE = 200
SAMPLING = 1010
```
### 2. Visualize EMG Signals

Monitor real-time 64-channel EMG data:

```bash
# Using console command
emager-live-64ch

# Or using direct Python execution
python examples/visualisation/live_64_channel.py
```

### 3. Data Collection

Run a screen-guided training session:

```bash
# Using console command
emager-screen-training

# Or using direct Python execution
python examples/data_collection/screen_guided_trainning.py
```

### 4. Train a Model

Train a CNN on collected EMG data:

```bash
# Using console command
emager-train-cnn

# Or using direct Python execution
python examples/training/train_cnn.py
```

### 5. Real-time Prediction

Test gesture recognition in real-time:

```bash
# Using console command
emager-realtime-predict

# Or using direct Python execution
python examples/realtime/realtime_prediction.py
```

### 6. Control a Prosthetic Hand

Run real-time control with a connected prosthetic:

```bash
# Using console command
emager-realtime-control

# Or using direct Python execution
python examples/realtime/realtime_control.py
```

## Configuration

> **Note**: For simple configuration, refer to [base_config_example.py](config_examples/base_config_example.py). You can create custom config files and add extra variables as needed. For advanced configuration or core modifications, see the detailed documentation below.

### Understanding the Configuration System

The configuration system uses a structured approach with automatic validation and type checking defined in [emager_tools/config/core.py](emager_tools/config/core.py).

**Important**: Configuration variable names are defined in `core.py` and **cannot be renamed** in your custom config files without modifying `core.py` (not recommended). The system validates all parameters against the `CoreConfig` dataclass.

#### Variable Types

1. **Mandatory Variables**: Must be defined in your config file. Missing mandatory variables will cause errors.
2. **Optional Variables**: Have default values in `core.py` or can be computed (e.g., `CONFIG_FILE_NAME`).
3. **Extra Variables**: Any additional variables you define are stored in the `EXTRA` dictionary and accessible via dot notation.

### Mandatory Configuration Parameters

All configuration files must define the following parameters:

#### Paths
| Parameter | Type | Description |
|-----------|------|-------------|
| `BASE_PATH` | `Path` | Root directory for datasets and models |
| `SESSION` | `str` | Session identifier (creates subdirectory under BASE_PATH) |
| `MEDIA_PATH` | `str` | Path to gesture images/media for libemg (autogenerated) |
| `MODEL_NAME` | `str` or `None` | Model filename or `None` to auto-detect last model |

#### Gesture/Classes Configuration
| Parameter | Type | Description |
|-----------|------|-------------|
| `CLASSES` | `List[int]` | List of gesture IDs to use (e.g., [2, 3, 30, 14, 18]) |
| `NUM_CLASSES` | `int` | Total number of gesture classes |
| `NUM_REPS` | `int` | Number of repetitions per gesture during data collection |
| `REP_TIME` | `int` | Duration (seconds) for each gestures |
| `REST_TIME` | `int` | Rest duration (seconds) between each gestures |

#### Data Acquisition and Model Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `MAJORITY_VOTE` | `int` | Number of consecutive predictions for majority voting |
| `WINDOW_SIZE` | `int` | EMG window size in samples |
| `WINDOW_INCREMENT` | `int` | Window sliding increment in samples |
| `EPOCH` | `int` | Number of training epochs |
| `SAMPLING` | `int` | Sampling rate in Hz (e.g., 1010 for EMaGerV1 and 2000 for V3) |
| `FILTER` | `bool` | Enable/disable signal filtering |
| `VIRTUAL` | `bool` | Use virtual/simulated EMG device (Not implemented)|
| `PORT` | `str` or `None` | Serial port for hardware connection (None for auto) |

#### Controller and Predictor Settings
| Parameter | Type | Description |
|-----------|------|-------------|
| `USE_GUI` | `bool` | Enable graphical user interface for realtime control |
| `POLL_SLEEP_DELAY` | `float` | Sleep time (seconds) to prevent busy-waiting when polling |
| `PREDICTOR_DELAY` | `float` | Prediction update frequency (seconds) |
| `PREDICTOR_TIMEOUT_DELAY` | `float` | Timeout (seconds) when waiting for predictions |
| `SMOOTH_WINDOW` | `int` | Smoothing window size (set to 1 to disable) |
| `SMOOTH_METHOD` | `str` | Smoothing method: `'mode'` (categorical) or `'mean'` (numeric) |
| `HEARTBEAT_INTERVAL` | `float` | Interval (seconds) for re-sending last gesture to maintain connection |

### Computed Properties

The following are automatically computed and do not need to be defined, but can be changed:

- `SESSION_PATH`: `BASE_PATH / SESSION`
- `SAVE_PATH`: Same as `SESSION_PATH`
- `DATAFOLDER`: Same as `SESSION_PATH`
- `DATASETS_PATH`: Same as `SESSION_PATH`
- `MODEL_PATH`: Automatically finds the last model if `MODEL_NAME` is `None`


###
>**Note** : Example configuration can be found here [config_examples/base_config_example.py](emager_tools/config/core.py)

## Testing

Run the complete test suite:

```bash
# Using console command (recommended)
emager-run-tests

# Or using direct Python execution
python tests/run_all_tests.py
```

The test suite includes:
- **Configuration System**: Loading/saving configs (Python, JSON, YAML)
- **Utility Functions**: Majority voting, transform decimation, packet printing
- **Model Finding**: Discovering and sorting model files
- **Constants**: Gesture and finger definitions
- **Gesture JSON**: Image and gesture dictionary utilities

Test individual components:

```bash
# Test hand control (console command)
emager-test-hand

# Test Psyonic hand (console command)
emager-test-psyonic

# Test hand wave (console command)
emager-test-wave

# Visualize EMG signals (console command)
emager-live-64ch

# Or use direct Python execution
python examples/hand_control/test_hand_control.py
python examples/hand_control/test_psyonic_hand.py
python examples/visualisation/live_64_channel.py
```

## Available Console Commands

After installing with `pip install -e .`, the following commands are available:

| Command | Description | Python Alternative |
|---------|-------------|-------------------|
| `emager-sgt` | Screen-guided data collection | `python examples/data_collection/screen_guided_trainning.py` |
| `emager-test-hand` | Test hand control interface | `python examples/hand_control/test_hand_control.py` |
| `emager-test-wave` | Test hand wave gestures | `python examples/hand_control/test_hand_wave.py` |
| `emager-test-psyonic` | Test Psyonic hand | `python examples/hand_control/test_psyonic_hand.py` |
| `emager-realtime-control` | Real-time prosthetic control | `python examples/realtime/realtime_control.py` |
| `emager-realtime-predict` | Real-time gesture prediction | `python examples/realtime/realtime_prediction.py` |
| `emager-train-cnn` | Train CNN model | `python examples/training/train_cnn.py` |
| `emager-visualize-libemg` | Visualize with libemg | `python examples/visualisation/libemg_visualize.py` |
| `emager-live-64ch` | Live 64-channel visualization | `python examples/visualisation/live_64_channel.py` |
| `emager-run-tests` | Run complete test suite | `python tests/run_all_tests.py` |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [libemg](https://github.com/libemg/libemg)
- Developed by the Smart Biomedical Microsystems Laboratory (SBIOML)

## Contact

**Author**: Étienne Michaud  
**Email**: etmic6@ulaval.ca  
**Organization**: Smart Biomedical Microsystems Laboratory

## Citation

If you use this software in your research, please cite:

```bibtex
@software{emagerpy2025,
  title = {emagerpy: EMG Signal Processing and Prosthetic Control Toolbox},
  author = {Michaud, Étienne},
  year = {2025},
  organization = {Smart Biomedical Microsystems Laboratory},
  license = {MIT}
}
```
