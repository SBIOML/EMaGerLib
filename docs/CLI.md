# Command-Line Interface Guide

Reference for command-line arguments in emagerpy commands.

## Table of Contents

- [Common Arguments](#common-arguments)
- [Usage Examples](#usage-examples)
- [Available Commands](#available-commands)

## Common Arguments

All emagerpy commands support these arguments:

### Configuration

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `-c, --config` | PATH | Load config file (.py, .yaml, .json) | `-c my_config.py` |
| `-h, --help` | Flag | Show help message | `--help` |

### Logging

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--log-level` | LEVEL | Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--log-to-file` | Flag | Enable file logging | False |
| `--no-log-to-file` | Flag | Disable file logging | - |
| `--log-file-path` | PATH | Exact log file path | Auto-generated |
| `--log-file-name` | NAME | Log filename (timestamp added) | Script name |

### Config Saving

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--save-config-name` | NAME | Save config with name (timestamp added) | - |
| `--save-config-path` | PATH | Where to save config | `./saved_configs/` |
| `--save-config-format` | FORMAT | Output format: json or yaml | json |

### Priority System

**Command-line args > Config file > System defaults**

Arguments override config file values, which override defaults.

## Usage Examples

### Basic Usage

```bash
# Use default config
emager-train-cnn

# Use custom config
emager-train-cnn -c my_config.py
```

### With Logging

```bash
# Debug mode with file logging
emager-realtime-predict --log-level DEBUG --log-to-file

# Custom log file name
emager-train-cnn --log-file-name training_session --log-to-file

# Quiet mode
emager-realtime-control --log-level WARNING
```

### Save Configuration

```bash
# Save used configuration for reproducibility
emager-train-cnn --save-config-name experiment_01

# Save as YAML
emager-screen-training --save-config-name data_session --save-config-format yaml
```

### Complete Example

```bash
# Full reproducible run with logging and config saving
emager-train-cnn \
  -c my_config.py \
  --log-level INFO \
  --log-to-file \
  --log-file-name training \
  --save-config-name trained_model_v1
```

## Available Commands

### Data Collection

**emager-screen-training** - Screen-guided gesture data collection

```bash
emager-screen-training -c my_config.py --log-to-file
```

### Training

**emager-train-cnn** - Train CNN model on EMG data

```bash
emager-train-cnn -c training_config.py --save-config-name model_v1
```

### Real-time

**emager-realtime-predict** - Test gesture recognition without hand control

```bash
emager-realtime-predict -c config.py --log-level DEBUG
```

**emager-realtime-control** - Control prosthetic hand in real-time

```bash
emager-realtime-control -c control_config.py --log-to-file
```

### Visualization

**emager-live-64ch** - Live 64-channel EMG visualization

```bash
emager-live-64ch
```

**emager-visualize-libemg** - Visualize with libemg tools

```bash
emager-visualize-libemg -c config.py
```

### Testing

**emager-test-hand** - Test hand control interface
**emager-test-psyonic** - Test Psyonic hand connection
**emager-test-wave** - Test wave gesture control
**emager-run-tests** - Run complete test suite

## Common Workflows

### Development

```bash
# 1. Collect data
emager-screen-training -c dev_config.py --log-to-file

# 2. Train model
emager-train-cnn -c dev_config.py --save-config-name model_v1

# 3. Test
emager-realtime-predict -c dev_config.py
```

### Experiment Tracking

```bash
# Run with complete logging
emager-train-cnn \
  -c exp_001/config.yaml \
  --log-to-file \
  --log-file-path exp_001/training.log \
  --save-config-path exp_001/config_used.json
```

## Troubleshooting

**Command not found**: Reinstall with `pip install -e .` or use `python examples/training/train_cnn.py` directly.

**Config not loading**: Check file exists and use absolute path.

**Permission errors**: Check write permissions on log/config directories.

## See Also

- [Configuration Guide](CONFIGURATION.md)
- [Quick Start](../README.md#quick-start)
