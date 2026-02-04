# Development Guide

Guide for contributing to and extending EMaGerLib.

## Table of Contents

- [Development Workflows](#development-workflows)
- [Setup Development Environment](#setup-development-environment)
- [Project Architecture](#project-architecture)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Contributing](#contributing)

## Development Workflows

### 1. Using EMaGerLib as-is

**For**: Standard EMG projects with custom configurations only

**Setup**:
```bash
pip install -e .
cp config_examples/base_config_example.py my_config.py
# Edit my_config.py
emager-screen-training -c my_config.py
```

**No code modification needed** - just configure and use.

### 2. Extending EMaGerLib

**For**: Adding new features (controllers, models, visualizations)

**Setup**:
```bash
git clone https://github.com/YOUR_USERNAME/emagerpy.git
cd emagerpy
pip install -e .
```

**Add modules to**:
- `emagerlib/control/` - New prosthetic interfaces
- `emagerlib/models/` - Custom models
- `emagerlib/visualization/` - Visualization tools
- `emagerlib/utils/` - Utility functions

### 3. Modifying core functionality

**For**: Changing libemg integration or core behavior

**Setup**:
```bash
# Fork and clone both repos
git clone https://github.com/YOUR_USERNAME/emagerpy.git
git clone https://github.com/libemg/libemg.git

# Install libemg first
cd libemg
pip install -e .

# Then EMaGerLib
cd ../emagerpy
pip install -e .
```

## Setup Development Environment

### Basic Setup

```bash
# Clone repository
git clone https://github.com/SBIOML/emagerpy.git
cd emagerpy

# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install in editable mode
pip install -e .

# Verify
emager-run-tests
```

### With Development Tools

```bash
# Install testing tools
pip install pytest pytest-cov

# Install code quality tools
pip install black flake8 pylint

# Install documentation tools
pip install sphinx sphinx-rtd-theme
```

## Project Architecture

### Directory Structure

```
EMaGerLib/
├── emagerlib/                 # Core library
│   ├── config/                # Configuration system
│   │   ├── core_config.py     # Config dataclass
│   │   ├── load_config.py     # Load Python/YAML/JSON
│   │   └── save_config.py     # Save configurations
│   ├── control/               # Hand control interfaces
│   │   ├── abstract_hand_control.py  # Base class
│   │   ├── psyonic_control.py        # Psyonic hand
│   │   └── interface_control.py      # Generic interface
│   ├── models/                # Neural network models
│   │   └── models.py          # EmagerCNN
│   ├── utils/                 # Utilities
│   │   ├── arg_parser.py      # CLI argument parsing
│   │   ├── find_models.py     # Model discovery
│   │   ├── gestures_json.py   # Gesture definitions
│   │   └── utils.py           # Helper functions
│   └── visualization/         # Visualization
│       └── realtime_gui.py    # Real-time GUI
├── examples/                  # Executable scripts
│   ├── data_collection/
│   ├── hand_control/
│   ├── realtime/
│   ├── training/
│   └── visualisation/
├── tests/                     # Unit tests
├── config_examples/           # Example configs
└── docs/                      # Documentation
```

### Key Components

**Configuration System** (`emagerlib/config/`):
- Loads Python/YAML/JSON configs
- Validates and type-checks
- Saves used configurations

**Control Interfaces** (`emagerlib/control/`):
- Abstract base class for hand controllers
- Implementations for specific hands
- Serial/BLE communication

**Models** (`emagerlib/models/`):
- CNN architecture for gesture recognition
- PyTorch Lightning integration
- Model loading and inference

**Utilities** (`emagerlib/utils/`):
- Argument parsing
- Model discovery
- Gesture management
- Helper functions

## Adding New Features

### Adding a New Hand Controller

**Step 1**: Create controller file

```python
# emagerlib/control/my_hand.py
from emagerlib.control.abstract_hand_control import AbstractHandControl

class MyHandControl(AbstractHandControl):
    def __init__(self, port="COM3"):
        super().__init__()
        self.port = port
        # Initialize your hand connection
        
    def send_command(self, gesture_id):
        # Implement gesture -> hand command mapping
        pass
        
    def close(self):
        # Clean up connection
        pass
```

**Step 2**: Use in your script

```python
from emagerlib.control.my_hand import MyHandControl

hand = MyHandControl(port="COM5")
hand.send_command(2)  # Hand close
hand.close()
```

### Adding a Custom Model

**Step 1**: Create model file

```python
# emagerlib/models/my_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Define your architecture
        
    def forward(self, x):
        # Implement forward pass
        pass
```

**Step 2**: Add to training script

```python
from emagerlib.models.my_model import MyCustomModel

model = MyCustomModel(input_size=64*200, num_classes=5)
# Use in your training pipeline
```

### Adding Visualization Tools

**Step 1**: Create visualization file

```python
# emagerlib/visualization/my_viz.py
import matplotlib.pyplot as plt

def visualize_predictions(predictions, true_labels):
    # Your visualization code
    pass
```

**Step 2**: Use in examples

```python
from emagerlib.visualization.my_viz import visualize_predictions

# In your script
visualize_predictions(preds, labels)
```

### Adding Utility Functions

Add to `emagerlib/utils/utils.py`:

```python
def my_utility_function(data):
    """
    Description of what it does.
    
    Args:
        data: Input data
        
    Returns:
        Processed data
    """
    # Implementation
    return processed_data
```

## Testing

### Run All Tests

```bash
emager-run-tests
```

### Run Specific Tests

```bash
# Test configuration
python tests/test_config.py

# Test utilities
python tests/test_utils.py

# With pytest
pytest tests/test_config.py -v
```

### Writing Tests

Create test file `tests/test_my_feature.py`:

```python
import unittest
from emagerlib.my_module import my_function

class TestMyFeature(unittest.TestCase):
    def test_basic_functionality(self):
        result = my_function(input_data)
        self.assertEqual(result, expected_output)
        
    def test_edge_case(self):
        result = my_function(edge_case_data)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## Code Standards

### Style Guide

- **PEP 8** compliance
- **Docstrings** for all public functions/classes
- **Type hints** where appropriate
- **Comments** for complex logic

### Example

```python
from typing import List, Optional
from pathlib import Path

def process_emg_data(data: List[float], window_size: int = 200) -> Optional[List[float]]:
    """
    Process EMG data with windowing.
    
    Args:
        data: Raw EMG samples
        window_size: Size of processing window
        
    Returns:
        Processed data or None if invalid
    """
    if len(data) < window_size:
        return None
        
    # Processing logic
    processed = [d * 1.0 for d in data[:window_size]]
    return processed
```

### Code Formatting

```bash
# Format with black
black emagerlib/

# Check with flake8
flake8 emagerlib/ --max-line-length=100

# Type checking
mypy emagerlib/ --ignore-missing-imports
```

## Contributing

### Git Flow Workflow

This project uses **Git Flow** for branch management. Git Flow provides a robust framework for managing releases and features.

**Learn more**: [Git Flow Introduction](https://danielkummer.github.io/git-flow-cheatsheet/index.html)

#### Branch Architecture

- **`main`**: Production-ready code, stable releases only
- **`develop`**: Integration branch for features, latest development code
- **`feature/*`**: New features (branched from `develop`)
- **`release/*`**: Release preparation (branched from `develop`)
- **`hotfix/*`**: Urgent fixes for production (branched from `main`)

#### Basic Usage

**Starting a new feature**:
```bash
# From develop branch
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

**Finishing a feature**:
```bash
# Merge back to develop
git checkout develop
git merge feature/my-new-feature
git push origin develop
git branch -d feature/my-new-feature
```

**Using git-flow tool** (optional):
```bash
# Initialize git-flow
git flow init

# Start feature
git flow feature start my-new-feature

# Finish feature (merges to develop)
git flow feature finish my-new-feature
```

### Contribution Workflow

1. **Fork** repository on GitHub
2. **Clone** your fork
3. **Create feature branch** from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/my-new-feature
   ```
4. **Make changes** and test
5. **Commit** with clear messages:
   ```bash
   git commit -m "feat: add new hand controller for XYZ hand"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
7. **Open Pull Request** targeting `develop` branch

### Pull Request Guidelines

- Clear description of changes
- Reference related issues
- All tests passing
- Code follows style guide
- Documentation updated

### Commit Messages

**Format**: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `style`: Formatting

Examples:
```bash
git commit -m "feat: add support for ABC prosthetic hand"
git commit -m "fix: resolve serial communication timeout"
git commit -m "docs: update configuration guide"
```

## Best Practices

### Configuration

- Always use config files, don't hardcode values
- Provide example configs for new features
- Document new config parameters

### Code Organization

- One class per file when possible
- Group related functionality
- Keep functions focused and small
- Use meaningful names

### Documentation

- Update README when adding features
- Add docstrings to all public APIs
- Include usage examples
- Update relevant docs/ files

### Testing

- Test new features
- Include edge cases
- Test on different platforms if possible
- Don't break existing tests

## Getting Help

- Check [GitHub Discussions](https://github.com/SBIOML/emagerpy/discussions)
- Open [GitHub Issues](https://github.com/SBIOML/emagerpy/issues)
- Read existing code for examples
- Contact: etmic6@ulaval.ca

## See Also

- [Installation Guide](INSTALLATION.md)
- [Configuration Guide](CONFIGURATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
