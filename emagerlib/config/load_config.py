import importlib.util
import json
import yaml
import types
from pathlib import Path
from .core_config import CoreConfig
from dataclasses import fields

def _build_config(data: dict, path: Path) -> CoreConfig:
    """Common logic for building CoreConfig from loaded data."""
    field_names = {f.name for f in fields(CoreConfig)}
    core = {}
    extra = {}

    for k, v in data.items():
        # Skip module objects and other non-serializable types
        if isinstance(v, types.ModuleType):
            continue
            
        if k in field_names:
            core[k] = v
        else:
            extra[k] = v

    if "BASE_PATH" in core:
        core["BASE_PATH"] = Path(core["BASE_PATH"])

    cfg = CoreConfig(**core)
    cfg.EXTRA = extra
    
    cfg.CONFIG_FILE_NAME = str(path.name)
    cfg.CONFIG_FILE_PATH = str(path.parent)
    
    return cfg

def load_config(path: str | Path) -> CoreConfig:
    """Load configuration from Python, JSON, or YAML file.
    File type is automatically detected based on file extension:
    - .py: Python config file
    - .json: JSON config file
    - .yaml, .yml: YAML config file
    Args:
        path: Path to the configuration file
    Returns:
        CoreConfig object with loaded configuration
    Raises:
        ValueError: If file extension is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.py':
        data = {}
        with open(path) as f:
            exec(f.read(), data)
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        
    elif suffix == '.json':
        with open(path) as f:
            data = json.load(f)
            
    elif suffix in ('.yaml', '.yml'):
        with open(path) as f:
            data = yaml.safe_load(f)
            
    else:
        raise ValueError(f"Unsupported config file type: {suffix}. Use .py, .json, .yaml, or .yml")
    
    return _build_config(data, path)
