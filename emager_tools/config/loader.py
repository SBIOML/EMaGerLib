import importlib.util
import json
import yaml
import types
from pathlib import Path
from .core import CoreConfig
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

def load_py_config(path: str | Path) -> CoreConfig:
    path = Path(path)

    data = {}
    with open(path) as f:
        exec(f.read(), data)

    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return _build_config(data, path)

def load_json_config(path: str | Path) -> CoreConfig:
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return _build_config(data, path)

def load_yaml_config(path: str | Path) -> CoreConfig:
    path = Path(path)

    with open(path) as f:
        data = yaml.safe_load(f)

    return _build_config(data, path)
