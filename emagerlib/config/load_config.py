import importlib.util
import json
import os
import yaml
import types
from pathlib import Path
from typing import Optional, Union
from .core_config import CoreConfig
from dataclasses import fields
from emagerlib import ROOT_EMAGERLIB


def _resolve_path(value, config_dir: Path) -> Path | None:
    """Resolve path values relative to the config file directory."""
    if value is None:
        return None

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = config_dir / p
    return p.resolve()


def _expand_placeholders(value):
    """Recursively expand placeholders in config values.

    Supports:
    - ${ROOT_EMAGERLIB}
    - environment variables via ${VAR_NAME}
    """
    if isinstance(value, dict):
        return {k: _expand_placeholders(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_placeholders(v) for v in value]

    if isinstance(value, str):
        expanded = value.replace("${ROOT_EMAGERLIB}", str(ROOT_EMAGERLIB))
        return os.path.expandvars(expanded)

    return value


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

    config_dir = path.parent

    # Normalize all path-like fields to pathlib.Path
    for key in ("BASE_PATH", "MEDIA_PATH", "PRETRAINED_MODEL_PATH", "LOG_FILE_PATH", "SAVE_CONFIG_PATH"):
        if key in core and core[key] is not None:
            core[key] = _resolve_path(core[key], config_dir)

    cfg = CoreConfig(**core)
    cfg.EXTRA = extra

    cfg.CONFIG_FILE_NAME = str(path.name)
    cfg.CONFIG_FILE_PATH = str(path.parent)

    return cfg


def load_config(path: Union[str, Path]) -> CoreConfig:
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
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == '.py':

        data = {"ROOT_EMAGERLIB": ROOT_EMAGERLIB}
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

    data = _expand_placeholders(data)

    return _build_config(data, path)
