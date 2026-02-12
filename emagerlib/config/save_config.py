import json
import yaml
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_config(cfg, path: Path, name: str | None = None, file_format: str = "json"):
    """Save configuration to file.
    
    Args:
        cfg: Configuration object to save
        path: Directory path to save the file
        name: Base name for the file (without extension)
        file_format: Format to save ('json' or 'yaml')
    
    Returns:
        Path to the saved file
    """
    path.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = cfg.__class__.__name__

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine file extension
    if file_format.lower() == "yaml":
        file = path / f"{name}_{timestamp}.yaml"
    else:  # default to json
        file = path / f"{name}_{timestamp}.json"
    
    # Convert config to dict and handle Path objects
    config_dict = asdict(cfg)
    
    # Convert Path objects and other non-serializable types to strings
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert any other type to string
            return str(obj)
    
    config_dict = make_serializable(config_dict)

    # Save to file
    with open(file, "w") as f:
        if file_format.lower() == "yaml":
            # Use safe_dump to avoid Python-specific tags
            yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False, sort_keys=False)
        else:  # json
            json.dump(config_dict, f, indent=4, default=str)

    return file

def print_config(cfg):
    logger.debug("===== CONFIG =====")
    for k, v in asdict(cfg).items():
        logger.debug(f"{k}: {v}")
