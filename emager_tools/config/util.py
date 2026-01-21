import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

def save_config(cfg, path: Path, name: str | None = None):
    path.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = cfg.__class__.__name__

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = path / f"{name}_{timestamp}.json"

    with open(file, "w") as f:
        json.dump(asdict(cfg), f, indent=4, default=str)

    return file

def print_config(cfg):
    print("===== CONFIG =====")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
