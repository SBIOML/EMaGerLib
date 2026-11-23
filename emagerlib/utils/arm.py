"""
Arm wearing the bracelet, read from the config (ARM = "left" or "right").

Every screen-guided training (SGT) script checks it before recording and saves it in the session's
collection_details.json ("arm"), so each recording documents which arm it comes from.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_ARMS = ("left", "right")


def get_arm(cfg) -> str:
    """ARM of the config: "left" or "right" (case-insensitive)."""
    arm = getattr(cfg, "ARM", None)
    if arm is None:
        raise ValueError('ARM is missing from the config: set it to "left" or "right" (arm wearing the bracelet).')
    arm = str(arm).strip().lower()
    if arm not in VALID_ARMS:
        raise ValueError(f'ARM = {cfg.ARM!r} in the config: it must be "left" or "right".')
    return arm


def save_arm(data_folder, cfg) -> None:
    """Add "arm" to the collection_details.json of a recorded session (if the file exists)."""
    details_file = Path(data_folder) / "collection_details.json"
    if not details_file.exists():
        logger.warning(f"{details_file} not found: arm not saved (nothing was recorded?).")
        return
    with open(details_file, "r") as f:
        details = json.load(f)
    details["arm"] = get_arm(cfg)
    with open(details_file, "w") as f:
        json.dump(details, f)
    logger.info(f"Arm '{details['arm']}' saved in {details_file}")
