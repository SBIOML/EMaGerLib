"""
Powerline notch frequency, read from the config (NOTCH_FREQ).

The powerline is 60 Hz in North America and 50 Hz in Europe (and most of the world). Every script
that filters EMG uses NOTCH_FREQ from the config, and every data collection script saves it in the
session's collection_details.json ("notch_freq"), so the recordings document where they were made.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_NOTCH_FREQS = (50, 60)
NOTCH_BANDWIDTH = 3  # Hz, width of the libemg notch filter


def get_notch_freq(cfg) -> int:
    """NOTCH_FREQ of the config (50 or 60 Hz)."""
    freq = getattr(cfg, "NOTCH_FREQ", None)
    if freq is None:
        raise ValueError("NOTCH_FREQ is missing from the config: set it to 60 (America) or 50 (Europe).")
    if int(freq) not in VALID_NOTCH_FREQS:
        raise ValueError(f"NOTCH_FREQ = {freq} in the config: it must be one of {VALID_NOTCH_FREQS} Hz.")
    return int(freq)


def notch_filter_dictionary(cfg, bandwidth: float = NOTCH_BANDWIDTH) -> dict:
    """libemg Filter dictionary of the powerline notch: filter.install_filters(notch_filter_dictionary(cfg))."""
    return {"name": "notch", "cutoff": get_notch_freq(cfg), "bandwidth": bandwidth}


def save_notch_freq(data_folder, cfg) -> None:
    """Add "notch_freq" to the collection_details.json of a recorded session (if the file exists)."""
    details_file = Path(data_folder) / "collection_details.json"
    if not details_file.exists():
        logger.warning(f"{details_file} not found: notch frequency not saved (nothing was recorded?).")
        return
    with open(details_file, "r") as f:
        details = json.load(f)
    details["notch_freq"] = get_notch_freq(cfg)
    with open(details_file, "w") as f:
        json.dump(details, f)
    logger.info(f"Notch frequency {details['notch_freq']} Hz saved in {details_file}")
