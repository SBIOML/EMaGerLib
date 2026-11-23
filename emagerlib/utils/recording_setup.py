"""
Notch frequency and arm fields of the SGT configuration windows (dearpygui).

The values come from the config (NOTCH_FREQ, ARM) and can be corrected in the window before starting, in case
the config was not updated. The chosen values are the ones saved in collection_details.json.
"""
import logging

import dearpygui.dearpygui as dpg

from emagerlib.utils.arm import get_arm
from emagerlib.utils.notch import get_notch_freq

logger = logging.getLogger(__name__)

NOTCH_CHOICES = ["50", "60"]
ARM_CHOICES = ["left", "right"]
TAGS = ("__dc_notch_freq", "__dc_arm", "__dc_setup_error")


def config_or_none(getter, cfg):
    """Value of the config (get_notch_freq / get_arm), or None with a warning if missing or invalid:
    it is then chosen in the configuration window."""
    try:
        return getter(cfg)
    except ValueError as error:
        logger.warning(f"{error} Choose it in the configuration window.")
        return None


def recording_setup_from_config(cfg):
    """(notch_freq, arm) of the config, None for a missing / invalid one."""
    return config_or_none(get_notch_freq, cfg), config_or_none(get_arm, cfg)


def add_setup_fields(notch_freq, arm, width=100, **group_kwargs):
    """Add the "Notch filter (Hz)" and "Arm" fields (pre-filled, empty if None) and an error line.
    group_kwargs are passed to the dearpygui group (e.g. parent=..., before=...)."""
    with dpg.group(**group_kwargs):
        with dpg.group(horizontal=True):
            dpg.add_text("Notch filter (Hz):")
            dpg.add_combo(NOTCH_CHOICES, tag="__dc_notch_freq", width=width,
                          default_value=str(notch_freq) if notch_freq else "")
            dpg.add_spacer(width=20)
            dpg.add_text("Arm:")
            dpg.add_combo(ARM_CHOICES, tag="__dc_arm", width=width, default_value=arm or "")
            dpg.add_spacer(width=20)
            dpg.add_text("Check these: they are saved with the data", color=(150, 150, 150))
        dpg.add_text("", tag="__dc_setup_error", color=(255, 90, 90))


def read_setup_fields(config_notch_freq, config_arm):
    """(notch_freq, arm) chosen in the window, or None (with a message in the window) if one is missing.
    A value different from the config is logged."""
    notch, arm = dpg.get_value("__dc_notch_freq"), dpg.get_value("__dc_arm")
    missing = [name for name, value in (("the notch frequency", notch), ("the arm", arm)) if not value]
    if missing:
        dpg.set_value("__dc_setup_error", f"Choose {' and '.join(missing)} before starting.")
        return None
    dpg.set_value("__dc_setup_error", "")
    notch = int(notch)
    for name, chosen, configured in (("Notch frequency", notch, config_notch_freq), ("Arm", arm, config_arm)):
        if chosen != configured:
            logger.warning(f"{name} changed in the GUI: {chosen} (config: {configured}). Saved with the data.")
    return notch, arm


def delete_setup_fields():
    for tag in TAGS:
        if dpg.does_alias_exist(tag):
            dpg.delete_item(tag)
