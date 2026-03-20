import serial.tools.list_ports
import sys
import time
import logging

logger = logging.getLogger(__name__)

def find_port(vid, pid):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == vid and port.pid == pid:
            logger.info(f"Found device: {port.device}")
            return port.device
    raise ValueError("Device not found")
    return None

def find_psoc():
    return find_port(0x04b4, 0xf155)

def find_pico():
    return find_port(0x2e8a, 0x0005)

def find_nrf_base_station():
    return find_port(12259, 256)

def find_teensy():
    # Teensy 4.1 VID:PID is 16c0:0483 in serial mode, but it can be in different modes so we check for multiple possibilities
    try:
        return find_port(0x16c0, 0x0483) # Serial mode
    except ValueError:
        logger.warning("Teensy not found in serial mode, trying USB mode...")
        try:
            return find_port(0x16c0, 0x0478) # USB mode
        except ValueError:
            logger.warning("Teensy not found in USB mode either.")
            raise ValueError("Teensy device not found in any known mode.")
