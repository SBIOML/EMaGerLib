import serial.tools.list_ports
import sys
import time

def find_port(vid, pid):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == vid and port.pid == pid:
            print(f"Found device: {port.device}")
            return port.device
    raise ValueError("Device not found")
    return None

def find_psoc():
    return find_port(0x04b4, 0xf155)

def find_pico():
    return find_port(0x2e8a, 0x0005)

def find_nrf_base_station():
    return find_port(12259, 256)
