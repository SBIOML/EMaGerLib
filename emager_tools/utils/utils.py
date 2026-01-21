import logging as log
import numpy as np


def set_logging():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    log.basicConfig(level=log.DEBUG, format=FORMAT)


def get_transform_decimation(transform):
    """
    Get the decimation factor of SigProc function `transform`.
    """
    return 1000 // len(transform(np.zeros((1000, 1))))


def print_packet(packet: bytes | bytearray | list[int], stuffed: bool = False) -> None:
    """Pretty-print a packet similar to the provided C++ `printPacket`.

    Args:
        packet: bytes, bytearray or list of ints (0-255)
        stuffed: whether the packet uses stuffing (affects field labels and warnings)
    """
    # normalize to list of ints
    if isinstance(packet, (bytes, bytearray)):
        data = list(packet)
    else:
        data = list(packet)

    length = len(data)
    print(f"Packet length: {length}")
    if length < 15 and not stuffed:
        print("Warning: expected >= 15 bytes (address + header + 12-byte payload + checksum)")
    if length < 17 and stuffed:
        print("Warning: expected >= 17 bytes (stuffing + address + header + 12-byte payload + checksum + stuffing)")

    print("Idx | Hex   | Dec   | Field")
    print("-----------------------------")
    for i, b in enumerate(data):
        # Index
        idx_str = str(i)

        # Hex (always two hex digits)
        hex_str = f"0x{b:02X}"

        # Decimal
        dec_str = str(b)

        # Field label logic mirrors the C++ code
        if i == 0:
            field = "Stuffing" if stuffed else "Address"
        elif i == 1:
            field = "Address" if stuffed else "Header"
        elif i == 2:
            field = "Header" if stuffed else "Payload"
        elif 3 <= i <= length-4:
            if stuffed and (b == 0x7D or b == 0x7E) and length > 17:
                field = "Stuffed"
            else:
                field = "Payload"
        elif i == length-3:
            field = "Payload" if stuffed else "Checksum"
        elif i == length-2:
            field = "Checksum" if stuffed else "Extra"
        elif i == length-1 and stuffed:
            field = "Stuffing"
        else:
            field = "Extra"

        # Align columns roughly: idx left, hex width 6, dec width 6
        print(f"{idx_str:<3} | {hex_str:<6} | {dec_str:<6} | {field}")

    print()
