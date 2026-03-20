"""Test suite for utility functions"""

import unittest
import numpy as np
from pathlib import Path
from emagerlib.utils.utils import get_transform_decimation, print_packet


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for general utility functions"""

    def test_01_get_transform_decimation(self):
        """Test transform decimation calculation"""
        # Create a simple transform that halves the data
        def half_transform(data):
            return data[::2]
        
        decimation = get_transform_decimation(half_transform)
        self.assertEqual(decimation, 2)
        print("✓ get_transform_decimation works correctly")

    def test_02_print_packet_basic(self):
        """Test packet printing doesn't crash"""
        packet = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        
        # Should not raise exception
        try:
            print_packet(packet, stuffed=False)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
        print("✓ print_packet handles basic packets")

    def test_03_print_packet_types(self):
        """Test packet printing with different input types"""
        # Test with bytes
        packet_bytes = bytes([0x01, 0x02, 0x03])
        print_packet(packet_bytes)
        
        # Test with bytearray
        packet_bytearray = bytearray([0x01, 0x02, 0x03])
        print_packet(packet_bytearray)
        
        # Test with list
        packet_list = [0x01, 0x02, 0x03]
        print_packet(packet_list)
        
        print("✓ print_packet handles multiple input types")


if __name__ == '__main__':
    unittest.main()
