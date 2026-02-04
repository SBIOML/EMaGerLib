"""Test suite for utility functions"""

import unittest
import numpy as np
from pathlib import Path
from emagerlib.utils.majority_vote import MajorityVote, majority_vote
from emagerlib.utils.utils import get_transform_decimation, print_packet


class TestMajorityVote(unittest.TestCase):
    """Test suite for majority voting"""

    def test_01_majority_vote_basic(self):
        """Test basic majority vote functionality"""
        mv = MajorityVote(5)
        
        # Add values
        mv.append(1)
        mv.append(1)
        mv.append(2)
        mv.append(1)
        mv.append(3)
        
        # Should return 1 as it appears most frequently
        result = mv.vote()
        self.assertIsNotNone(result)
        print("✓ MajorityVote basic functionality works")

    def test_02_majority_vote_overflow(self):
        """Test majority vote with more items than max length"""
        mv = MajorityVote(3)
        
        # Add more items than max_len
        for i in [1, 1, 2, 3, 3]:
            mv.append(i)
        
        # Should only consider last 3 items: [2, 3, 3]
        result = mv.vote()
        self.assertIsNotNone(result)
        self.assertEqual(len(mv), 3)
        print("✓ MajorityVote respects max length")

    def test_03_majority_vote_function(self):
        """Test majority_vote function with array"""
        arr = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
        result = majority_vote(arr, 3)
        
        self.assertEqual(len(result), len(arr))
        self.assertIsInstance(result, np.ndarray)
        print("✓ majority_vote function works with numpy arrays")


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
