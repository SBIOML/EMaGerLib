"""Test suite for control constants"""

import unittest
from emager_tools.control.constants import Finger, Gesture, PIN_SERVO


class TestConstants(unittest.TestCase):
    """Test suite for control constants and enumerations"""

    def test_01_gesture_unique_values(self):
        """Test that gesture values are unique"""
        gesture_attrs = [attr for attr in dir(Gesture) if not attr.startswith('_')]
        gesture_values = [getattr(Gesture, attr) for attr in gesture_attrs]
        
        self.assertEqual(len(gesture_values), len(set(gesture_values)),
                        "All gesture values should be unique")
        print("✓ All Gesture values are unique")

    def test_02_pin_servo_access(self):
        """Test PIN_SERVO can be accessed by finger index"""
        # Should be able to access pins using Finger constants
        thumb_pin = PIN_SERVO[Finger.THUMB]
        index_pin = PIN_SERVO[Finger.INDEX]
        
        self.assertEqual(thumb_pin, 13)
        self.assertEqual(index_pin, 14)
        print("✓ PIN_SERVO accessible by Finger constants")


if __name__ == '__main__':
    unittest.main()
