"""Test suite for transition screen-guided training"""

import unittest
from emagerlib.utils.transition_sgt import eulerian_gesture_sequence


class TestEulerianGestureSequence(unittest.TestCase):
    """Test suite for the Eulerian gesture transition order"""

    def test_01_every_transition_once(self):
        """Each ordered transition i -> j (i != j) appears exactly once"""
        for n in range(2, 12):
            seq = eulerian_gesture_sequence(n)
            transitions = list(zip(seq[:-1], seq[1:]))
            expected = {(i, j) for i in range(n) for j in range(n) if i != j}

            self.assertEqual(len(seq), n * (n - 1) + 1)
            self.assertEqual(len(transitions), len(expected))
            self.assertEqual(set(transitions), expected)
        print("eulerian_gesture_sequence covers every transition exactly once")

    def test_02_starts_and_ends_on_start(self):
        """The sequence is a circuit starting and ending on the start gesture"""
        seq = eulerian_gesture_sequence(5, start=2)
        self.assertEqual(seq[0], 2)
        self.assertEqual(seq[-1], 2)
        print("eulerian_gesture_sequence is a closed circuit")


if __name__ == "__main__":
    unittest.main()
