from collections import deque
from scipy import stats
import numpy as np


class MajorityVote(deque):
    def __init__(self, max_len):
        """
        MajorityVote object abstracts the classic majority voting algorithm.

        :param max_len: int, the maximum length of the queue

        Example:
        >>> import random as rd
        >>> q = MajorityVote(10)
        >>> for _ in range(100):
        >>>    r = rd.randint(0, 6)
        >>>    q.append(r)
        >>>    vote = q.vote() # Get the majority vote
        """
        super().__init__(maxlen=max_len)

    def vote(self) -> np.ndarray:
        return stats.mode(self).mode


def majority_vote(arr: np.ndarray, n_votes):
    # arr.shape == (n_votes,)
    ret = np.zeros((0,), dtype=np.uint8)
    q = MajorityVote(n_votes)
    for i in arr:
        q.append(i)
        ret = np.append(ret, q.vote())
    return ret
