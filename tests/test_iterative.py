# tests/test_iterative.py
import numpy as np
from core.solver_iterative import fictitious_play

def test_prisoners_dilemma_fp():
    A = np.array([[3,0],[5,1]])
    B = np.array([[3,5],[0,1]])
    p, q, iters = fictitious_play(A, B, iterations=5000)
    # Prisoner's Dilemma pure NE is (Defect, Defect) -> index 1
    assert p.shape == (2,)
    assert q.shape == (2,)
    # Expect p and q to put most mass on index 1 (defect). Allow tolerance.
    assert p[1] > 0.8
    assert q[1] > 0.8
