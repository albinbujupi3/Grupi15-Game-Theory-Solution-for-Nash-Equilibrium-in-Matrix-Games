# tests/test_lp.py
import numpy as np
from core.solver_lp import solve_zero_sum_lp

def test_matching_pennies_lp():
    A = np.array([[1,-1],[-1,1]])
    p, q, v = solve_zero_sum_lp(A)
    assert np.allclose(p, [0.5,0.5], atol=1e-6)
    assert np.allclose(q, [0.5,0.5], atol=1e-6)
    assert abs(v) < 1e-6
