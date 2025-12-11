import numpy as np
from core.equilibrium_checker import unilateral_deviation_gain

def test_checker_on_pure_ne():
    # Simple coordination where (0,0) is pure NE
    A = np.array([[2,0],[0,1]])
    B = np.array([[1,0],[0,2]])
    p = np.array([1.0, 0.0])
    q = np.array([1.0, 0.0])
    res = unilateral_deviation_gain(A, B, p, q)
    assert res['gain_p1'] <= 0 + 1e-8
    assert res['gain_p2'] <= 0 + 1e-8