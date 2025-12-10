import numpy as np

class PayoffMatrix:
    """
    Holds payoff matrices for two players.
    A and B must be same shape (m x n).
    """
    def __init__(self, A, B):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        if self.A.shape != self.B.shape:
            raise ValueError("A and B must have same shape (m x n).")
        if self.A.ndim != 2:
            raise ValueError("A and B must be 2D matrices.")

    def n_strategies_p1(self):
        return self.A.shape[0]

    def n_strategies_p2(self):
        return self.A.shape[1]

    def is_zero_sum(self, tol=1e-8):
        return np.allclose(self.A + self.B, 0.0, atol=tol)

