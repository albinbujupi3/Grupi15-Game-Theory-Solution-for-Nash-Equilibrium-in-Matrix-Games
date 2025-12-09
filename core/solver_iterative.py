# core/solver_lp.py
import numpy as np
from scipy.optimize import linprog

def solve_nash_lp(A, B):
    """
    Solve for a Nash equilibrium approximation using LP (maximin) for zero-sum like reductions.
    For general-sum bimatrix, we solve two independent maxmin problems as an approximation:
    returns mixed strategies p (len m) and q (len n).
    Note: For true bimatrix NE use support enumeration or Lemke-Howson.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape

    # Solve for Player 1: maximize v subject to p>=0, sum(p)=1, A^T p >= v
    # Transform to linprog (minimize c^T x). We'll solve using the standard trick:
    # Minimize -v  subject to A^T p - v >= 0, sum p = 1, p >= 0
    # Variables: [p_0..p_{m-1}, v]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -v  <=> maximize v

    # Inequality: for each column j: -A[:,j]^T p + v <= 0  =>  (-A[:,j], 1) dot x <= 0
    A_ub = []
    b_ub = []
    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -A[:, j]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Equality constraint sum(p) = 1:  (1...1, 0) dot x = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]  # p_i >=0, v free
    res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res1.success:
        raise RuntimeError("LP solver failed for player 1: " + str(res1.message))

    p = res1.x[:m]
    if p.sum() <= 0:
        p = np.ones(m) / m
    else:
        p = p / p.sum()

    # Solve for Player 2 similarly on B (columns are player2's pure strategies)
    # We construct LP for q of length n:
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = []
    b_ub = []
    for i in range(m):
        row = np.zeros(n + 1)
        row[:n] = -B[i, :]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n + [(None, None)]
    res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res2.success:
        raise RuntimeError("LP solver failed for player 2: " + str(res2.message))

    q = res2.x[:n]
    if q.sum() <= 0:
        q = np.ones(n) / n
    else:
        q = q / q.sum()

    return p, q
