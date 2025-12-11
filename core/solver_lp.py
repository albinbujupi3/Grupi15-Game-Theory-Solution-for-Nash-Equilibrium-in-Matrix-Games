import numpy as np

def normalize(vec):
    vec = np.array(vec, dtype=float)
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s

def solve_zero_sum_lp(A):
    """
    Solve zero-sum game with payoff matrix A (player1) and -A (player2)
    using linear programming (scipy.optimize.linprog).
    Returns (p, q, v) where p: distribution for player1, q: distribution for player2, v: game value.
    """
    try:
        from scipy.optimize import linprog
    except Exception as e:
        raise RuntimeError("scipy is required for LP solver (install scipy).") from e

    A = np.array(A, dtype=float)
    m, n = A.shape

    # Variables: p_0..p_{m-1}, v
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -v -> maximize v

    # Constraints: for each column j: -sum_i p_i * A[i,j] + v <= 0
    A_ub = np.zeros((n, m + 1))
    b_ub = np.zeros(n)
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, m] = 1.0

    # Equality: sum p_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)

    p = normalize(res.x[:m])
    v = res.x[-1]

    # Compute q as distribution over columns that minimize expected payoff
    expected_cols = p @ A
    min_val = expected_cols.min()
    tol = 1e-9
    support = np.where(np.abs(expected_cols - min_val) <= tol)[0]
    if support.size == 0:
        support = np.array([expected_cols.argmin()])
    q = np.zeros(n)
    q[support] = 1.0 / support.size
    q = normalize(q)
    return p, q, float(v)