import numpy as np

def normalize(vec):
    vec = np.array(vec, dtype=float)
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s

def fictitious_play(A, B, iterations=20000, tol=1e-7):
    """
    Fictitious play for bimatrix games.
    Returns (p, q, iters) - empirical mixed strategies and number of iterations used.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape

    cum_p = np.zeros(m)
    cum_q = np.zeros(n)

    # start uniform
    p = np.ones(m) / m
    q = np.ones(n) / n
    cum_p += p
    cum_q += q
    last_p = p.copy()
    last_q = q.copy()

    for t in range(1, iterations + 1):
        # Player1 best response to q
        exp_p = A @ q
        best_p_idxs = np.where(np.abs(exp_p - exp_p.max()) <= 1e-12)[0]
        br_p = np.zeros(m); br_p[best_p_idxs] = 1.0 / best_p_idxs.size

        # Player2 best response to p
        exp_q = p @ B
        best_q_idxs = np.where(np.abs(exp_q - exp_q.max()) <= 1e-12)[0]
        br_q = np.zeros(n); br_q[best_q_idxs] = 1.0 / best_q_idxs.size

        cum_p += br_p
        cum_q += br_q

        p = cum_p / cum_p.sum()
        q = cum_q / cum_q.sum()

        if np.linalg.norm(p - last_p, ord=1) < tol and np.linalg.norm(q - last_q, ord=1) < tol:
            return p, q, t
        last_p = p.copy()
        last_q = q.copy()

    return p, q, iterations