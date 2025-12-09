# algorithms/support_enumeration.py
import itertools
import numpy as np

def support_enumeration(A, B, tol=1e-9):
    """
    Simple support enumeration to find Nash equilibria of a bimatrix game.
    WARNING: exponential in supports; suitable for small games.
    Returns list of (p, q) equilibria (numpy arrays).
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape
    equilibria = []

    # iterate over supports sizes
    for r in range(1, m+1):
        for support_p in itertools.combinations(range(m), r):
            # p has support 'support_p'
            for s in range(1, n+1):
                for support_q in itertools.combinations(range(n), s):
                    # attempt to solve for mixed strategies with these supports
                    # Solve linear equations: for player1, expected payoff of rows in support equal; rows outside <= that payoff
                    # Let variables be probabilities for support entries
                    # Build system for p: (A_support - 1*value) * q = 0 ; easier to solve by solving linear system for q, p
                    # We'll solve by solving for p that makes support_q all best responses and q that makes support_p best responses
                    try:
                        # Solve for p: choose q such that rows in support_p have equal expected payoff and >= others
                        # We can solve small linear systems by setting equalities and normalization
                        # For unknowns q_support
                        k = len(support_q)
                        # Build matrix for equalities for player1: for i in support_p, expected_payoff_i = v
                        # Equivalent: (A[i, support_q] - A[ref_i, support_q]) * q_support = 0 for i != ref
                        if k == 0:
                            continue
                        # Solve for q_support by linear system of size (k) with normalization
                        # Build for player1: for i in support_p, E_i = sum_j A[i,j] q_j = v
                        # Subtract the last equation to eliminate v
                        M = []
                        b = []
                        for idx_i in support_p[:-1]:
                            row = []
                            for j in support_q:
                                row.append(A[idx_i, j] - A[support_p[-1], j])
                            M.append(row)
                            b.append(0.0)
                        # normalization for q
                        M.append([1.0]*k)
                        b.append(1.0)
                        M = np.array(M, dtype=float)
                        b = np.array(b, dtype=float)
                        q_support = np.linalg.solve(M, b)
                        if np.any(q_support < -tol):
                            continue
                        q_full = np.zeros(n)
                        for idx, j in enumerate(support_q):
                            q_full[j] = max(0.0, q_support[idx])
                        q_full = q_full / q_full.sum()

                        # Now solve for p_support similarly from B
                        l = len(support_p)
                        M2 = []
                        b2 = []
                        for idx_j in support_q[:-1]:
                            row = []
                            for i in support_p:
                                row.append(B[i, idx_j] - B[i, support_q[-1]])
                            M2.append(row)
                            b2.append(0.0)
                        M2.append([1.0]*l)
                        b2.append(1.0)
                        M2 = np.array(M2, dtype=float)
                        b2 = np.array(b2, dtype=float)
                        p_support = np.linalg.solve(M2, b2)
                        if np.any(p_support < -tol):
                            continue
                        p_full = np.zeros(m)
                        for idx, i in enumerate(support_p):
                            p_full[i] = max(0.0, p_support[idx])
                        p_full = p_full / p_full.sum()

                        # Verify best-response conditions
                        # Player1: rows in support have equal expected payoff and >= others
                        exp_rows = A @ q_full
                        v = exp_rows[support_p[0]]
                        if np.any(exp_rows + tol < v):
                            continue
                        # Player2:
                        exp_cols = p_full @ B
                        w = exp_cols[support_q[0]]
                        if np.any(exp_cols + tol < w):
                            continue

                        # Passed checks -> record equilibrium (rounded small negatives to zero)
                        equilibria.append((p_full, q_full))
                    except np.linalg.LinAlgError:
                        continue
                    except Exception:
                        continue
    # deduplicate approx
    unique = []
    for p, q in equilibria:
        found = False
        for rp, rq in unique:
            if np.allclose(p, rp, atol=1e-8) and np.allclose(q, rq, atol=1e-8):
                found = True
                break
        if not found:
            unique.append((p, q))
    return unique