import numpy as np

def unilateral_deviation_gain(A, B, p, q):
    """
    Given mixed strategies p (m) and q (n), compute:
      - current expected utilities u1, u2
      - best pure deviation for each player and the gain (increase in utility)
    Returns a dict with keys: gain_p1, best_dev_p1, gain_p2, best_dev_p2, u1, u2
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    u1 = float(p @ A @ q)
    u2 = float(p @ B @ q)

    # Player1 pure deviations (rows)
    row_payoffs = A @ q  # expected payoff for each pure row
    gains1 = row_payoffs - u1
    best1 = int(np.argmax(gains1))
    gain1 = float(gains1[best1])

    # Player2 pure deviations (columns)
    col_payoffs = p @ B
    gains2 = col_payoffs - u2
    best2 = int(np.argmax(gains2))
    gain2 = float(gains2[best2])

    return {
        'gain_p1': gain1,
        'best_dev_p1': best1,
        'gain_p2': gain2,
        'best_dev_p2': best2,
        'u1': u1,
        'u2': u2
    }