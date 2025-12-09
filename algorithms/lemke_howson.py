# algorithms/lemke_howson.py
import numpy as np

def lemke_howson_all(A, B):
    """
    Wrapper to compute Lemke-Howson / support enumeration equilibria using nashpy if available.
    Returns list of (p, q) numpy arrays.
    If nashpy is not installed, raises RuntimeError.
    """
    try:
        import nashpy as nash
    except Exception as e:
        raise RuntimeError("nashpy is required for Lemke-Howson/support-enumeration features. Install with `pip install nashpy`.") from e

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    game = nash.Game(A, B)

    eqs = []
    for eq in game.support_enumeration():
        p, q = eq
        eqs.append((np.array(p, dtype=float), np.array(q, dtype=float)))

    # deduplicate approx
    unique = []
    for p, q in eqs:
        found = False
        for rp, rq in unique:
            if np.allclose(p, rp, atol=1e-8) and np.allclose(q, rq, atol=1e-8):
                found = True
                break
        if not found:
            unique.append((p, q))
    return unique