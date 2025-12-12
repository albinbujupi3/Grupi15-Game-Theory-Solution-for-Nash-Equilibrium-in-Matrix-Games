# main.py
import argparse
import numpy as np
from core.payoff_matrix import PayoffMatrix
from core.solver_lp import solve_zero_sum_lp
from core.solver_iterative import fictitious_play
from core.equilibrium_checker import unilateral_deviation_gain
from algorithms.lemke_howson import lemke_howson_all
from utils.matrix_loader import load_matrix
from utils.matrix_examples import EXAMPLES


def print_equilibrium(p, q, info=None):
    np.set_printoptions(precision=6, suppress=True)
    print("Player 1 strategy p:", p)
    print("Player 2 strategy q:", q)
    if info:
        for k, v in info.items():
            print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Nash Equilibrium solver for two-player matrix games.")
    parser.add_argument('--a', type=str, help="Path to payoff matrix A (.npy or .csv)")
    parser.add_argument('--b', type=str, help="Path to payoff matrix B (.npy or .csv)")
    parser.add_argument('--example', type=str, choices=list(EXAMPLES.keys()), help="Predefined example name")
    parser.add_argument('--method', type=str, choices=['auto','lp-zero-sum','fictitious-play','lemke-howson'], default='auto')
    parser.add_argument('--fp-iters', type=int, default=20000)
    args = parser.parse_args()

    if args.example:
        A, B = EXAMPLES[args.example]
    else:
        if not args.a or not args.b:
            parser.error("Provide both --a and --b or use --example")
        A = load_matrix(args.a)
        B = load_matrix(args.b)

    # Validate
    pm = PayoffMatrix(A, B)

    method = args.method
    if method == 'auto':
        if pm.is_zero_sum():
            method = 'lp-zero-sum'
        else:
            method = 'fictitious-play'

    print("Using method:", method)
    if method == 'lp-zero-sum':
        p, q, v = solve_zero_sum_lp(pm.A)
        info = {'value': v}
        dev = unilateral_deviation_gain(pm.A, pm.B, p, q)
        info.update(dev)
        print_equilibrium(p, q, info)
    elif method == 'fictitious-play':
        p, q, iters = fictitious_play(pm.A, pm.B, iterations=args.fp_iters)
        info = {'iterations': iters}
        dev = unilateral_deviation_gain(pm.A, pm.B, p, q)
        info.update(dev)
        print_equilibrium(p, q, info)
    elif method == 'lemke-howson':
        try:
            eqs = lemke_howson_all(pm.A, pm.B)
            for idx, (p, q) in enumerate(eqs, start=1):
                print(f"--- Equilibrium {idx} ---")
                dev = unilateral_deviation_gain(pm.A, pm.B, p, q)
                print_equilibrium(p, q, dev)
        except Exception as e:
            print("Lemke-Howson failed or nashpy not installed:", e)

if __name__ == '__main__':
    main()
