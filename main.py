import argparse
import numpy as np
from core.payoff_matrix import PayoffMatrix
from core.solver_lp import solve_zero_sum_lp
from core.solver_iterative import fictitious_play
from core.equilibrium_checker import unilateral_deviation_gain


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

