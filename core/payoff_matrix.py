import numpy as np


class PayoffMatrix:
    """
    Representation of a two-player (bi-matrix) game.

    Player A payoff matrix: A
    Player B payoff matrix: B

    Supports:
        - Symmetric games (A, B identical or transposed if needed)
        - Asymmetric games (different payoffs per player)

    Example:
        A = [[2, -1],
             [0,  1]]
        B = [[2,  0],
             [-1, 1]]

        game = PayoffMatrix(A, B)
    """

    def __init__(self, A, B):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)

        self._validate_dimensions()

    def _validate_dimensions(self):
        """
        Ensure the payoff matrices are compatible in size.
        """
        if self.A.shape != self.B.shape:
            raise ValueError(
                f"Payoff matrices must have same dimensions. Got {self.A.shape} and {self.B.shape}"
            )

        if len(self.A.shape) != 2:
            raise ValueError("Payoff matrices must be 2D. e.g. [[1,2],[3,4]]")

    @property
    def num_strategies_A(self):
        return self.A.shape[0]

    @property
    def num_strategies_B(self):
        return self.A.shape[1]

    def is_symmetric(self):
        """
        Determines if the game is symmetric (A = B^T).
        """
        return np.allclose(self.A, self.B.T)

    def normalize_strategy(self, strategy):
        """
        Normalize mixed strategies to sum to 1.
        """
        strategy = np.array(strategy, dtype=float)

        if strategy.sum() == 0:
            raise ValueError("Strategy vector cannot sum to zero.")

        return strategy / strategy.sum()

    def expected_payoff(self, x, y):
        """
        Calculated expected payoff for mixed strategies.

        x: Player A mixed strategy vector
        y: Player B mixed strategy vector

        Returns (u_A, u_B): expected returns
        """
        x = self.normalize_strategy(x)
        y = self.normalize_strategy(y)

        u_A = x @ self.A @ y.T
        u_B = x @ self.B @ y.T

        return float(u_A), float(u_B)

    def best_response_A(self, y):
        """
        Return best pure strategy response for A given B's mixed strategy y.
        """
        y = self.normalize_strategy(y)
        payoffs = self.A @ y.T
        return np.argmax(payoffs)

    def best_response_B(self, x):
        """
        Return best pure strategy response for B given A's mixed strategy x.
        """
        x = self.normalize_strategy(x)
        payoffs = x @ self.B
        return np.argmax(payoffs)

    def info(self):
        """
        Print formatted game information.
        """
        print("=== Bimatrix Game ===")
        print("Player A payoff matrix:\n", self.A)
        print("Player B payoff matrix:\n", self.B)
        print(f"Symmetric: {self.is_symmetric()}")
        print(f"Strategies: A={self.num_strategies_A}, B={self.num_strategies_B}")
