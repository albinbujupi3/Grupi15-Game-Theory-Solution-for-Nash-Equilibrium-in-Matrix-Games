import numpy as np

EXAMPLES = {
    'prisoners_dilemma': (
        np.array([[3,0],[5,1]]),
        np.array([[3,5],[0,1]])
    ),
    'matching_pennies': (
        np.array([[1,-1],[-1,1]]),
        np.array([[-1,1],[1,-1]])
    ),
    'battle_of_the_sexes': (
        np.array([[2,0],[0,1]]),
        np.array([[1,0],[0,2]])
    )
}