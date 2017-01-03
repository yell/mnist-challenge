import random

def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]