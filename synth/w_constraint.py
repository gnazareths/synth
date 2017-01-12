import numpy as np

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1
