import math
import numpy as np

def glorot(val):
    stdv = math.sqrt(6.0 / (val.size(-2) + val.size(-1)))
    return np.random.uniform(-stdv, stdv)

def differentiate(var, loss_func):
    '''differnetiate loss function with respecto var'''
    loss_func()
    pass