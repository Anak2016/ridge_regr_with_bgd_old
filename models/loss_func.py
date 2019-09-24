import numpy as np
def mean_square_error(y,x,beta, lmda=0.1, l1=True):
    '''

    :param y: dim = n
    :param x: dim = n * p
    :param beta: dim = 1 * p (to automatically broadcast along dim 0)
    :param lmda:
    :param l1:
    :return:
    '''
    penalty = 0
    if l1:
        penalty =  lmda * np.square(beta).sum(axis=0)

    return np.square(y - x.dot(beta.T)).sum(axis=1) + penalty