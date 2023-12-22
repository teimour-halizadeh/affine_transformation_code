import numpy as np
import scipy.linalg as LA


'''
The module contains some of the functions that we need,
but they are not part of the main results.
'''


def cont_mat(A, B):
    n = np.size(A, 0)
    ctrb = np.hstack([B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, n)])
    return ctrb

def obv_mat(A,C):
    n = np.size(A, 0)
    obsv = np.vstack([C] + [C @ np.linalg.matrix_power(A, i)
                               for i in range(1, n)])
    return obsv



