import numpy as np
import scipy.linalg as LA




def key_generation(l, q, rm):

    # np.random.seed(seed=42)
    r = (np.random.rand(l, 1) * (2 *rm)) - rm
    P = np.random.permutation(np.eye(q,q))


    # Here we generate $R$
    R = (np.random.rand(l, l) * (2 *rm)) - rm
    
    # check if R is invertible
    a = LA.eig(R, left=False, right=False)
    b = min(abs(a))

    while(b <= 0.2):
        R = (np.random.rand(l, l) * (2 *rm)) - rm
    
        # check if R is invertible
        a = LA.eig(R, left=False, right=False)
        b = min(abs(a))


    # R = np.eye(l, l)
    # r = np.zeros((l,1))
   
    return R, r, P

