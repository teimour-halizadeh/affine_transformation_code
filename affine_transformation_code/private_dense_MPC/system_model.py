import numpy as np
import scipy.linalg as LA

from private_dense_MPC.constants import T1, T2, T3, T4
from private_dense_MPC.constants import A1, A2, A3, A4, gamma1, gamma2, k1, k2, kc
'''
The models that we want to use as our case study are given in this submodule
'''

def system_model():
    '''
    The system's model for Quadratic Tank process. The parameters are from the paper 
    <<The Quadruple-Tank Process: A Multivariable
    Laboratory Process with an Adjustable Zero>>.
    '''
    Ac = np.array([
        [-1/T1, 0, A3/(A1*T3), 0],
        [ 0, -1/T2, 0, A4/(A2*T4)],
        [0, 0,  -1/(T3), 0],
        [0, 0, 0, -1/(T4)]
    ])
    Bc = np.array([
        [(gamma1*k1)/A1, 0],
        [0, (gamma2 * k2)/A2],
        [0, ((1-gamma2)*k2)/A3],
        [((1-gamma1)*k1)/A4, 0]
    ])
    Cc = np.array([
        [kc, 0, 0, 0],
        [0, kc, 0, 0]
    ])
    Dc = np.zeros((2,2))

    return Ac, Bc, Cc, Dc

# def Robot_system_model():

#     a = np.array([[1, 1],[0, 1]])
#     b = np.array([[0.5],[1]])

    
#     A = LA.kron(np.eye(2), a)
#     B = LA.kron(np.eye(2), b)
#     C = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
#     D = np.zeros((2,2))



#     # An Artificial unstable system 
#     A = np.array([[1.2, 0], 
#                   [0, 0.9]])
#     B = np.array([[1],[0.1]])

#     C = np.array([[1, 0],
#                   [0, 1]])

#     D = np.zeros((1,1))


#     return A, B, C, D


    