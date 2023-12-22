import numpy as np
import scipy.linalg as LA

r"""
    Computes the Quadratic program matrices using the system and cost matrices.
    The final program that we want to build is:
    $$
	\min_{z} \quad \frac{1}{2}z^{\top} H z + x_0^{\top}Fz
	+ { x_0^{\top} Yx_0} 
    $$
    $$
    \quad \text{s.t.} \quad  Gz \leq W + Ox_0
    $$
"""

def QP_matrices(N, A, B, C, P, Q, R, umin, umax, ymin, ymax):

    """
    Computes the Quadratic program matrices using the system and cost matrices
    The matrices are given in the Section 4 of the paper.

    """
    # We get the size of A
    n = np.shape(A)[0]

    # We get the size of B
    m = np.shape(B)[1]

    # create identity matrix
    I_n = np.eye(n)

    # The place holder for the S matrix
    S = np.zeros((N * n, N * n))

    # This is the S matrix, which contains the power of A
    for j in range(N):
        
        # Here we build the power of A
        Sj = np.vstack([I_n] + [np.linalg.matrix_power(A, i)
                               for i in range(1, N - (j))])
        # We replace the Sj in the correct place in S
        S[(j*n): , (j*n):(j+1)*n] = Sj
    

    # The matrix calT can be computed using S
    calT = S[:, 0:n] @ A

    # We need Bbar for next step
    Bbar = LA.kron(np.eye(N), B)

    # This is the final calS that we use
    calS = S @ Bbar


    # The compact calQ
    calQ = LA.kron(np.eye(N), Q)
    # The last block is P
    calQ[(N-1) * n:, (N-1)*n:] = P

    # Here we compute calR
    calR = LA.kron(np.eye(N), R)

    # Here are the final H and F matrices that we usually work with
    H = 2 * (calR + calS.T @ calQ @ calS)
    F = 2* (calT.T @ calQ @ calS)

    # This is the constant term that we do not include in the cost function, but we compute it here
    Y = Q + calT.T @ calQ @ calT


    # Here we construct the constraints 
    C_block = LA.kron(np.eye(N), C)

    calg = C_block @ calS
    calo = C_block @ calT


    G = np.vstack((np.eye(N*m), -np.eye(N*m), calg, -calg))
    O = np.vstack((np.zeros((N*m, n)), np.zeros((N*m, n)), -calo, calo ))
    
    Umax = LA.kron(np.ones((N, 1)), umax)
    Umin = LA.kron(-np.ones((N, 1)), umin)
    Ymax = LA.kron(np.ones((N, 1)), ymax)
    Ymin = LA.kron(-np.ones((N, 1)), ymin)

    W = np.vstack((Umax, Umin, Ymax, Ymin))



    return H, F, Y, G, W, O, calS, calR, calQ, S, calT