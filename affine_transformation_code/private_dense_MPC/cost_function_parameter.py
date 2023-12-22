import numpy as np


r'''
The module simply determines the cost function parameters in the optimization
$$
	J(x,u):= x_N^{\top}Px_N + \sum_{k=0}^{N-1} x_k^{\top}Qx_k + u_k^{\top} R u_k
$$
'''

def cost_function_parameter(n, m, p):
    Q = 2 * np.eye(n)
    # P = np.zeros((n,n))
    P = 0 * np.eye(n)
    R = 1 * np.eye(m)

    # The limits for the actuators

    ul = 1
    yl = 2

    umax = ul * np.ones((m, 1))
    umin = -ul * np.ones((m, 1))
    ymax = yl * np.ones((p, 1))
    ymin = -yl * np.ones((p, 1))
    return P, Q, R, umax, umin, ymax, ymin



# def Robot_cost_function_parameter(n, m, p):
#     C = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])


#     Q = 1 * np.eye(n)
#     # Q = C.T @ Qy @ C

#     # P = C.T @ Qy @ C
#     P = 0 * np.eye(n)
#     R = 0.1 * np.eye(m)

#     # The limits for the actuators

#     ul = 10
#     yl = 2

#     umax = ul * np.ones((m, 1))
#     umin = -ul * np.ones((m, 1))
#     ymax = yl * np.ones((p, 1))
#     ymin = -yl * np.ones((p, 1))
#     return P, Q, R, umax, umin, ymax, ymin

