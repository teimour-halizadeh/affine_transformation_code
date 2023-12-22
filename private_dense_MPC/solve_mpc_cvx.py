
from qpsolvers import solve_qp
import numpy as np



def solve_mpc_cvx(H_tilde, f_tilde, G_tilde, e_tilde):

    l = np.size(H_tilde, 1)

    z = cp.Variable(l)

    # Create two constraints.
    constraints = [G_tilde @ z <= e_tilde ]



    # Form objective.
    obj = cp.Minimize((1/2)*cp.quad_form(z, H_tilde) + (f_tilde.T) @ z)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  

    return prob.status, prob.value, z.value.reshape(l, 1)



def solve_mpc_qp(H_tilde, f_tilde, G_tilde, e_tilde):
    l = np.size(H_tilde, 1)
    x = solve_qp(H_tilde, f_tilde, G_tilde, e_tilde, solver='daqp')

    return x.reshape(l,1)
