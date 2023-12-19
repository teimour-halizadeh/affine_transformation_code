import numpy as np
from solve_mpc_cvx import solve_mpc_cvx
from solve_mpc_cvx import solve_mpc_qp
import key_generation as kg







def apply_mpc_to_sys(time_varying_trans, rm, H, F, G, W, O, x0, A, B, m, p, I, N, Nc, Dis_time, d):
    n = x0.shape[0]
    X = np.zeros((n, (I * Nc) + 1))
    X[:, [0]] = x0
    q = 2*N*m + 2*N*p    # Number of constraints 
    l = N * m   # dimension of the decision variable
    U_applied = np.zeros((m, (I * Nc)))

    f_data = np.zeros((l, I))
    zeta_data = np.zeros((l, I))

    
    R, r, P = kg.key_generation(l, q, rm)
    

    for i in range(I):

        if time_varying_trans == True:
            xx, r, P = kg.key_generation(l, q, rm)

        # This is what we send to Cloud!
        H_tilde = R.T @ H @ R
        f_tilde = R.T @ (F.T @ x0 + H @ r)
        G_tilde = P @ G @ R
        e_tilde = (P @ (W + O @ x0 - G @ r)).reshape(q, )

        # Cloud saves this data
        f_data[:,[i]] = f_tilde

        zeta = solve_mpc_qp(H_tilde, f_tilde, G_tilde, e_tilde)

        # Cloud also save this data from the results of its computation
        zeta_data[:,[i]] = zeta

        # Inverse of the transformation done by Plant
        U = R @ zeta + r


        for k in range(Nc):

            if i <Dis_time:
                X[:, [(i*Nc)+k+1]] = A @ X[:, [(i*Nc)+k]] + B @ U[k*m :(k+1)*m, :] 
            else:
                X[:, [(i*Nc)+k+1]] = A @ X[:, [(i*Nc)+k]] + B @ (U[k*m :(k+1)*m, :] + d * (0.99)**(k+((i-Dis_time)*Nc)))

        

    
        U_applied[:, (i * Nc):(i * Nc)+Nc] = U[0:Nc*m,:].reshape(m, Nc)
        x0 = X[:, [(i*Nc)+k]]
    return X, U_applied, f_data, zeta_data
    