
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg as LA
from scipy.signal import cont2discrete



import private_dense_MPC.QPMatrices as QP
from private_dense_MPC.apply_mpc_to_sys import apply_mpc_to_sys
from private_dense_MPC.cost_function_parameter import cost_function_parameter
from private_dense_MPC.system_model import system_model
from private_dense_MPC.utils import cont_mat
from private_dense_MPC.utils import obv_mat


from private_dense_MPC.constants import Ts, I, Nc, x0, d, Dis_time, prediction_horizon, rm



'''
The results for the MPC while Random affine transformation is used as 
privacy preserving mechanism.
'''



'''
Loading the system model from the system_model function.

'''
Ac, Bc, Cc, Dc = system_model()
print("A = {}, \n B = {}, \n C =  {}".format(Ac, Bc, Cc))

'''Discretized system using Scipy package using ZOH method'''

disc_sys = cont2discrete((Ac, Bc, Cc, Dc), Ts, method='zoh')

# We unpack the tuple here
(A, B, C, D) = disc_sys[0:4]
print("A = {}, \n B = {}, \n C =  {} \n D = {}".format(A, B, C, D))




'''Check whether the system is controllable and observable'''

print("Controllability rank is {}".format(np.linalg.matrix_rank(cont_mat(A,B))))

print("Observability rank is {}".format(np.linalg.matrix_rank(obv_mat(A,C))))

# Here we want to form the matrices for the LQ in compact form


# The cost function parameters
n = np.size(A, 0)
m = np.size(B, 1)
p = np.size(C, 0)


P, Q, R, umax, umin, ymax, ymin = cost_function_parameter(n, m, p)





# Here we compute the Compact matrices given in the paper; note we need to 
   # compute them only once

state_data = {}
input_data = {}

F_tilde_data = {}
Z_tilde_data = {}







# properties for the random transformation
time_varying_trans = False



for N in prediction_horizon:

    H, F, Y, G, W, O, calS, calR, calQ, S, calT = QP.QP_matrices(N, A, B, C,
                                                                P, Q, R, umin, umax, ymin, ymax)
    print("singular values for F matrix to check if it is full column rank: {}".format(LA.svdvals(F)))
    
    state_data['X_N_{}'.format(N)], input_data['U_N_{}'.format(N)],F_tilde_data[
        'F_N_{}'.format(N)], Z_tilde_data['Z_N_{}'.format(N)] = apply_mpc_to_sys(time_varying_trans,
                                                                                 rm, H, F, G, W, O, x0, A, B,
                                                                                 m, p, I, N, Nc, Dis_time, d)

# Save data
# name = my_data
np.savez("./data/my_data", F_tilde_data=F_tilde_data, Z_tilde_data=Z_tilde_data,
          A=A, B=B, state_data=state_data, input_data=input_data,
            prediction_horizon=prediction_horizon)





# Here we plot the system response that we have received from the previous block

Time_steps = np.arange((Nc*I) + 1) * Ts


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 11})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


fig1 = plt.figure()
ax = fig1.subplots(2,1)

a = list(state_data.keys())
print(a)
# plot here
ax[0].plot(Time_steps, 0.5 * state_data[a[0]][0,:], Time_steps, 0.5 * state_data[a[1]][0,:],
            Time_steps, 0.5 * state_data[a[2]][0,:])
ax[1].plot(Time_steps, 0.5 * state_data[a[0]][1,:], Time_steps, 0.5 * state_data[a[1]][1,:],
            Time_steps, 0.5 * state_data[a[2]][1,:])






# add information here
ax[0].grid(linestyle='dotted', linewidth=0.5)
ax[1].grid(linestyle='dotted', linewidth=0.5)
ax[0].set_ylabel(r"$y_1$(cm)")
ax[1].set_ylabel(r"$y_2$(cm)")
ax[1].set_xlabel("Time(s)")
ax[0].legend(('$N={}$'.format(prediction_horizon[0]),"$N={}$".format(prediction_horizon[1]),
              '$N={}$'.format(prediction_horizon[2])))




plt.tight_layout()




fig1.savefig("outputs.pdf", transparent=True, dpi=300)

# plt.show()



from sklearn.metrics import mean_squared_error as mse

K = np.size(state_data[a[0]][0:2,:], 1)
y_true = np.zeros((p, K))

e5 = mse(y_true, state_data[a[0]][0:2,:], squared=False)
e20 = mse(y_true, state_data[a[1]][0:2,:], squared=False)
e50 = mse(y_true, state_data[a[2]][0:2,:], squared=False)

print('e5 {}, e20 {} and e50 {}'.format(e5, e20, e50))




Time_steps = np.arange((Nc*I) ) * Ts
b = list(input_data.keys())
print(b)


fig2 = plt.figure()
ax = fig2.subplots(2,1)

# plot here
ax[0].plot(Time_steps, input_data[b[0]][0,:], Time_steps, input_data[b[1]][0,:],
            Time_steps, input_data[b[2]][0,:])
ax[1].plot(Time_steps, input_data[b[0]][1,:], Time_steps, input_data[b[1]][1,:],
            Time_steps, input_data[b[2]][1,:])



mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')


# add information here
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel(r"$u_1$(v)")
ax[1].set_ylabel(r"$u_2$(v)")
ax[1].set_xlabel("Time(s)")
ax[0].legend(('$N={}$'.format(prediction_horizon[0]),"$N={}$".format(prediction_horizon[1]),
              '$N={}$'.format(prediction_horizon[2])))

# fig.suptitle(r"\TeX\ is Number "
#           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#           fontsize=16, color='gray')



plt.tight_layout()


# save the figure here
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.style.use('seaborn-paper')

fig2.savefig("inputs.eps", transparent=True)

plt.show()


