import os
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl


from private_dense_MPC.estimation_for_transformed_system import theorem_6
from private_dense_MPC.estimation_for_transformed_system import eigen_error
from private_dense_MPC.constants import Dis_time, Nc, m_eff




'''
The file includes the result in Theorem 6 of the paper. 
For running this file, we need to have the data before hand by running the Plant_control.py file. 
The goal is to estimate the eigenvalues of Ahat matrix. 

'''





# Clear the terminal for showing the results
os.system('cls' if os.name == 'nt' else 'clear')


# Load data from all the experiments
loaded_data = np.load("./data/my_data.npz", allow_pickle=True)


F_tilde_data = loaded_data["F_tilde_data"]
Z_tilde_data = loaded_data["Z_tilde_data"]
A = loaded_data["A"]
B = loaded_data["B"]
pr_hor = loaded_data["prediction_horizon"]


n_rank = np.size(A,0)  # The dimension of state




F_name = list(F_tilde_data.item().keys())
print(F_name)


Z_name = list(Z_tilde_data.item().keys())
print(Z_name)









Estimated_eigenvalues = {}
for fn, zn in zip(F_name, Z_name):
    F = F_tilde_data.item().get(fn)[:,0:Dis_time]
    Z = Z_tilde_data.item().get(zn)[:,0:Dis_time]
    

    # Correctness check: here we check the dimension of the received data

    print("The dimension of F data is: {}".format(F.shape))

    print("The dimension of Z data is: {}".format(Z.shape))


    value, status, Theta, E1, E0, error_in_decomposition,DeltaZ  = theorem_6(
                                                                F, Z, n_rank, m_eff)

    print("The value for the optimization in Theorem 6 is: {}".format(value))
    print("The status for the optimization in terms of feasibility is: {}".format(status))
    # print(E0 @ Theta)
    print("Eigenvalues for E0 * Theta{}".format(LA.eig(E0 @ Theta, left=False, right=False)))

    # print(DeltaZ[:, 0:-1] @ Theta)
    error_signals = LA.norm( (E0[:, 1:Dis_time-5] - E1[:, 0:Dis_time-6]), ord='fro')
    print("The error between E0 and E1 is{}".format(error_signals))
    print("Error in decomposition.{}".format(error_in_decomposition))

    Ahat = E1 @ Theta
    # print("This is Ahat^Nc:{}".format(Ahat))

    print("Here we have the results for eigenvalues")
    eig_A = LA.eig(A, left=False, right=False).reshape(n_rank, 1)



    est_eigen_Ahat = LA.eig(Ahat, left=False, right=False).reshape(n_rank, 1)
    Estimated_eigenvalues[fn] = est_eigen_Ahat
    print("EigenValues of true A^Nc matrix are: {}".format(np.power(eig_A, Nc)))
    print("Absolute value of eigenvalues of Ahat matrix is: {}".format(abs(est_eigen_Ahat)))
    print("EigenValues of Ahat matrix are: {}".format(est_eigen_Ahat))


    error = eigen_error(eig_A, est_eigen_Ahat,n_rank) # computing the error between the estimations
    print("The  error between the estimated and true eigen values is: {}".format(error))










plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 11})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('font', family='serif')


fig1 = plt.figure()

ax = fig1.subplots(1,1)



# plot here
circ = plt.Circle((0, 0), radius=1,
                   edgecolor='b', facecolor='None')
ax.add_patch(circ)

ax.scatter(np.power(eig_A, Nc).real, 
           np.power(eig_A, 2).imag , color='black', marker = 'X')

ax.scatter(Estimated_eigenvalues[F_name[0]].real,
            Estimated_eigenvalues[F_name[0]].imag , color='tab:blue', marker = '^')

ax.scatter(Estimated_eigenvalues[F_name[1]].real,
            Estimated_eigenvalues[F_name[1]].imag , color='red', marker = 's')

ax.scatter(Estimated_eigenvalues[F_name[2]].real,
            Estimated_eigenvalues[F_name[2]].imag , color='green', marker = 'o')




ax.legend(('Unit Circle', r'$\lambda(A)$' ,
           '' 'N={}'.format(pr_hor[0]),'N={}'.format(pr_hor[1]),
           'N={}'.format(pr_hor[2])),loc='center left')

ax.set_xlim(0.75, 1.05)
ax.set_ylim(-0.2, 0.2)


ax.set_xlabel(r"$Re(\hat{\lambda})$")
ax.set_ylabel(r"$Im(\hat{\lambda})$")

ax.grid(linestyle='dotted', linewidth=0.5)




plt.tight_layout()


# save the figure here

plt.show()

fig1.savefig("eigen_values_estimation.pdf", transparent=True, dpi=300)