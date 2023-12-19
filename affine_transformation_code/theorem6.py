import numpy as np
import scipy.linalg as LA


import matplotlib.pyplot as plt
import matplotlib as mpl


from estimation_for_transfromed_system import theorem_6
from estimation_for_transfromed_system import data_analyses
from estimation_for_transfromed_system import eigen_error

from constants import Dis_time, Nc, m_eff


import os
os.system('cls' if os.name == 'nt' else 'clear')


# Load data from all the experiments

loaded_data = np.load("my_data.npz", allow_pickle=True)


F_tilde_data = loaded_data["F_tilde_data"]
Z_tilde_data = loaded_data["Z_tilde_data"]
A = loaded_data["A"]
B = loaded_data["B"]
pr_hor = loaded_data["prediction_horizon"]


n_rank = np.size(A,0)  # The dimension of state
# Data Analyses





F_name = list(F_tilde_data.item().keys())
print(F_name)


Z_name = list(Z_tilde_data.item().keys())
print(Z_name)








for fn, zn in zip(F_name, Z_name):
    F = F_tilde_data.item().get(fn)[:,0:Dis_time]
    Z = Z_tilde_data.item().get(zn)[:,0:Dis_time]

    # Correctness check: here we check the dimension of the received data

    print("The dimension of F data is:{}".format(F.shape))

    print("The dimension of Z data is:{}".format(Z.shape))

    rankDeltaF, rankDeltaZ, rankDeltaFDeltaZ, svd_DeltaF, svd_DeltaZ, svd_DeltaFDeltaZ, DeltaF, DeltaZ = data_analyses(F,Z)

    print("rank for DeltaF is :{}".format(rankDeltaF), '\n', "rank for DeltaZ is :{}".format(rankDeltaZ), '\n',
           "rank for DeltaF DeltaZ is :{}".format(rankDeltaFDeltaZ))
    print("Singular Values for DeltaF are:  {}".format(svd_DeltaF), '\n',
        "Singular Values for DeltaZ is:  {}".format(svd_DeltaZ), '\n',
            "Singular Values for DeltaF DeltaZ are:  {}".format(svd_DeltaFDeltaZ))
    print("DeltaF is :{}".format(DeltaF[:,5]), '\n',
          " DeltaZ is :{}".format(DeltaZ[:,5]))
    print("===============================================================================")



# Theorem SIX Simulation





Estimated_eigenvalues = {}
for fn, zn in zip(F_name, Z_name):
    F = F_tilde_data.item().get(fn)[:,0:Dis_time]
    Z = Z_tilde_data.item().get(zn)[:,0:Dis_time]
    

    # Correctness check: here we check the dimension of the received data

    print("The dimension of F data is:{}".format(F.shape))

    print("The dimension of Z data is:{}".format(Z.shape))


    value, status, Theta, E1, E0, error_in_decomposition,DeltaZ  = theorem_6(F, Z, n_rank, m_eff)

    print(value)
    print(status)
    # print(E0 @ Theta)
    print("Eigenvalues for E0 * Theta{}".format(LA.eig(E0 @ Theta, left=False, right=False)))

    # print(DeltaZ[:, 0:-1] @ Theta)
    error_signals = LA.norm( (E0[:, 1:Dis_time-5] - E1[:, 0:Dis_time-6]), ord='fro')
    print("The error between E0 and E1 is{}".format(error_signals))
    print("Error in decomposition.{}".format(error_in_decomposition))

    Ahat = E1 @ Theta
    # print("This is Ahat^Nc:{}".format(Ahat))

    print("here we have the results for eigenvalues")
    eig_A = LA.eig(A, left=False, right=False).reshape(n_rank, 1)



    est_eigen_Ahat = LA.eig(Ahat, left=False, right=False).reshape(n_rank, 1)
    Estimated_eigenvalues[fn] = est_eigen_Ahat
    print("EigenValues of true A^Nc matrix {}".format(np.power(eig_A, Nc)))
    print("absolute value of eigenvalues{}".format(abs(est_eigen_Ahat)))
    print("EigenValues of Ahat matrix {}".format(est_eigen_Ahat))


    error = eigen_error(eig_A, est_eigen_Ahat,n_rank) # computing the error between the estimations
    print("The  error between the estimated and true eigen values is: ")
    print(error)



plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 11})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



mpl.rc('font', family='serif')


fig1 = plt.figure()

ax = fig1.subplots(1,1)



# plot here
circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
ax.add_patch(circ)

ax.scatter(np.power(eig_A, Nc).real, np.power(eig_A, 2).imag , color='black', marker = 'X')

ax.scatter(Estimated_eigenvalues[F_name[0]].real, Estimated_eigenvalues[F_name[0]].imag , color='tab:blue', marker = '^')

ax.scatter(Estimated_eigenvalues[F_name[1]].real, Estimated_eigenvalues[F_name[1]].imag , color='red', marker = 's')

ax.scatter(Estimated_eigenvalues[F_name[2]].real, Estimated_eigenvalues[F_name[2]].imag , color='green', marker = 'o')




ax.legend(('Unit Circle', r'$\lambda(A)$' ,'' 'N={}'.format(pr_hor[0]),'N={}'.format(pr_hor[1]),
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