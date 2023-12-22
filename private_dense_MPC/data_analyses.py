
import os
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl

from private_dense_MPC.estimation_for_transformed_system import data_analyses
from private_dense_MPC.constants import Dis_time, Nc, m_eff




'''
In this module we analyse the data that Cloud has obtained 
during its the search for the optimizer in the Secure MPC. 
What we check is the rank of the transformed data and to see that
we can use Theorem 6 for identifying the parameters.
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




