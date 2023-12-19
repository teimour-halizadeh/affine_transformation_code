import numpy as np
import cvxpy as cp
import scipy.linalg as LA

import itertools



from qpsolvers import solve_qp



'''
This is the core file for the theorems that we develop.
Based on the selected function, and the given data, it spits out 
the estimation!
'''


def data_analyses(F, Z):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]
    DeltaFDeltaZ = np.vstack((DeltaF[:,0:-1], DeltaZ[:,0:-1]))


    rankDeltaF = np.linalg.matrix_rank(DeltaF[:,0:-1])
    rankDeltaZ = np.linalg.matrix_rank(DeltaZ[:,0:-1])
    rankDeltaFDeltaZ = np.linalg.matrix_rank(DeltaFDeltaZ)

    svd_DeltaF = LA.svdvals(DeltaF)
    svd_DeltaZ = LA.svdvals(DeltaZ)
    svd_DeltaFDeltaZ = LA.svdvals(DeltaFDeltaZ)

    DeltaF_norm = LA.norm(DeltaF, ord='fro')
    DeltaZ_norm = LA.norm(DeltaZ, ord='fro')




    return rankDeltaF, rankDeltaZ, rankDeltaFDeltaZ, svd_DeltaF, svd_DeltaZ, svd_DeltaFDeltaZ, DeltaF, DeltaZ



def pure_least_sqaure(F, Z):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]


    DeltaFDeltaZ = np.vstack((DeltaF[:,0:-1], DeltaZ[:,0:-1]))


    estimation = (DeltaF[:,1:]) @ LA.pinv(DeltaFDeltaZ)
    Ahat = estimation[:, 0:l]
    Bhat = estimation[:, l:]


    error_signals = LA.norm( (DeltaF[:,1:] - estimation @ DeltaFDeltaZ), ord='fro')

    error_signals = np.allclose(estimation @ DeltaFDeltaZ, DeltaF[:,1:])

    # Dynamic mode decomposition idea: Use the SVD of X+ and project what you have 
    # obtained from the least square

    # U,s,Vh = LA.svd(DeltaF[:,1:])

    # Ahat_projected = U[:, 0:4].T @ Ahat @ U[:, 0:4]
    # Bhat_projected = U[:, 0:4].T @ Bhat



    return  Ahat, Bhat, error_signals







def projection_idea(F, Z, n_rank, m_rank):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]


    # Here we find E0 and E1 given in the code
    U,s,Vh = LA.svd(DeltaF[:,0:-1], lapack_driver='gesvd')
    DeltaFb = U[:,0:n_rank] 
    E0 = np.diag(s[0:n_rank]) @ Vh[0:n_rank,:]
    E1 = LA.pinv(DeltaFb) @ DeltaF[:,1:]



    #Clean the data for  for DeltaZ
    U,s,Vh = LA.svd(DeltaZ, lapack_driver='gesvd')
    DeltaZ_clean = U[:,0:m_rank] @ np.diag(s[0:m_rank]) @ Vh[0:m_rank,:]



    DeltaEDeltaZ = np.vstack((E0, DeltaZ_clean[:,0:-1]))


    estimation = E1 @ LA.pinv(DeltaEDeltaZ)


    Ahat = estimation[:, 0:n_rank]
    Bhat = estimation[:, n_rank:]



    return  Ahat, Bhat




def clean_data_least_square(F, Z, n_eff, m_eff):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]



    # clean the data of $DeltaF$
    U,s,Vh = LA.svd(DeltaF, lapack_driver='gesvd')
    DeltaF = U[:,0:n_eff] @ np.diag(s[0:n_eff]) @ Vh[0:n_eff,:]

    #Clean the data for  for DeltaZ
    U,s,Vh = LA.svd(DeltaZ, lapack_driver='gesvd')
    DeltaZ = U[:,0:m_eff] @ np.diag(s[0:m_eff]) @ Vh[0:m_eff,:]




    DeltaFDeltaZ = np.vstack((DeltaF[:,0:-1], DeltaZ[:,0:-1]))


    estimation = (DeltaF[:,1:]) @ LA.pinv(DeltaFDeltaZ)
    Ahat = estimation[:, 0:l]
    Bhat = estimation[:, l:]


    error_signals = LA.norm( (DeltaF[:,1:] - estimation @ DeltaFDeltaZ), ord='fro')

    error_signals = np.allclose(estimation @ DeltaFDeltaZ, DeltaF[:,1:])



    return  Ahat, Bhat, error_signals








def Rank_constraint(F, Z):
    '''
    For now, the idea is to compactly estimate $tilde{A} & tilde{B}$ and impose the rank condition
    on this value as a compact form.
    '''

    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]


    DeltaFDeltaZ = np.vstack((DeltaF[:,0:-1], DeltaZ[:,0:-1]))




    tildeAB = cp.Variable((l, 2*l))

    # Create two constraints.
    constraints = []



    # Form objective.
    # obj = cp.Minimize(cp.normNuc(tildeAB))
    obj = cp.Minimize(cp.normNuc(tildeAB) 
                      + 10**(3) * cp.norm(tildeAB @ DeltaFDeltaZ - DeltaF[:,1:], p='fro'))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    prob.solve()  

    tildeAB = tildeAB.value
    Ahat = tildeAB[:, 0:l]
    Bhat = tildeAB[:, l:]
    return prob.status, Ahat, Bhat




def theorem_6(F, Z, n_rank, m_eff):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]


    # Here we find E0 and E1 given in the code
    DeltaF0 = DeltaF[:,0:-1]
    U,s,Vh = LA.svd(DeltaF0, lapack_driver='gesvd')
    DeltaFb = U[:,0:n_rank] 
    E0 = np.diag(s[0:n_rank]) @ Vh[0:n_rank,:]

    error_in_decomposition = LA.norm(DeltaF[:,0:-1] - (DeltaFb@E0), ord='fro')


    # This is one way to compute E1, but it may introduce error
    # E1 = LA.pinv(DeltaFb) @ DeltaF[:,1:]



    hat_E1 = LA.pinv(DeltaFb) @ DeltaF[:,1:]

    # hat_E1 = write_deltaF1(DeltaFb, DeltaF[:,-1:])

    # Here we want to put the shifted version onf E0 in E1 and then put the final sample 
    # at the end. 
    E1 = np.hstack((E0[:,1:], hat_E1[:,-1:]))

    # E1 = np.hstack((E0[:,1:], hat_E1))







    # #Clean the data for  for DeltaZ
    # U,s,Vh = LA.svd(DeltaZ, lapack_driver='gesvd')
    # DeltaZ = U[:,0:m_eff] @ np.diag(s[0:m_eff]) @ Vh[0:m_eff,:]


  
# Here we have the optimization. 
  ##################################################################################################
    Theta = cp.Variable((I-2, n_rank))

    # Create two constraints.
    # constraints = [E0@Theta==np.eye(n_rank), DeltaZ[:, 0:-1]@Theta == np.zeros((l, n_rank))]
    constraints = [ E0@Theta == np.eye(n_rank)]


    # Form objective.
    # obj = cp.Minimize(cp.normNuc(tildeAB))
    # obj = cp.Minimize( cp.norm(1* DeltaZ[:, 0:-1]@Theta, 'fro'))

    obj = cp.Minimize( cp.norm(1* DeltaZ[:, 0:-1]@Theta, 2))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    prob.solve()  

    Theta = Theta.value

    #################################################################################################################
    

    


    return  prob.value, prob.status, Theta, E1, E0, error_in_decomposition, DeltaZ




def theorem_6_QPsolver(F, Z, n_rank, m_eff):
    l = np.size(F, 0)
    I = np.size(F, 1)


    DeltaF = F[:, 1:] - F[:, 0:-1]
    DeltaZ = Z[:, 1:] - Z[:, 0:-1]


    # Here we find E0 and E1 given in the code
    DeltaF0 = DeltaF[:,0:-1]
    U,s,Vh = LA.svd(DeltaF0, lapack_driver='gesvd')
    DeltaFb = U[:,0:n_rank] 
    E0 = np.diag(s[0:n_rank]) @ Vh[0:n_rank,:]

    error_in_decomposition = LA.norm(DeltaF[:,0:-1] - (DeltaFb@E0), ord='fro')


    # This is one way to compute E1, but it may introduce error
    # E1 = LA.pinv(DeltaFb) @ DeltaF[:,1:]



    hat_E1 = LA.pinv(DeltaFb) @ DeltaF[:,1:]

    # hat_E1 = write_deltaF1(DeltaFb, DeltaF[:,-1:])

    # Here we want to put the shifted version onf E0 in E1 and then put the final sample 
    # at the end. 
    E1 = np.hstack((E0[:,1:], hat_E1[:,-1:]))

    # E1 = np.hstack((E0[:,1:], hat_E1))







    # #Clean the data for  for DeltaZ
    # U,s,Vh = LA.svd(DeltaZ, lapack_driver='gesvd')
    # DeltaZ = U[:,0:m_eff] @ np.diag(s[0:m_eff]) @ Vh[0:m_eff,:]


  
# Here we have the optimization. 
  ##################################################################################################
    # Theta = cp.Variable((I-2, n_rank))

    # The constraint 
    Acon = LA.kron(np.eye(n_rank), E0)
    bcon = (np.eye(n_rank)).reshape((-1, 1), order="F")



    Phalf = LA.kron(np.eye(n_rank), DeltaZ[:, 0:-1])
    P = Phalf.T @ Phalf
    q = np.zeros(((I-2) * n_rank,1))
    G = np.eye((I-2) * n_rank)
    h = np.inf * np.ones(((I-2) * n_rank, 1))

    vec_theta = solve_qp(P, q, G, h, Acon, bcon, solver='osqp')
    # 'daqp', 'ecos', 'osqp', 'scs']
    # print(vec_theta)
    value = LA.norm(Phalf @ vec_theta)

    Theta = vec_theta.reshape((I-2,n_rank), order="F")

    #################################################################################################################
    

    


    return  Theta, E1, E0, value 



def eigen_error(eig_A, est_eigen_Ahat, n_rank):
  
  '''Here we compute the max_distance between the true and estimated eigen values'''
  b = np.tile(est_eigen_Ahat,(1,np.math.factorial(n_rank)))


  c = np.empty((n_rank, np.math.factorial(n_rank)))
  list1 = list(range(n_rank))
  perm = list(itertools.permutations(list1))
  for i, v in enumerate(perm):
    c[:, [i]] = eig_A[v,:]
 # all the values that this vector can have



  d = np.multiply(b-c, np.conj(b-c))   # Element wise multiplication with the conjugate
  e = LA.norm(eig_A)
  f = LA.norm(d,ord=-1)
  error = np.sqrt(f)/(e) # Here we take the minimum among all the columns using the L = -1 norm
  return error 