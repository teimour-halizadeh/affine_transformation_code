import numpy as np



'''
All the constants that we used for simulating the QTP are given here. 
The parameters determine the operating point of the system, number of instances that 
we want to run the simulation, initial condition of the system, 
the range for the privacy preserving mechanism, ... 
'''



# Model Parameters
T1 = 63
T2 = 91
T3 = 39
T4 = 56
A1 = 28
A3 = 28 
A2 = 32
A4 = 32
kc = 0.5
k1 = 3.14
k2 = 3.29
gamma1 = 0.43
gamma2 = 0.34



Ts = 2.0  # Sampling period for the system

I = 500 # number of instances that we want to solve MPC problem




Nc = 1 # The control horizon (how many steps you want to apply
#                        the calculated optimizer to the system?)




x0 = 1 * np.array([
    [2],
    [-2],
    [-1],
    [2]]) # initial condition



d =  1*np.array([[3],
              [-3]]) # disturbance for the system

Dis_time = 100      # The moment that we want to have the disturbance
                    # for the system is selected here


prediction_horizon = [5, 20, 50] # prediction horizon


rm = 10**6  # The range for generating random numbers


n_rank = 4  # the number of states
m_eff = 7  # a parameter for cleaning the Z data


# The prediction horizon range for Theorem 4 in the paper
low_range_N = 10
up_range_N = 80







