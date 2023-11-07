'''
author: mzhan167

plot_concentration test

'''

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt
from modules_main import Module_Conc

Module_Conc = Module_Conc()

#initial_conditions

I = np.array([100])
R = np.array([100])

#                 low  high short long
SW_vec = np.array([30,  50,  25,   0], dtype=np.dtype(float))
S_vec  = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
IW_vec = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
CH_vec = np.array([20,  20,   0,   0], dtype=np.dtype(float))
H_vec  = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
SC_vec = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
RH_vec = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
PH_vec = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))
P_vec  = np.array([ 0,   0,   0,   0], dtype=np.dtype(float))

C0 = np.concatenate([SW_vec, S_vec, IW_vec, CH_vec, H_vec, SC_vec, RH_vec, PH_vec, P_vec, I, R]) # "concatenate" -> combined lists

t = [0,3600*8] # timescale
t_eval = np.linspace(t[0], t[1], 51) # 51 for collecting dots number


# function

conc = [0.1,500]
C00 = np.linspace(conc[0], conc[1], 51)

c_i = []
p_low = []
p_high = []
p_low_1 = []
p_high_1 = []

for i in range(len(C00)):
    if (C00[i] - 32) < 0: 
        C0[36] = 0  # k=2, C0[18]
    else: C0[36] = C00[i] - 32 #overdose of "bottom"
    
    Module_Conc.k_rh_f = 1e-4 
    Module_Conc.k_ph_f = 1e-4
    #Module_Conc.k_rh_r = 2.2e-3 # /s 
    #Module_Conc.k_ph_r = 2.2e-3 # /s 
    Module_Conc.k_rh_r = 3.5e-4
    Module_Conc.k_ph_r = 3.5e-4
    Module_Conc.k_per  = 2.2e-3 
    C = Module_Conc.reaction(t, C0, t_eval)    
    
    Module_Conc.k_rh_f = 1e-4
    Module_Conc.k_ph_f = 1e-4
    Module_Conc.k_rh_r = 2.2e-3 # /s 
    Module_Conc.k_ph_r = 2.2e-3 # /s 
    Module_Conc.k_per  = 2.2e-3 
    C_1 = Module_Conc.reaction(t, C0, t_eval)
    
    
    c_i.append(C00[i])
    p_low.append(C.y[32][-1])
    p_high.append(C.y[33][-1])
    
    p_low_1.append(C_1.y[32][-1])
    p_high_1.append(C_1.y[33][-1])

    
# plot information

plt.plot(c_i, p_low, 'r--', linewidth=2.0)
plt.plot(c_i, p_high, 'b--', linewidth=2.0)

plt.scatter(10, 0, c='red')
plt.scatter(25, 39.96, c='red')
plt.scatter(50, 55.43, c='red')
plt.scatter(100, 75, c='red')
plt.scatter(200, 48.38, c='red')
plt.scatter(300, 50.4, c='red')
plt.scatter(400, 48.15, c='red')
plt.scatter(500, 47.18, c='red')

plt.scatter(10, 0, c='blue')
plt.scatter(25, 0, c='blue')
plt.scatter(50, 0, c='blue')
plt.scatter(100, 0, c='blue')
plt.scatter(200, 26.63, c='blue')
plt.scatter(300, 24.53, c='blue')
plt.scatter(400, 26.85, c='blue')
plt.scatter(500, 27.75, c='blue')

plt.ylim([0,100])
plt.xlabel('[Input]')
plt.ylabel('Conc. of Product')
plt.legend(['Product_low', 'Product_high'])
#plt.savefig("conc[0.1~500]r_3.5e-4.pdf")