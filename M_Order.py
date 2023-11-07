'''
author: mzhan167

plot_order test

'''

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt
from modules_main import Module_Order

Module_Order = Module_Order()

Module_Order.k_bind = 3e-3
Module_Order.k_leak_A = 2e-5
Module_Order.k_leak_B = 1e-9

#initial_conditions

SW_A  = np.array([50], dtype=np.dtype(float))
S_A   = np.array([ 0], dtype=np.dtype(float))
IW_A  = np.array([ 0], dtype=np.dtype(float))
CH_A  = np.array([20], dtype=np.dtype(float))
H_A   = np.array([ 0], dtype=np.dtype(float))
SC_A  = np.array([ 0], dtype=np.dtype(float))
R1H_A = np.array([ 0], dtype=np.dtype(float))
P1H_A = np.array([ 0], dtype=np.dtype(float))
P1_A  = np.array([ 0], dtype=np.dtype(float))
R2H_A = np.array([ 0], dtype=np.dtype(float))
P2H_A = np.array([ 0], dtype=np.dtype(float))
P2_A  = np.array([ 0], dtype=np.dtype(float))

SW_B  = np.array([50], dtype=np.dtype(float))
S_B   = np.array([ 0], dtype=np.dtype(float))
IW_B  = np.array([ 0], dtype=np.dtype(float))
CH_B  = np.array([20], dtype=np.dtype(float))
H_B   = np.array([ 0], dtype=np.dtype(float))
SC_B  = np.array([ 0], dtype=np.dtype(float))
R1H_B = np.array([ 0], dtype=np.dtype(float))
P1H_B = np.array([ 0], dtype=np.dtype(float))
P1_B  = np.array([ 0], dtype=np.dtype(float))
R2H_B = np.array([ 0], dtype=np.dtype(float))
P2H_B = np.array([ 0], dtype=np.dtype(float))
P2_B  = np.array([ 0], dtype=np.dtype(float))

I_A   = np.array([ 0], dtype=np.dtype(float))
I_B   = np.array([ 0], dtype=np.dtype(float))

R1 = np.array([100], dtype=np.dtype(float))
R2 = np.array([100], dtype=np.dtype(float))

C_A = np.array([10], dtype=np.dtype(float))
C_B = np.array([10], dtype=np.dtype(float))


C0 = np.concatenate([SW_A, S_A, IW_A, CH_A, H_A, SC_A, 
                     R1H_A, P1H_A, P1_A, 
                     R2H_A, P2H_A, P2_A,
                     SW_B, S_B, IW_B, CH_B, H_B, SC_B, 
                     R1H_B, P1H_B, P1_B, 
                     R2H_B, P2H_B, P2_B,
                     I_A, I_B, 
                     R1, R2, C_A, C_B]) # "concatenate" -> combined lists
C0_1 = np.concatenate([SW_A, S_A, IW_A, CH_A, H_A, SC_A, 
                     R1H_A, P1H_A, P1_A, 
                     R2H_A, P2H_A, P2_A,
                     SW_B, S_B, IW_B, CH_B, H_B, SC_B, 
                     R1H_B, P1H_B, P1_B, 
                     R2H_B, P2H_B, P2_B,
                     I_A, I_B, 
                     R1, R2, C_A, C_B]) # "concatenate" -> combined lists


t_x = []
t_xx = []
p_A_AB = []
p_B_AB = []
p_A_BA = []
p_B_BA = []

t = [60*0, 60*65]
T00 = np.linspace(t[0], t[1], 21)

for t00 in T00:
    
    t_0 = [0, 1e-20]
    t_eval_0 = np.linspace(t_0[0], t_0[1], 7)
    
    C_0 = Module_Order.reaction(t_0, C0, t_eval_0)
    
    t_step = t00
    t_1 = [1e-20, t_step] # timescale 7hr = 7*3600s
    t_eval_1 = np.linspace(t_1[0], t_1[1], 51)
    t_2 = [t_step, 3600*6]
    t_eval_2 = np.linspace(t_2[0], t_2[1], 51)
    
    idx_1=np.where(C_0.t == 1e-20)
    C_0.y[-6,idx_1] += 50  # A
    C_0.y[-5,idx_1] += 0    # B
    C_1 = Module_Order.reaction(t_1, C_0.y[:,-1], t_eval_1)
    
    idx_2=np.where(C_1.t == t_step) #floor 
    C_1.y[-6,idx_2] += 0     # A
    C_1.y[-5,idx_2] += 50    # B
    C_2 = Module_Order.reaction(t_2, C_1.y[:,-1], t_eval_2)

    t_x.append(t00)
    p_A_AB.append(C_2.y[8][-1])
    p_B_AB.append(C_2.y[20][-1])
    
    
for x00 in T00:
    
    t_0_1 = [0, 1e-20]
    t_eval_0_1 = np.linspace(t_0_1[0], t_0_1[1], 7)
    
    C_0_1 = Module_Order.reaction(t_0_1, C0_1, t_eval_0_1)
    
    t_step_1 = x00
    t_1_1 = [1e-20, t_step_1] # timescale 7hr = 7*3600s
    t_eval_1_1 = np.linspace(t_1_1[0], t_1_1[1], 51)
    t_2_1 = [t_step_1, 3600*6]
    t_eval_2_1 = np.linspace(t_2_1[0], t_2_1[1], 51)
    
    idx_1_1=np.where(C_0_1.t == 1e-20)
    C_0_1.y[-6,idx_1_1] += 0      # A
    C_0_1.y[-5,idx_1_1] += 50    # B
    C_1_1 = Module_Order.reaction(t_1_1, C_0_1.y[:,-1], t_eval_1_1)
    
    idx_2_1=np.where(C_1_1.t == t_step_1) #floor 
    C_1_1.y[-6,idx_2_1] += 50  # A
    C_1_1.y[-5,idx_2_1] += 0    # B
    C_2_1 = Module_Order.reaction(t_2_1, C_1_1.y[:,-1], t_eval_2_1)

    t_xx.append(-x00)
    p_A_BA.append(C_2_1.y[8][-1])
    p_B_BA.append(C_2_1.y[20][-1])

# plot information
plt.plot(t_x, p_A_AB, 'r-', linewidth=2.0)
plt.plot(t_x, p_B_AB, 'b-', linewidth=2.0)
plt.plot(t_xx, p_A_BA, 'r-', linewidth=2.0)
plt.plot(t_xx, p_B_BA, 'b-', linewidth=2.0)
#plt.title("concentration(A-B)=500-50 nM")
plt.ylim([0, 50])
plt.xlim([-70*60, 70*60])
plt.xlabel('time(A-B)')
plt.ylabel('Yield of product')
plt.legend(['p1_A', 'p1_B'])

# B-A_A
plt.scatter(-60*60, 18.02, c='red')
plt.scatter(-50*60, 18.63, c='red')
plt.scatter(-40*60, 19.78, c='red')
plt.scatter(-30*60, 19.89, c='red')
plt.scatter(-20*60, 15.10, c='red')
plt.scatter(-10*60, 24.32, c='red')
plt.scatter(0*60, 30.28, c='red')
# A-B_A
plt.scatter(0*60, 25.97, c='red')
plt.scatter(10*60, 25.63, c='red')
plt.scatter(20*60, 28.42, c='red')
plt.scatter(30*60, 28.98, c='red')
plt.scatter(40*60, 28.12, c='red')
plt.scatter(50*60, 30.89, c='red')
plt.scatter(60*60, 29.56, c='red')

# B-A_B
plt.scatter(-60*60, 15.28, c='blue')
plt.scatter(-50*60, 15.17, c='blue')
plt.scatter(-40*60, 14.71, c='blue')
plt.scatter(-30*60, 13.81, c='blue')
plt.scatter(-20*60, 14.64, c='blue')
plt.scatter(-10*60, 12.61, c='blue')
plt.scatter(0*60, 8.34, c='blue')
# A-B_B
plt.scatter(0*60, 6.31, c='blue')
plt.scatter(10*60, 5.57, c='blue')
plt.scatter(20*60, 7.88, c='blue')
plt.scatter(30*60, 9.37, c='blue')
plt.scatter(40*60, 8.82, c='blue')
plt.scatter(50*60, 7.58, c='blue')
plt.scatter(60*60, 7.41, c='blue')


#plt.savefig("order_1.pdf")