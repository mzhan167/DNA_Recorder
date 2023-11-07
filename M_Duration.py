'''
author: mzhan167

plot_duration test

'''

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt
from modules_main import Module_Dura

Module_Dura = Module_Dura()

#initial_conditions

I = np.array([300], dtype=np.dtype(float))
R = np.array([100], dtype=np.dtype(float))
blocker = np.array([0], dtype=np.dtype(float))

#                short long
SW_vec = np.array([25,  0], dtype=np.dtype(float))
S_vec  = np.array([ 0,  0], dtype=np.dtype(float))
IW_vec = np.array([ 0,  0], dtype=np.dtype(float))
CH_vec = np.array([11, 20], dtype=np.dtype(float))
H_vec  = np.array([ 0,  0], dtype=np.dtype(float))
SC_vec = np.array([ 0,  0], dtype=np.dtype(float))
RH_vec = np.array([ 0,  0], dtype=np.dtype(float))
PH_vec = np.array([ 0,  0], dtype=np.dtype(float))
P_vec  = np.array([ 0,  0], dtype=np.dtype(float))

# time-delay   short long
CG   = np.array([0,  20], dtype=np.dtype(float))
G    = np.array([0,   0], dtype=np.dtype(float))
IC   = np.array([0,   0], dtype=np.dtype(float))
T    = np.array([0,  25], dtype=np.dtype(float))
TG   = np.array([0,   0], dtype=np.dtype(float)) 
eTG  = np.array([0,   0], dtype=np.dtype(float))
eT   = np.array([0,   0], dtype=np.dtype(float)) 
TD   = np.array([0, 150], dtype=np.dtype(float))
TDG  = np.array([0,   0], dtype=np.dtype(float))
eTDG = np.array([0,   0], dtype=np.dtype(float))
eTD  = np.array([0,   0], dtype=np.dtype(float))
D    = np.array([0,   0], dtype=np.dtype(float))
preSW = np.array([0, 50], dtype=np.dtype(float))
Dpre = np.array([0,   0], dtype=np.dtype(float))

C0 = np.concatenate([SW_vec, S_vec, IW_vec, CH_vec, H_vec, SC_vec, RH_vec, PH_vec, P_vec, CG, G, IC, T, TG, eTG, eT, TD, TDG, eTDG, eTD, D, preSW, Dpre, R, blocker, I]) # "concatenate" -> combined lists ##

t_00 = np.array([30, 60*5, 60*10, 60*15, 60*20, 60*22, 60*25, 60*28, 60*30, 60*32, 60*35, 60*40, 60*42, 60*45, 60*48, 60*50, 60*52, 60*55, 60*60, 60*70, 60*80, 60*90, 60*120, 60*150]) #critical: 60*30 sec

t_x = []
p_short = []
p_long = []
p_short_save = []
p_long_save = []

for x in t_00:
    t_step = x
    
    t_1 = [0, t_step] # timescale 7hr = 7*3600s
    t_eval = np.linspace(t_1[0], t_1[1], 11) # 51 for collecting dots number
    t_2 = [t_step, 3600*6]
    t_eval_2 = np.linspace(t_2[0], t_2[1], 51) # 51 for collecting dots number
    
    Module_Dura.k_tg_f = 5e-4
    Module_Dura.k_tdg_f = 5e-4
    Module_Dura.k_etg_f = 5e-4
    Module_Dura.k_etdg_f = 5e-4
    Module_Dura.k_delay = 1.4e-3 # go through the critical point
    
    Module_Dura.k_tg_r = 2e-2 ### 2e-2 to 5e-3 NSD
    Module_Dura.k_tdg_r = 2e-2 ###
    Module_Dura.k_etg_r = 2e-2 ###
    Module_Dura.k_etdg_r = 2e-2 ###
    
    Module_Dura.k_d_release = 3e-3    
    Module_Dura.k_t_and_delay_bind = 8e-4 
    Module_Dura.k_rh_f = 3e-3
    Module_Dura.k_ph_f = 3e-3
    Module_Dura.k_per  = 1.6e-3
    Module_Dura.k_rh_r = 6e-2
    Module_Dura.k_ph_r = 6e-2
    Module_Dura.k_DtopreSW = 0.8e-5
    
    C = Module_Dura.reaction(t_1, C0, t_eval)
    idx=np.where(C['t'] == t_step) #floor 
    C['y'][-2,idx] += 400    #C[-1]=C[-1]+100
    C_2 = Module_Dura.reaction(t_2, C.y[:,-1], t_eval_2) 
    
    
    
    Module_Dura.k_tg_f = 3e-3
    Module_Dura.k_tdg_f = 3e-3
    Module_Dura.k_etg_f = 3e-3
    Module_Dura.k_etdg_f = 3e-3
    Module_Dura.k_delay = 1.8e-3 #go through the critical point
    Module_Dura.k_tg_r = 2e-2 ###
    Module_Dura.k_tdg_r = 2e-2 ###
    Module_Dura.k_etg_r = 2e-2 ###
    Module_Dura.k_etdg_r = 2e-2 ###
    Module_Dura.k_d_release = 1e-3
    Module_Dura.k_t_and_delay_bind = 3e-4 
    Module_Dura.k_rh_f = 1.1e-3
    Module_Dura.k_ph_f = 1.1e-3
    Module_Dura.k_per  = 1.8e-3
    Module_Dura.k_rh_r = 1e-1
    Module_Dura.k_ph_r = 1e-1
    Module_Dura.k_DtopreSW = 3e-5
    C_save = Module_Dura.reaction(t_1, C0, t_eval)
    idx=np.where(C['t'] == t_step) #floor 
    C_save['y'][-2,idx] += 1000    #C[-1]=C[-1]+100
    C_save_2 = Module_Dura.reaction(t_2, C_save.y[:,-1], t_eval_2) 
    
    
    
    t_x.append(x)
    p_short.append(C_2.y[16][-1]) 
    p_long.append(C_2.y[17][-1]) 
    
    p_short_save.append(C_save_2.y[16][-1]) 
    p_long_save.append(C_save_2.y[17][-1]) 

# experiment data

plt.scatter(0*60, 79.12, c='red')
plt.scatter(10*60, 89.91, c='red')
plt.scatter(20*60, 84.93, c='red')
plt.scatter(30*60, 86.22, c='red')
plt.scatter(60*60, 58.43, c='red')
plt.scatter(90*60, 52.71, c='red')
plt.scatter(120*60, 42.67, c='red')
plt.scatter(150*60, 45.81, c='red')

plt.scatter(0*60, 0, c='blue')
plt.scatter(10*60, 0, c='blue')
plt.scatter(20*60, 0, c='blue')
plt.scatter(30*60, 4.38, c='blue')
plt.scatter(60*60, 27.84, c='blue')
plt.scatter(90*60, 39.67, c='blue')
plt.scatter(120*60, 49.66, c='blue')
plt.scatter(150*60, 47.43, c='blue')
    

# plot information
plt.plot(t_x, p_short, 'r--', linewidth=2.0)
plt.plot(t_x, p_long, 'b--', linewidth=2.0)
plt.ylim([-2,100])
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.legend(['product_short','product_long'])
#plt.savefig("duration[delayS_preSW 5e-5].pdf")

