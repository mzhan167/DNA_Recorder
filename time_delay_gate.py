'''
author: mzhan167

plot_timedelay test

'''


import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt
from modules_main import Module_timedelay

Module_timedelay = Module_timedelay()

#initial conditions

C0 = np.array([200, 20, 0, 0, 50, 0, 0, 0, 50, 0, 0, 0, 0], dtype=np.dtype(float))

t = [0, 3600*8.5] # timescale 7hr = 7*3600s
t_eval = np.linspace(t[0], t[1], 51) # 51 for collecting dots number
T00 = np.array([0, 50, 100, 150, 200, 250, 300])
ax = plt.gca() # for the same color in a loop

for i in T00:
    C0[4] = i
    
    #Module_timedelay.k_delay = 1.3e-3 
    C = Module_timedelay.reaction(t, C0, t_eval)
    
    
    Module_timedelay.k_tg_f = 5e-4
    Module_timedelay.k_tdg_f = 5e-4
    Module_timedelay.k_etg_f = 5e-4
    Module_timedelay.k_etdg_f = 5e-4
    Module_timedelay.k_delay = 1.4e-3
    Module_timedelay.k_tg_r = 2e-2 
    Module_timedelay.k_tdg_r = 2e-2 
    Module_timedelay.k_etg_r = 2e-2 
    Module_timedelay.k_etdg_r = 2e-2 
    Module_timedelay.k_d_release = 3e-3
    Module_timedelay.k_t_and_delay_bind = 8e-4 

    C_increase = Module_timedelay.reaction(t, C0, t_eval)
    Module_timedelay.k_d_release = 5e-3
    Module_timedelay.k_delay = 1.4e-3
    Module_timedelay.k_tg_r = 4e-2
    Module_timedelay.k_tdg_r = 4e-2
    Module_timedelay.k_etg_r = 4e-2
    Module_timedelay.k_etdg_r = 4e-2
    C_decrease = Module_timedelay.reaction(t, C0, t_eval)
    
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(C_increase.t, (C_increase.y[12]/50)*11+2, '--', linewidth=2.0, color=color)

# plot information

plt.ylim([0,16])
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.legend(['0.0x Timer', '1.0x Timer', '2.0x Timer', '3.0x Timer', '4.0x Timer', '5.0x Timer', '6.0x Timer',])
plt.title("[Timer] = 1x~6x, 1x = 50 nM")
#plt.savefig("in_de_k_delay_optimized 50 timerdelay_para_dura"+str(i)+" nM timer.pdf")