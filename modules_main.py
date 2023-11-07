'''
#########################################

all_simulation: 2023-03-12
@author: mzhan167

#########################################

For reaction system

'''

import numpy as np
import scipy.integrate as spi 
import matplotlib.pyplot as plt

# defind a reaction system:

#scale = 1e9 # M -> nM

       ###############################################################
       #                                                             #
       #                   Module_Concentration                      #
       #                                                             #
       ###############################################################

class Module_Conc:
    def __init__(self):
        
        self.k_inputtosignal_mat = np.array([3e-3,  5e-7, 3e-3,  3e-3]) # /nM /s
        self.k_signaltoinput_mat = np.array([3e-11, 5e-8, 3e-11, 3e-9])  # /nM /s  #, dtype=np.longdouble  precision=1e-18
        self.k_signaltohairpin_mat = 3e-3 # /nM /s
        self.k_hairpintosignal_mat = 3e-9 # /nM /s
        
        self.k_rh_f = 3e-3 # /nM /s
        self.k_rh_r = 2.2e-3 # /s
        self.k_per  = 5.8e-4 # /s
        self.k_ph_f = 3e-3 # /nM /s
        self.k_ph_r = 2.2e-3 # /s 

        self.filter_conc = np.array([1, 1, 0, 0])

    def rxn(self, t, x):
        k = 4 # amount of mian properties (e.g. 2 for low & high)
        conv_mat = np.ones(k) # Conversion matrix
        
        C_SW = x[    :   k] # Transducer [x[0], x[1]....x[k-1]]
        C_S  = x[  k : 2*k] # Signal
        C_IW = x[2*k : 3*k] # Waste
        C_CH = x[3*k : 4*k] # Covered-Hairpin
        C_H  = x[4*k : 5*k] # Hairpin
        C_SC = x[5*k : 6*k] # Waste
        C_RH = x[6*k : 7*k] # intermediated product
        C_PH = x[7*k : 8*k] # intermediated product
        C_P  = x[8*k : 9*k] # Product

        C_I  = x[9*k] # Input x[0]
        C_R  = x[9*k+1] # Primer
        
        # rate equation

        dSWdt = -self.k_inputtosignal_mat*(C_I*C_SW) +self.k_signaltoinput_mat*(C_S*C_IW)    
        dSdt  =  self.k_inputtosignal_mat*(C_I*C_SW) -self.k_signaltoinput_mat*(C_S*C_IW) -self.k_signaltohairpin_mat*C_S*C_CH +self.k_hairpintosignal_mat*C_H*C_SC
        dIWdt =  self.k_inputtosignal_mat*(C_I*C_SW) -self.k_signaltoinput_mat*(C_S*C_IW)  #Waste
        dCHdt = -self.k_signaltohairpin_mat*C_S*C_CH +self.k_hairpintosignal_mat*C_H*C_SC
        dHdt  =  self.k_signaltohairpin_mat*C_S*C_CH -self.k_hairpintosignal_mat*C_H*C_SC -self.k_rh_f*C_R*C_H +self.k_rh_r*C_RH -self.k_ph_f*C_P*C_H +self.k_ph_r*C_PH 
        dSCdt =  self.k_signaltohairpin_mat*C_S*C_CH -self.k_hairpintosignal_mat*C_H*C_SC  #Waste
        dRHdt =  self.k_rh_f*C_R*C_H -self.k_rh_r*C_RH -self.k_per*C_RH
        dPHdt =  self.k_ph_f*C_P*C_H -self.k_ph_r*C_PH +self.k_per*C_RH
        dPdt  = -self.k_ph_f*C_P*C_H +self.k_ph_r*C_PH
        #print(dPdt)
        dIdt  = -conv_mat@((self.k_inputtosignal_mat*(C_I*C_SW)).T) + conv_mat@((self.k_signaltoinput_mat*(C_S*C_IW)).T)
        dRdt  = -self.k_rh_f*C_R*conv_mat@((self.filter_conc*C_H).T) +self.k_rh_r*conv_mat@((self.filter_conc*C_RH).T) # k=2, no filter


        return np.concatenate([dSWdt, dSdt, dIWdt, dCHdt, dHdt, dSCdt, dRHdt, dPHdt, dPdt, np.array([dIdt]), np.array([dRdt])], dtype=np.dtype(float))

    
    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau') # max_step=1 precision=1e-18
    

    
    


       ###############################################################
       #                                                             #
       #                    Module_Time_Delay                        #
       #                                                             #
       ###############################################################
        
    
class Module_timedelay:
    def __init__(self):
        
        self.k_inputtogate_mat = 3e-3
        self.k_gatetoinput_mat = 5e-9
        
        self.k_tg_f = 3e-3
        self.k_tdg_f = 3e-3
        self.k_tg_r = 7.8e-2
        self.k_tdg_r = 7.8e-2
        self.k_delay = 4e-4
        self.k_etg_f = 3e-3
        self.k_etdg_f = 3e-3
        self.k_etg_r = 7.8e-2
        self.k_etdg_r = 7.8e-2
        
        self.k_d_release = 5e-4 
        self.k_d_recover = 1.4e-14

        self.k_t_and_delay_bind = 3e-3 # /nM /s
        self.k_t_and_delay_unbind = 6.6e-10

        
    def rxn(self, t, x):
        C_I    = x[0] # Input x[0]
        C_CG   = x[1] # Primer
        C_G    = x[2]
        C_IC   = x[3]
        C_T    = x[4]
        C_TG   = x[5]
        C_eTG  = x[6]
        C_eT   = x[7]
        C_TD   = x[8]
        C_TDG  = x[9]
        C_eTDG = x[10]
        C_eTD  = x[11]
        C_D    = x[12]

        # rate equation
        
        dIdt   = -self.k_inputtogate_mat*C_I*C_CG +self.k_gatetoinput_mat*C_G*C_IC
        dCGdt  = -self.k_inputtogate_mat*C_I*C_CG +self.k_gatetoinput_mat*C_G*C_IC
        dGdt   =  self.k_inputtogate_mat*C_I*C_CG -self.k_gatetoinput_mat*C_G*C_IC -self.k_tg_f*C_T*C_G +self.k_tg_r*C_TG -self.k_etg_f*C_eT*C_G +self.k_etg_r*C_eTG -self.k_tdg_f*C_TD*C_G +self.k_tdg_r*C_TDG -self.k_etdg_f*C_eTD*C_G +self.k_etdg_r*C_eTDG
        dICdt  =  self.k_inputtogate_mat*C_I*C_CG -self.k_gatetoinput_mat*C_G*C_IC
        dTdt   = -self.k_tg_f*C_T*C_G +self.k_tg_r*C_TG -self.k_t_and_delay_bind*C_T*C_D +self.k_t_and_delay_unbind*C_TD
        dTGdt  =  self.k_tg_f*C_T*C_G -self.k_tg_r*C_TG -self.k_delay*C_TG
        deTGdt =  self.k_etg_f*C_eT*C_G -self.k_etg_r*C_eTG +self.k_delay*C_TG
        deTdt  = -self.k_etg_f*C_eT*C_G +self.k_etg_r*C_eTG +self.k_d_release*C_eTD -self.k_d_recover*C_eT*C_D
        dTDdt  = -self.k_tdg_f*C_TD*C_G +self.k_tdg_r*C_TDG +self.k_t_and_delay_bind*C_T*C_D -self.k_t_and_delay_unbind*C_TD
        dTDGdt =  self.k_tdg_f*C_TD*C_G -self.k_tdg_r*C_TDG -self.k_delay*C_TDG
        deTDGdt = self.k_etdg_f*C_eTD*C_G -self.k_etdg_r*C_eTDG +self.k_delay*C_TDG
        deTDdt = -self.k_etdg_f*C_eTD*C_G +self.k_etdg_r*C_eTDG -self.k_d_release*C_eTD +self.k_d_recover*C_eT*C_D
        dDdt   =  self.k_d_release*C_eTD -self.k_d_recover*C_eT*C_D -self.k_t_and_delay_bind*C_T*C_D +self.k_t_and_delay_unbind*C_TD

        return np.array([dIdt, dCGdt, dGdt, dICdt, dTdt, dTGdt, deTGdt, deTdt, dTDdt, dTDGdt, deTDGdt, deTDdt, dDdt], dtype=np.dtype(float))

    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau')
    
    
    
    

    
       ###############################################################
       #                                                             #
       #                       Module_Duration                       #
       #                                                             #
       ###############################################################

        
class Module_Dura:
    def __init__(self):
        # I + CG <-> G + IC
        self.k_inputtogate_mat = 3e-3
        self.k_gatetoinput_mat = 5e-9

        # T + G <-> TG -> eTG <-> eT + G
        # TD + G <-> TDG -> eTDG <-> eTD + G
        self.k_tg_f = 3e-3
        self.k_tdg_f = 3e-3
        self.k_tg_r = 7.8e-2
        self.k_tdg_r = 7.8e-2
        self.k_delay = 4e-4 #2.5e-2
        self.k_etg_f = 3e-3
        self.k_etdg_f = 3e-3
        self.k_etg_r = 7.8e-2
        self.k_etdg_r = 7.8e-2
        
        # eTD <-> eT + D
        self.k_d_release = 5e-4 
        self.k_d_recover = 1.4e-14 # deltaG = timer_hairpin deltaG
        
        # T + D <-> TD
        self.k_t_and_delay_bind = 3e-3 # /nM /s
        self.k_t_and_delay_unbind = 6.6e-10
        
        # D + preSW <-> SW + Dpre  ## for SW_long
        self.k_DtopreSW = 5e-5
        self.k_preSWtoD = 5e-7
        
        # I + SW <-> S + IW
        # S + CH <-> H + SC
        self.k_inputtosignal_mat = np.array([5e-4, 3e-3]) # /nM /s
        self.k_signaltoinput_mat = np.array([5e-8, 5e-10])  # /nM /s
        self.k_signaltohairpin_mat = 3e-3 # /nM /s
        self.k_hairpintosignal_mat = 5e-9 # /nM /s
        
        # R + H <-> RH -> PH <-> P + H
        self.k_rh_f = 3e-3 # /nM /s
        self.k_rh_r = 1e-1 # /s
        self.k_per  = 5.8e-4 #1.7e-2 # /s 
        self.k_ph_f = 3e-3 # /nM /s
        self.k_ph_r = 1e-1 # /s
        
        # I + W <-> IW
        # S + W <-> SW   
        self.k_i_and_block_bind = 3e-3 # /nM /s
        self.k_i_and_block_unbind = 3.73e-20 # /nM /s
        self.k_s_and_block_bind = 3e-3
        self.k_s_and_block_unbind = np.array([7e-11, 3.4e-14])
        
        self.filter_d = np.array([0, 1])
        self.filter_b = np.array([1, 0])
        

    def rxn(self, t, x):
        
        k = 2 # amount of mian properties (e.g. 2 for low & high)
        conv_mat = np.ones(k) # Conversion matrix
        
        C_SW = x[    :   k] # Transducer [x[0], x[1]....x[k-1]]
        C_S  = x[  k : 2*k] # Signal
        C_IW = x[2*k : 3*k] # Waste
        C_CH = x[3*k : 4*k] # Covered-Hairpin
        C_H  = x[4*k : 5*k] # Hairpin
        C_SC = x[5*k : 6*k] # Waste
        C_RH = x[6*k : 7*k] # intermediated product
        C_PH = x[7*k : 8*k] # intermediated product
        C_P  = x[8*k : 9*k] # Product

        C_CG   = x[9*k  : 10*k]
        C_G    = x[10*k : 11*k]
        C_IC   = x[11*k : 12*k]
        C_T    = x[12*k : 13*k] # T and : [T, TD]
        C_TG   = x[13*k : 14*k]
        C_eTG  = x[14*k : 15*k]
        C_eT   = x[15*k : 16*k]

        C_TD   = x[16*k : 17*k]
        C_TDG  = x[17*k : 18*k]
        C_eTDG = x[18*k : 19*k]
        C_eTD  = x[19*k : 20*k]

        C_D     = x[20*k : 21*k]
        C_preSW = x[21*k : 22*k] # for long
        C_Dpre  = x[22*k : 23*k]


        C_R       = x[23*k] # Primer
        C_blocker = x[23*k+1]
        C_I       = x[23*k+2] # Input x[0]

        # rate equations

        dSWdt = -self.k_inputtosignal_mat*(C_I*C_SW) +self.k_signaltoinput_mat*C_S*C_IW +self.filter_d*self.k_DtopreSW*(C_preSW*C_D) -self.filter_d*self.k_preSWtoD*C_SW*C_Dpre  +self.k_s_and_block_bind*C_S*C_blocker -self.k_s_and_block_unbind*C_SW
        dSdt  =  self.k_inputtosignal_mat*(C_I*C_SW) -self.k_signaltoinput_mat*C_S*C_IW -self.k_signaltohairpin_mat*C_S*C_CH +self.k_hairpintosignal_mat*C_H*C_SC  -self.k_s_and_block_bind*C_S*C_blocker +self.k_s_and_block_unbind*C_SW
        dIWdt =  self.k_inputtosignal_mat*(C_I*C_SW) -self.k_signaltoinput_mat*C_S*C_IW +self.k_i_and_block_bind*C_I*C_blocker -self.k_i_and_block_unbind*C_IW
        
        dCHdt = -self.k_signaltohairpin_mat*C_S*C_CH +self.k_hairpintosignal_mat*C_H*C_SC
        dHdt  =  self.k_signaltohairpin_mat*C_S*C_CH -self.k_hairpintosignal_mat*C_H*C_SC -self.k_rh_f*C_R*C_H +self.k_rh_r*C_RH -self.k_ph_f*C_P*C_H +self.k_ph_r*C_PH 
        dSCdt =  self.k_signaltohairpin_mat*C_S*C_CH -self.k_hairpintosignal_mat*C_H*C_SC  #Waste
        dRHdt =  self.k_rh_f*C_R*C_H -self.k_rh_r*C_RH -self.k_per*C_RH
        dPHdt =  self.k_ph_f*C_P*C_H -self.k_ph_r*C_PH +self.k_per*C_RH
        dPdt  = -self.k_ph_f*C_P*C_H +self.k_ph_r*C_PH

        dCGdt  = -self.k_inputtogate_mat*C_I*C_CG +self.k_gatetoinput_mat*C_G*C_IC
        dGdt   =  self.k_inputtogate_mat*C_I*C_CG -self.k_gatetoinput_mat*C_G*C_IC -self.k_tg_f*C_T*C_G +self.k_tg_r*C_TG -self.k_etg_f*C_eT*C_G +self.k_etg_r*C_eTG -self.k_tdg_f*C_TD*C_G +self.k_tdg_r*C_TDG -self.k_etdg_f*C_eTD*C_G +self.k_etdg_r*C_eTDG
        dICdt  =  self.k_inputtogate_mat*C_I*C_CG -self.k_gatetoinput_mat*C_G*C_IC   # final_conc = initial_CG -> C0[1]
        dTdt   = -self.k_tg_f*C_T*C_G +self.k_tg_r*C_TG -self.k_t_and_delay_bind*C_T*C_D +self.k_t_and_delay_unbind*C_TD
        dTGdt  =  self.k_tg_f*C_T*C_G -self.k_tg_r*C_TG -self.k_delay*C_TG
        deTGdt =  self.k_etg_f*C_eT*C_G -self.k_etg_r*C_eTG +self.k_delay*C_TG
        deTdt  = -self.k_etg_f*C_eT*C_G +self.k_etg_r*C_eTG +self.k_d_release*C_eTD -self.k_d_recover*C_eT*C_D  

        dTDdt  = -self.k_tdg_f*C_TD*C_G +self.k_tdg_r*C_TDG +self.k_t_and_delay_bind*C_T*C_D -self.k_t_and_delay_unbind*C_TD
        dTDGdt =  self.k_tdg_f*C_TD*C_G -self.k_tdg_r*C_TDG -self.k_delay*C_TDG
        deTDGdt = self.k_etdg_f*C_eTD*C_G -self.k_etdg_r*C_eTDG +self.k_delay*C_TDG
        deTDdt = -self.k_etdg_f*C_eTD*C_G +self.k_etdg_r*C_eTDG -self.k_d_release*C_eTD +self.k_d_recover*C_eT*C_D

        dDdt   =  self.k_d_release*C_eTD -self.k_d_recover*C_eT*C_D -self.k_t_and_delay_bind*C_T*C_D +self.k_t_and_delay_unbind*C_TD -self.filter_d*self.k_DtopreSW*(C_preSW*C_D) +self.filter_d*self.k_preSWtoD*C_SW*C_Dpre  # NO MORE THAN C0[8] 
        dpreSWdt = -self.filter_d*self.k_DtopreSW*(C_preSW*C_D) + self.filter_d*self.k_preSWtoD*C_SW*C_Dpre
        dDpredt  =  self.filter_d*self.k_DtopreSW*(C_preSW*C_D) - self.filter_d*self.k_preSWtoD*C_SW*C_Dpre

        dRdt  = -self.k_rh_f*C_R*conv_mat@(C_H.T) +self.k_rh_r*conv_mat@(C_RH.T)  
        dblockerdt = -self.k_i_and_block_bind*C_I*C_blocker +conv_mat@((self.k_i_and_block_unbind*C_IW).T) -conv_mat@((self.k_s_and_block_bind*C_S*C_blocker).T) +conv_mat@((self.k_s_and_block_unbind*C_SW).T)

        dIdt  = -conv_mat@((self.k_inputtosignal_mat*(C_I*C_SW)).T) +self.k_signaltoinput_mat*C_S@(C_IW.T) -conv_mat@(self.filter_d*((self.k_inputtogate_mat*C_I*C_CG).T)) +conv_mat@(self.filter_d*((self.k_gatetoinput_mat*C_G*C_IC).T)) -self.k_i_and_block_bind*C_I*C_blocker +conv_mat@((self.k_i_and_block_unbind*C_IW).T)


        return np.concatenate([dSWdt, dSdt, dIWdt, dCHdt, dHdt, dSCdt, dRHdt, dPHdt, dPdt, 
                               dCGdt, dGdt, dICdt, dTdt, dTGdt, 
                               deTGdt, deTdt, dTDdt, dTDGdt, deTDGdt, 
                               deTDdt, dDdt, dpreSWdt, dDpredt, np.array([dRdt]), np.array([dblockerdt]), np.array([dIdt])], dtype=np.dtype(float))
   
    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau')






       ###############################################################
       #                                                             #
       #                        Module_Order                         #
       #                                                             #
       ###############################################################

class Module_Order:
    def __init__(self):
        
        self.k_inputtosignal_mat = 3e-3 # /nM /s
        self.k_signaltoinput_mat = 3e-11 # /nM /s  
        self.k_signaltohairpin_mat = 3e-3 # /nM /s
        self.k_hairpintosignal_mat = 3e-9 # /nM /s
        
        self.k_rh_f = 1e-4 # /nM /s
        self.k_rh_f_2 = 1.1e-3
        self.k_rh_r = 2.2e-3 # /s 
        self.k_rh_r_2 = 1e-1
        self.k_per  = 5.8e-4 # /s 
        self.k_ph_f = 1e-4 # /nM /s
        self.k_ph_f_2 = 1.1e-3
        self.k_ph_r = 2.2e-3 # /s 
        self.k_ph_r_2 = 1e-1
        
    def rxn(self, t, x):
        k = 1 # amount of mian properties (e.g. 2 for low & high)
        conv_mat = np.ones(k) # Conversion matrix
        
        C_SW_A  = x[     :    k] # Transducer [x[0], x[1]....x[k-1]]
        C_S_A   = x[   k :  2*k] # Signal
        C_IW_A  = x[ 2*k :  3*k] # Waste
        C_CH_A  = x[ 3*k :  4*k] # Covered-Hairpin
        C_H_A   = x[ 4*k :  5*k] # Hairpin
        C_SC_A  = x[ 5*k :  6*k] # Waste
        
        C_R1H_A = x[ 6*k :  7*k] # intermediated product
        C_P1H_A = x[ 7*k :  8*k] # intermediated product
        C_P1_A  = x[ 8*k :  9*k] # Product
        C_R2H_A = x[ 9*k : 10*k] # intermediated product
        C_P2H_A = x[10*k : 11*k] # intermediated product
        C_P2_A  = x[11*k : 12*k] # Product
        
        C_SW_B  = x[12*k : 13*k] # Transducer [x[0], x[1]....x[k-1]]
        C_S_B   = x[13*k : 14*k] # Signal
        C_IW_B  = x[14*k : 15*k] # Waste
        C_CH_B  = x[15*k : 16*k] # Covered-Hairpin
        C_H_B   = x[16*k : 17*k] # Hairpin
        C_SC_B  = x[17*k : 18*k] # Waste
        
        C_R1H_B = x[18*k : 19*k] # intermediated product
        C_P1H_B = x[19*k : 20*k] # intermediated product
        C_P1_B  = x[20*k : 21*k] # Product
        C_R2H_B = x[21*k : 22*k] # intermediated product
        C_P2H_B = x[22*k : 23*k] # intermediated product
        C_P2_B  = x[23*k : 24*k] # Product

        C_I_A   = x[24*k : 25*k] # Input x[0]
        C_I_B   = x[25*k : 26*k]
        C_R1    = x[26*k : 27*k] # Primer
        C_R2    = x[27*k : 28*k] # Primer
        
        # rate equation

        dSW_Adt = -self.k_inputtosignal_mat*(C_I_A*C_SW_A) +self.k_signaltoinput_mat*(C_S_A*C_IW_A)    
        dS_Adt  =  self.k_inputtosignal_mat*(C_I_A*C_SW_A) -self.k_signaltoinput_mat*(C_S_A*C_IW_A) -self.k_signaltohairpin_mat*C_S_A*C_CH_A +self.k_hairpintosignal_mat*C_H_A*C_SC_A
        dIW_Adt =  self.k_inputtosignal_mat*(C_I_A*C_SW_A) -self.k_signaltoinput_mat*(C_S_A*C_IW_A)  #Waste
        dCH_Adt = -self.k_signaltohairpin_mat*C_S_A*C_CH_A +self.k_hairpintosignal_mat*C_H_A*C_SC_A
        dH_Adt  =  self.k_signaltohairpin_mat*C_S_A*C_CH_A -self.k_hairpintosignal_mat*C_H_A*C_SC_A -self.k_rh_f*C_R1*C_H_A +self.k_rh_r*C_R1H_A -self.k_ph_f*C_P1_A*C_H_A +self.k_ph_r*C_P1H_A   -self.k_rh_f_2*C_R2*C_H_A +self.k_rh_r_2*C_R2H_A -self.k_ph_f_2*C_P2_A*C_H_A +self.k_ph_r_2*C_P2H_A
        dSC_Adt =  self.k_signaltohairpin_mat*C_S_A*C_CH_A -self.k_hairpintosignal_mat*C_H_A*C_SC_A  #Waste
        
        dR1H_Adt =  self.k_rh_f*C_R1*C_H_A -self.k_rh_r*C_R1H_A -self.k_per*C_R1H_A
        dP1H_Adt =  self.k_ph_f*C_P1_A*C_H_A -self.k_ph_r*C_P1H_A +self.k_per*C_R1H_A
        dP1_Adt  = -self.k_ph_f*C_P1_A*C_H_A +self.k_ph_r*C_P1H_A
        dR2H_Adt =  self.k_rh_f_2*C_R2*C_H_A -self.k_rh_r_2*C_R2H_A -self.k_per*C_R2H_A
        dP2H_Adt =  self.k_ph_f_2*C_P2_A*C_H_A -self.k_ph_r_2*C_P2H_A +self.k_per*C_R2H_A
        dP2_Adt  = -self.k_ph_f_2*C_P2_A*C_H_A +self.k_ph_r_2*C_P2H_A
        
        dSW_Bdt = -self.k_inputtosignal_mat*(C_I_B*C_SW_B) +self.k_signaltoinput_mat*(C_S_B*C_IW_B)    
        dS_Bdt  =  self.k_inputtosignal_mat*(C_I_B*C_SW_B) -self.k_signaltoinput_mat*(C_S_B*C_IW_B) -self.k_signaltohairpin_mat*C_S_B*C_CH_B +self.k_hairpintosignal_mat*C_H_B*C_SC_B
        dIW_Bdt =  self.k_inputtosignal_mat*(C_I_B*C_SW_B) -self.k_signaltoinput_mat*(C_S_B*C_IW_B)  #Waste
        dCH_Bdt = -self.k_signaltohairpin_mat*C_S_B*C_CH_B +self.k_hairpintosignal_mat*C_H_B*C_SC_B
        dH_Bdt  =  self.k_signaltohairpin_mat*C_S_B*C_CH_B -self.k_hairpintosignal_mat*C_H_B*C_SC_B -self.k_rh_f*C_R1*C_H_B +self.k_rh_r*C_R1H_B -self.k_ph_f*C_P1_B*C_H_B +self.k_ph_r*C_P1H_B    -self.k_rh_f_2*C_R2*C_H_B +self.k_rh_r_2*C_R2H_B -self.k_ph_f_2*C_P2_B*C_H_B +self.k_ph_r_2*C_P2H_B
        dSC_Bdt =  self.k_signaltohairpin_mat*C_S_B*C_CH_B -self.k_hairpintosignal_mat*C_H_B*C_SC_B  #Waste
        
        dR1H_Bdt =  self.k_rh_f*C_R1*C_H_B -self.k_rh_r*C_R1H_B -self.k_per*C_R1H_B
        dP1H_Bdt =  self.k_ph_f*C_P1_B*C_H_B -self.k_ph_r*C_P1H_B +self.k_per*C_R1H_B
        dP1_Bdt  = -self.k_ph_f*C_P1_B*C_H_B +self.k_ph_r*C_P1H_B
        dR2H_Bdt =  self.k_rh_f_2*C_R2*C_H_B -self.k_rh_r_2*C_R2H_B -self.k_per*C_R2H_B
        dP2H_Bdt =  self.k_ph_f_2*C_P2_B*C_H_B -self.k_ph_r_2*C_P2H_B +self.k_per*C_R2H_B
        dP2_Bdt  = -self.k_ph_f_2*C_P2_B*C_H_B +self.k_ph_r_2*C_P2H_B
        
        dI_Adt  = -self.k_inputtosignal_mat*C_I_A*C_SW_A + self.k_signaltoinput_mat*C_S_A*C_IW_A
        dI_Bdt  = -self.k_inputtosignal_mat*C_I_B*C_SW_B + self.k_signaltoinput_mat*C_S_B*C_IW_B
        
        dR1dt  = -self.k_rh_f*C_R1*C_H_A +self.k_rh_r*C_R1H_A   -self.k_rh_f*C_R1*C_H_B +self.k_rh_r*C_R1H_B
        dR2dt  = -self.k_rh_f_2*C_R2*C_H_A +self.k_rh_r_2*C_R2H_A   -self.k_rh_f_2*C_R2*C_H_B +self.k_rh_r_2*C_R2H_B

        return np.concatenate([dSW_Adt, dS_Adt, dIW_Adt, dCH_Adt, dH_Adt, dSC_Adt, dR1H_Adt, dP1H_Adt, dP1_Adt, dR2H_Adt, dP2H_Adt,  dP2_Adt, dSW_Bdt, dS_Bdt, dIW_Bdt, dCH_Bdt, dH_Bdt, dSC_Bdt, dR1H_Bdt, dP1H_Bdt, dP1_Bdt, dR2H_Bdt, dP2H_Bdt, dP2_Bdt,  dI_Adt, dI_Bdt, dR1dt, dR2dt], dtype=np.dtype(float))

    def reaction(self, t, C0, t_eval):
        return spi.solve_ivp(self.rxn, t, C0, t_eval = t_eval, method='Radau')