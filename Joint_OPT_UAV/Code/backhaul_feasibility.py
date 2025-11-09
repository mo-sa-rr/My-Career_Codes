import numpy as np

def bfc(sched, UAV_loc, user_loc, sigma2, H, B, user_profile,bandwidth,power, back_power, backhaul):
    bandwidth = 1
    
    _, K = user_loc.shape
    _, M = UAV_loc.shape
    _, D = backhaul.shape

    feas_array = np.zeros((M,D))
    
    sched = np.round(sched,decimals=3)

    for i in range(M):
        backcons = 0 
        for j in range(K):
            for k in range(B):
                path_loss_signal = H**2 + np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2
                
                interference_at_init = 0

                for d in range(M):
                    if d != i:
                        path_loss_interf = H**2 + np.linalg.norm(UAV_loc[:, d] - user_loc[:, j])**2
                        gain_interf = 1 / path_loss_interf

                        
                        interference_at_init +=  power[d, k] * gain_interf


                val = np.log2(interference_at_init + sigma2)

                signal_variable = power[i, k] / path_loss_signal
                log_term = sched[k,i,j]*(np.log2(signal_variable + interference_at_init + sigma2) - val) 

                

                backcons += log_term
                
        
        
        for r in range(D):
            path_loss_backhaul = H**2 + np.linalg.norm(UAV_loc[:, i] - backhaul[:, r])**2
            rate_const = np.log2(1 + (back_power / (path_loss_backhaul * sigma2)))

            feas_array[i,r] = backcons-rate_const



    return feas_array
