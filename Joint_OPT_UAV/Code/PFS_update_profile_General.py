import numpy as np

def PFS_update_profile(sched, UAV_loc, user_loc, sigma2, H, B, user_profile, update_rate,bandwidth,power):
    bandwidth = 1
    
    _, K = user_loc.shape
    _, M = UAV_loc.shape

    user_rate = np.zeros(K)

    for i in range(M):
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

                

                user_rate[j] += log_term*bandwidth

    # Update user profile
    for j in range(K):
        user_profile[j] = 1.0 / ((1 - update_rate) / user_profile[j] + update_rate * user_rate[j])

    y = user_profile
    x = user_rate
    return y, x
