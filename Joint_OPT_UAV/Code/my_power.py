import numpy as np
import cvxpy as cp

def cvx_power(sched, user_loc, user_profile, UAV_loc, sigma2, H, B, backhaul, 
                         back_initial, back_power, pow_lim, pow_init, acc, iteration):


    eps = 1e-9  ###########################


    # in-place (keeps the same array object)
    back_initial = np.maximum(back_initial, eps)

    
    # --- Get dimensions from the input arrays ---
    K = user_loc.shape[1]
    M = UAV_loc.shape[1]
    D = backhaul.shape[1]

    # --- 1. Define Optimization Variables ---

    power = cp.Variable((M, B+1), name="power", nonneg=True)

    


    z_switch = cp.Variable((M,D),name="z_switch",nonneg=True)

    # --- 2. Define the Objective Function ---
    # The objective is to maximize the weighted sum-rate of all users.
    obj = 0

     





    
    
 
    
    # --- 3. Define Constraints ---
    g = []  # List to store all constraints


    for i in range(M):

        back_cons = 0
        for j in range(K):
            for k in range(B):
                path_loss_signal = H**2 + np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2
                
                interference_at_init = 0
                derivative_term = 0
                interference_variable = 0

                for d in range(M):
                    if d != i:
                        path_loss_interf = H**2 + np.linalg.norm(UAV_loc[:, d] - user_loc[:, j])**2
                        gain_interf = 1 / path_loss_interf
                        

                        
                        
                        interference_at_init +=  pow_init[d, k] * gain_interf
                        derivative_term +=  gain_interf * (power[d, k] - pow_init[d, k])
                        interference_variable +=  power[d, k] * gain_interf

                val0 = np.log2(interference_at_init + sigma2)
                val1 = np.log2(pow_init[i, k] / path_loss_signal+interference_at_init + sigma2)
                expr = val0 + derivative_term / ((interference_at_init + sigma2) * np.log(2))

                signal_variable = power[i, k] / path_loss_signal
                log_term = cp.log(signal_variable + interference_variable + sigma2) / np.log(2)

                
                
                obj += (user_profile[j]**1) *sched[k, i, j] *(log_term - expr)
                

                derivative_term_back = (derivative_term + (power[i, k] - pow_init[i, k])/path_loss_signal)/(pow_init[i, k] / path_loss_signal+interference_at_init + sigma2)/np.log(2)
                
                back_cons += sched[k, i, j] * (val1 + derivative_term_back - cp.log(interference_variable + sigma2)/np.log(2))

        weighted_capacity_i = 0
        
         
      
        for r in range(D):
            temp1= 0
            temp2= 0
            temp3= 0
            for d in range(M):
                if d != i:
                    
                    # Calculate the denominator term
                    denominator1 = H**2 + np.linalg.norm(UAV_loc[:, d] - backhaul[:, r])**2
                    # Sum the scheduled resources multiplied by the interference term
                    temp1 +=  (pow_init[d,-1] / denominator1)
                            
                    temp2 += -back_initial[i,r]/(denominator1)*(power[d,-1]-pow_init[d,-1])
                    temp3 += (power[d,-1] / denominator1)
                    


            
            val0 = back_initial[i,r]*np.log2(back_initial[i,r]/(temp1 + sigma2))
            val_der = val0 + temp2/(temp1+sigma2)/np.log(2) + (np.log(back_initial[i,r]/(temp1 + sigma2))+1)/np.log(2)*(z_switch[i,r]-back_initial[i,r])
            val2 = temp3 + power[i,-1]/(H**2 + np.linalg.norm(UAV_loc[:, i] - backhaul[:, r])**2) + sigma2
            
            
            weighted_capacity_i += val_der-cp.rel_entr(z_switch[i,r],val2)/np.log(2)
        

        g.append(back_cons <= weighted_capacity_i)

   

    

            
    

            
    g.append(cp.sum(z_switch,axis=1) == 1)
    g.append(z_switch >= 0)
    g.append(z_switch <= 1)

    
    
           

    # -- Constraint Set C: Total Power Limit --
    g.append(cp.sum(power,axis=1) <= pow_lim)
    

    objective = cp.Maximize(obj)
    problem = cp.Problem(objective, g)


        

    
    problem.solve(
    solver=cp.MOSEK,verbose = x,
    mosek_params={
        # Relative optimality gap tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": acc,

        # Dual feasibility tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": acc,

        # Primal feasibility tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": acc,

        # Absolute gap tolerance
        # "MSK_DPAR_INTPNT_CO_TOL_MU_RED": acc,

        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000
    }
)



    return z_switch.value , power.value, problem.value
