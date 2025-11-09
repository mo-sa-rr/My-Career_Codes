def cvx_sched(user_loc,UAV_loc,initial_sched,user_profile,sigma2,H,B,backhaul,back_power,back_initial,pow_init,acc,counter,reward):
    import numpy as np
    import cvxpy as cp


    # Get dimensions 
    K = user_loc.shape[1]  
    M = UAV_loc.shape[1]   
    D = backhaul.shape[1] 

    alpha = cp.Variable((B,M,K),name="alpha",nonneg=True)


    obj = 0






    g = []  # List to store constraints

    # First constraint: sum over first two dimensions minus 1 <= 0
    # This should be sum over M and K for each band, but the original code sums over M and B.

    temp1 = cp.sum(alpha, axis=(0, 1)) - 1
    g.append(temp1 <= 0)


    # Second set of constraints: sum over users (axis=1) minus 1 <= 0 for each UAV i and band k

    g.append(cp.sum(alpha, axis=2) <= 1)

    

    

    
    
   
    for i in range(M):
        back_cons = 0

        for j in range(K):
            for k in range(B):

                
                dist_sq_signal = np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2
                b = pow_init[i,k] / (H**2 + dist_sq_signal)
                
                temp_initial = 0
                for d in range(M):
                    if d != i:
                        dist_sq = np.linalg.norm(UAV_loc[:, d] - user_loc[:, j])**2
                        interference_term = pow_init[d,k] / (H**2 + dist_sq)
                        temp_initial += interference_term


                obj += (user_profile[j]**1)*alpha[k,i,j]*np.log2(1+b/(temp_initial + sigma2)) + (reward)*(initial_sched[k,i,j]**3 + 3*(initial_sched[k,i,j]**2)*(alpha[k,i,j] - initial_sched[k,i,j]))
               
               
                                
                
                back_cons += alpha[k,i,j]*np.log2(1+b/(temp_initial + sigma2))

        for r in range(D):
            temp1= 0
            temp2= 0
            temp3= 0

            for d in range(M):
                if d != i:
                    
                    # Calculate the denominator term
                    denominator1 = H**2 + np.linalg.norm(UAV_loc[:, d] - backhaul[:, r])**2
                    temp1 +=  (pow_init[d,-1] / denominator1)

            dist_sq = np.linalg.norm(UAV_loc[:,i] - backhaul[:,r])**2

                    
            
            rate_const = np.log2(1 + (pow_init[i,-1] / ((H**2 + dist_sq) * (temp1+sigma2))))
            
            
            g.append(back_initial[i,r]*(back_cons - rate_const)<=0)



    


    
    objective = cp.Maximize(obj)
            
    g.append(alpha <= 1)
    
 
    
    problem = cp.Problem(objective, g)
    
    problem.solve(
    solver=cp.MOSEK,
    mosek_params={
        # Relative optimality gap tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": acc,

        # Dual feasibility tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": acc,

        # Primal feasibility tolerance (default: 1e-8)
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": acc,

        # Absolute gap tolerance
        # "MSK_DPAR_INTPNT_CO_TOL_MU_RED": acc,
    }
)

    return alpha.value, problem.value
