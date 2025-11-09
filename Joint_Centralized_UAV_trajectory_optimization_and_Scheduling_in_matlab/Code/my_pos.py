def cvx_pos(user_loc,UAV_loc,main_UAV_loc,initial_sched,user_profile,sigma2,H,B,backhaul,back_power,back_initial,max_speed,min_dist,pow_init, acc, iteration):

    import numpy as np
    import cvxpy as cp

    
    # Get dimensions (using shape instead of size)
    K = user_loc.shape[1]  
    M = UAV_loc.shape[1]   
    D = backhaul.shape[1] 

    q = cp.Variable((2,M),name="q")
    s_slack = cp.Variable((M,K),name="s_slack",nonneg=True)
    w_slack = cp.Variable((M,K),name="w_slack",nonneg=True)
    s_slack1 = cp.Variable((M,D),name="s_slack",nonneg=True)
    w_slack1= cp.Variable((M,D),name="w_slack",nonneg=True)

    
    # objective function
    obj = 0


    



    
    g = []  # List to store constraints

    # max speed
    for i in range(M):
        g.append(cp.norm(q[:,i]-main_UAV_loc[:,i])<=max_speed)

    # safe distance
# A clearer and safer way to write the same constraint
    for i in range(M):
        for j in range(i + 1, M):
            q_i0 = UAV_loc[:, i]
            q_j0 = UAV_loc[:, j]
            
            # This is f(q0)
            dist_sq_initial = np.sum((q_i0 - q_j0)**2)
            
            # This is grad(f(q0))^T * (q - q0)
            grad_term = 2 * (q_i0 - q_j0).T @ ((q[:, i] - q_i0) - (q[:, j] - q_j0))
            
            # The full linearized constraint: f(q0) + grad^T * (q - q0) >= min_dist^2
            linearized_dist_sq = dist_sq_initial + grad_term
            g.append(linearized_dist_sq >= min_dist**2)




                

    for i in range(M):
        backhaul_temp = 0
        for j in range(K):
            for k in range(B):
                temp1 = 0.0
                temp2 = 0.0
                deriv_w = 0.0
                
                for d in range(M):
                    if d != i:
                        # Calculate the denominator term
                        denominator1 = H**2 + np.linalg.norm(UAV_loc[:, d] - user_loc[:, j])**2
                        denominator2 = H**2 + s_slack[d,j]
                        # Sum the scheduled resources multiplied by the interference term
                        temp1 +=  (pow_init[d,k] / denominator1)
                        temp2 += pow_init[d,k]* w_slack[d,j]
                        deriv_w += pow_init[d,k]*(w_slack[d,j] - 1/denominator1)
                
                
               
                
                # Calculate the numerator term
                numerator = pow_init[i,k] / (H**2 + np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2)

                # forming taylor approx.
                val0 = numerator + (temp1 + sigma2)
                val1 = (np.log2(val0))
                

                grad = []
                var = []
                for l in range(M):

                    grad.append(-cp.abs(pow_init[l,k])/val0/(H**2+np.linalg.norm(UAV_loc[:, l] - user_loc[:, j])**2)**2/np.log(2))
                    var.append(cp.norm(q[:,l]-user_loc[:,j],2)**2-np.linalg.norm(UAV_loc[:, l] - user_loc[:, j])**2)


                result_obj = cp.sum([a1*a2 for a1, a2 in zip(grad, var)])
                result_back = cp.sum([grad[l]*var[l] for l in range(M) if l != i])*val0/(temp1 + sigma2)
                
                expr_obj = val1 + result_obj - (np.log2(temp1 + sigma2) + deriv_w/(temp1+sigma2)/np.log(2))
                

                obj += initial_sched[k,i,j]*(user_profile[j]**1) * expr_obj

################################## front rate needed to pass through backhaul

                expr_back = np.log2(temp1 + sigma2) + result_back - (np.log2(val0) +
                deriv_w/(val0)/np.log(2) + pow_init[i,k]/(val0)/np.log(2)*(w_slack[i,j]  - 1/(H**2 + np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2))   )
                
                
                backhaul_temp += initial_sched[k,i,j] * (-expr_back)



                

            g.append(s_slack[i,j]==np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2 + 2*(UAV_loc[:, i]-user_loc[:, j]).T@(q[:, i]-UAV_loc[:, i]) )
            g.append(w_slack[i,j]>=cp.inv_pos(H**2 + s_slack[i,j]) )

            ################################################################################ backhaul rate
            back_arr = []
        for c in range(D):
            temp1 = 0.0
            temp2 = 0.0
            deriv_w = 0.0
            
            for d in range(M):
                if d != i:
                    # Calculate the denominator term
                    denominator1 = H**2 + np.linalg.norm(UAV_loc[:, d] - backhaul[:, c])**2
                    denominator2 = H**2 + s_slack1[d,c]
                    # Sum the scheduled resources multiplied by the interference term
                    temp1 +=  (pow_init[d,-1] / denominator1)
                    temp2 += pow_init[d,-1]* w_slack1[d,c]
                    deriv_w += pow_init[d,-1]*(w_slack1[d,c] - 1/denominator1)
            
            
           
            
            # Calculate the numerator term
            numerator = pow_init[i,-1] / (H**2 + np.linalg.norm(UAV_loc[:, i] - backhaul[:, c])**2)

            # forming taylor approx.
            val0 = numerator + (temp1 + sigma2)
            val1 = (np.log2(val0))
            

            grad = []
            var = []
            for l in range(M):

                grad.append(-cp.abs(pow_init[l,-1])/val0/(H**2+np.linalg.norm(UAV_loc[:, l] - backhaul[:, c])**2)**2/np.log(2))
                var.append(cp.norm(q[:,l]-backhaul[:, c],2)**2-np.linalg.norm(UAV_loc[:, l] - backhaul[:, c])**2)



            result_obj = cp.sum([a1*a2 for a1, a2 in zip(grad, var)])
            
            expr_obj1 = val1 + result_obj - (np.log2(temp1 + sigma2) + deriv_w/(temp1+sigma2)/np.log(2))

            

##################################


            g.append(s_slack1[i,c]==np.linalg.norm(UAV_loc[:, i] - backhaul[:, c])**2 + 2*(UAV_loc[:, i]-backhaul[:, c]).T@(q[:, i]-UAV_loc[:, i]) )
            g.append(w_slack1[i,c]>=cp.inv_pos(H**2 + s_slack1[i,c]) )

            
          ###############################backhaul constraints  
        # for r in range(D):

            if np.round(back_initial[i,c],decimals=2)>0:
                g.append(backhaul_temp-expr_obj1<=0)
  
               



    
    objective = cp.Maximize(obj)

    for i in range(M):
        g.append(cp.norm(q[:,i] - UAV_loc[:,i], 2) <= max_speed/10)
   
    
    problem = cp.Problem(objective, g)


    x = False
    if iteration== 2222:
        x = True

    
    
    problem.solve(
    solver=cp.MOSEK,verbose=x,
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

    

    
    return q.value, problem.value
