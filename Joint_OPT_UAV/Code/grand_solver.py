import pdb

import numpy as np
from my_pos import cvx_pos

from my_sched import cvx_sched

from my_power import cvx_power

from PFS_update_profile_General import PFS_update_profile as pfs1
import pdb
import cvxpy as cp
import time
from backhaul_feasibility import bfc
from back_adjust import back_adjust



def grand_solver(user_loc, UAV_loc, initial_sched, user_profile, sigma2, H, B, backhaul, back_power, back_initial, max_speed, min_dist,pow_init,pow_lim):

    _, M = UAV_loc.shape
    _, K = user_loc.shape
    _, D = back_initial.shape

    # --- Initialization ---
    # Current iteration's variables
    o1 = initial_sched.copy()
    o2 = back_initial.copy()
    o3 = pow_init.copy()
    o4 = UAV_loc.copy()
    
    # Previous iteration's variables (for momentum)
    o1_prev, o2_prev, o3_prev , o4_prev = o1.copy(), o2.copy(), o3.copy(), o4.copy()
    
    # Variables for convergence check (initialized to fail the first check)
    o1_temp, o4_temp = o1 + 1, o4 + 1
    
    val_temp = -10000

    
    
    counter_sched = 0

    con_crit = 0.0003

    # --- Alternating Optimization Loop ---
    start_time = time.time()



    reward = 10



    flag = 1
    
    while True:

        counter_sched += 1
        
        # gamma = 0.9**(counter_sched-1)
        gamma= 0

        
        # counter = 0
        
        # while True:
        #     try:
        o1_next = o1 + 0 * (o1 - o1_prev)
        o2_next = o2 + gamma * (o2 - o2_prev)
        o3_next = o3 + gamma * (o3 - o3_prev)
        o4_next = o4 + gamma * (o4 - o4_prev)

        
        # print("3",o3_next)
        # print("4",o4_next)


        acc = 1e-8
        
        if counter_sched <= 1000 and flag:
            for n in range(9):
                try:
                    o1_candidate, val_sched = cvx_sched(
                        user_loc.copy(), o4_next.copy(), o1_next.copy(), 
                        user_profile.copy(), sigma2, H, B, backhaul.copy(), back_power,
                        o2_next.copy(),o3_next.copy(),acc, counter_sched, reward)
                    
                    if o1_candidate is not None:
                        break
        
                    acc *= 10
                    print("sched none",acc)    
                    
                except Exception as e:
                    print("sched",e)
                    acc *= 10


        o1_candidate = np.abs(o1_candidate)

        int_meter = np.sum((o1_candidate)*(-o1_candidate + 1))

        if int_meter <=0.001:
            flag = 0
            
        # Step 1: Solve for schedule and backhaul (candidate solution)

           
        
                       

        # o1_candidate = np.round(o1_candidate,decimals=3)
        
        # print("o1 (schedule):\n", np.round(o1_candidate,decimals=3),reward)  

        
        



        # print("feaspos:\n", back_feas)
        
        acc = 1e-8
       
        
        for n in range(9):
            try:
                o2_candidate, o3_candidate, val_pow = cvx_power(
                    o1_candidate.copy(), user_loc.copy(), user_profile.copy(), o4_next.copy(), sigma2, H, B, backhaul.copy(), 
                         o2_next.copy(), back_power, pow_lim, o3_next.copy(), acc, counter_sched
                )

                if o3_candidate is not None:

                    # print("pow",acc)
                    break
                
                acc *= 10
                # print("pow none",acc)
                                        

            except Exception as e:
                # print("pow",e)
                acc *= 10



        # pdb.set_trace()
        o2_candidate = back_adjust(o1_candidate.copy(), o4_next.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(),1,o3_candidate.copy(), back_power, backhaul.copy())
        
        # print("o3 (powers):\n", np.round(o3_candidate,decimals=3))
        # print("o2 (backschedule):\n", o2_candidate)
        
        acc = 1e-8
        
        for n in range(9):
            try:
                o4_candidate, val_pos = cvx_pos(
                    user_loc.copy(), o4_next.copy(), UAV_loc.copy(), o1_candidate.copy(), 
                    user_profile.copy(), sigma2, H, B, backhaul.copy(), back_power, 
                    o2_candidate.copy(), max_speed, min_dist, o3_candidate.copy(), acc, counter_sched)
                
                if o4_candidate is not None:
                    break

                acc *= 10
                # print("pos none",acc)
                
            except Exception as e:
                # print("pos",e)
                acc *= 10

        # print("o4 (positions):\n", o4_candidate)

    # print("feas:\n", back_feas)
            #     break

            

        
        back_feas = bfc(o1_candidate.copy(), o4_candidate.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(),1,o3_candidate.copy(), back_power, backhaul.copy())   
        

        # Update previous values for the next main iteration's momentum calculation
        o1_prev, o2_prev = o1.copy(), o2.copy()
        o3_prev = o3.copy()
        o4_prev = o4.copy()
        # --- Update successful results ---
        # If we are here, it means both solvers succeeded in the try block
        o1 = o1_candidate
        o2 = o2_candidate
        o3 = o3_candidate
        o4 = o4_candidate



        if counter_sched%50 == 0:
            con_crit = 0.0015




        
        
        # --- Convergence Check ---
        if counter_sched % 1 == 0:
            _ , val_rate1 = pfs1(o1_next, o4_next, user_loc.copy(), sigma2, H, B, user_profile.copy(), 0.2 ,1 ,o3_next.copy())
            _ , val_rate2 = pfs1(o1_candidate.copy(), o4_next.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(), 0.2 ,1 ,o3_next.copy())
            _ , val_rate3 = pfs1(o1_candidate.copy(), o4_candidate.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(), 0.2 ,1 ,o3_next.copy())
            _ , val_rate4 = pfs1(o1_candidate.copy(), o4_candidate.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(), 0.2 ,1 ,o3_candidate.copy())



            #////////// uncomment below lines to check variable solutions after each block 
            
            #print("val_rate1:\n", val_rate1)
            #print("sum_rate1:\n", np.sum(val_rate1))
            # print("sum_rate2:\n", np.sum(val_rate2))
            # print("sum_rate3:\n", np.sum(val_rate3))
            # print("sum_rate4:\n", np.sum(val_rate4))
            
            # print("val_func1:\n", np.sum(val_rate1*user_profile))
            # print("val_func2:\n", np.sum(val_rate2*user_profile))
            # print("val_func3:\n", np.sum(val_rate3*user_profile), val_pos)
            print("val_func4:\n", np.sum(val_rate4*user_profile)+reward*np.sum(o1_candidate**3), np.sum(val_rate4*user_profile)+reward*np.sum(o1_candidate**3)-val_temp) 
            # print("crit:", con_crit)
            # # # # print("profile:\n", user_profile)
            
            # print("o1 (schedule):\n", o1)
            # print("o2 (back_schedule):\n", o2_candidate)
            # print("adj (back_schedule):\n", o2_adjusted)
            # # print("o3 (powers):\n", o3)
            # print("o4 (positions):\n", o4)
            # print("(back_feas):\n", back_feas)
            # print("counter:\n", counter_sched)
            # print("10 iter convergence time",time.time()-start_time)

            print(f"\n--- Iteration: {counter_sched} ---")
            

            
            val_func = np.sum(val_rate4*user_profile)+reward*np.sum(o1_candidate**3)
            val_con = np.abs(val_func-val_temp)





            int_meter = np.sum((o1_candidate)*(-o1_candidate + 1))
            
            if (val_con<=con_crit or counter_sched>=50) and int_meter>0.001:
                print("\n reward boost \n")
                o1 = initial_sched.copy()
                o2 = back_initial.copy()
                o3 = pow_init.copy()
                o4 = UAV_loc.copy()

                counter_sched = 0
                reward = reward*10

                val_con = 100
            


            if val_con<=con_crit or counter_sched>=100:
               
                print("\n✅ Convergence criteria met!")
                
                print("(back_feas):\n", back_feas)
                break
                
            
            # Store current values for the next convergence check
            o1_temp, o4_temp = o1.copy(), o4.copy()
            start_time = time.time()
            val_temp = val_func



        
            
    _ , val_rate5 = pfs1(o1_candidate.copy(), o4_candidate.copy(), user_loc.copy(), sigma2, H, B, user_profile.copy(), 0.2 ,1 ,o3_candidate.copy())
    print(f'\nOptimization finished in {counter_sched} iterations.')
    return o1_candidate, o2_candidate, o3_candidate, o4_candidate, np.sum(val_rate5*user_profile), val_rate5