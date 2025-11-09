import numpy as np

def back_adjust(sched, UAV_loc, user_loc, sigma2, H, B, user_profile,bandwidth,power, back_power, backhaul):
    # keep API & style the same; ignore back_power and use the last column of `power` for backhaul
    bandwidth = 1
    
    _, K = user_loc.shape
    _, M = UAV_loc.shape
    _, D = backhaul.shape

    feas_array = np.zeros((M, D))
    back_corr  = np.zeros((M, D))

    # your original rounding
    sched = np.round(sched, decimals=3)

    for i in range(M):
        backcons = 0.0
        # ---- access-side load (unchanged) ----
        for j in range(K):
            for k in range(B):
                path_loss_signal = H**2 + np.linalg.norm(UAV_loc[:, i] - user_loc[:, j])**2

                interference_at_init = 0.0
                for d in range(M):
                    if d != i:
                        path_loss_interf = H**2 + np.linalg.norm(UAV_loc[:, d] - user_loc[:, j])**2
                        gain_interf = 1.0 / path_loss_interf
                        interference_at_init += power[d, k] * gain_interf

                val = np.log2(interference_at_init + sigma2)
                signal_variable = power[i, k] / path_loss_signal
                log_term = sched[k, i, j] * (np.log2(signal_variable + interference_at_init + sigma2) - val)
                backcons += log_term

        # ---- shared-band backhaul capacity using power[:, -1] ----
        # bp is the per-UAV backhaul power (last column)
        bp = power[:, -1]

        for r in range(D):
            # signal from UAV i to backhaul r
            PL_i = H**2 + np.linalg.norm(UAV_loc[:, i] - backhaul[:, r])**2
            S = bp[i] / PL_i

            # interference at r from all other UAVs on the backhaul band
            I = 0.0
            for d in range(M):
                if d != i:
                    PL_d = H**2 + np.linalg.norm(UAV_loc[:, d] - backhaul[:, r])**2
                    I += bp[d] / PL_d

            # interference-aware backhaul capacity: log2((S+I+N)/(I+N))
            rate_const = np.log2(S + I + sigma2) - np.log2(I + sigma2)

            if backcons - rate_const <= 1e-3:
                back_corr[i, r] = 1

    row_sums = back_corr.sum(axis=1, keepdims=True)
    back_corr_normalized = back_corr / row_sums
    return back_corr_normalized
