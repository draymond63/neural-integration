import numpy as np
import scipy 

def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))



def get_velocity_scale_factor(velocity_data: np.ndarray, ssp_space, dt: float):
    pathlen = velocity_data.shape[0]

    real_freqs = (ssp_space.phase_matrix @ velocity_data.T)
    vel_scaling_factor = 1/np.max(np.abs(real_freqs))
    vels_scaled = velocity_data*vel_scaling_factor
    velocity_func = lambda t: vels_scaled[int(np.minimum(np.floor(t/dt), pathlen-2))]
    return velocity_func, vel_scaling_factor