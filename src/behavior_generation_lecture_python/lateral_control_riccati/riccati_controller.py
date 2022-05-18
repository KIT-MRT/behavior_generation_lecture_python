import numpy as np


def feedback_law(k_lqr, k_dist_comp, e_l, e_psi, kappa_r, beta, r):
    x = np.array([e_l, e_psi, beta, r])
    delta = np.dot(-k_lqr, x) + k_dist_comp * kappa_r

    return delta
