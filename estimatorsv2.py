import numpy as np 
from scipy.linalg import pinv

def l2_shrinkage(mu, Sigma, kappa, T, scale=0.5):
    tau = np.trace(Sigma)
    gamma = tau / (kappa**2 * T * scale)
    # print(f"kappa={kappa:.6f} | gamma={gamma:.6f} | scale={scale}")
    Sigma_reg = Sigma + gamma * np.eye(len(mu))
    b = pinv(Sigma_reg) @ mu
    return b

def cross_sectional_r2(mu, Sigma, b):
    pred = Sigma @ b
    ss_res = np.sum((mu - pred)**2)
    ss_tot = np.sum(mu**2) + 1e-6
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0