import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#read in the full_df from data.py file 
FILE_PATH = "/Users/madisonpoore/Desktop/ff25_orth_ex_scaled.pkl"
df_full = pd.read_pickle(FILE_PATH)

# print("Shape:", df_full.shape)
# print("Columns:", df_full.columns.tolist())

#port_cols is the list of portfolio names
port_cols = [col for col in df_full.columns 
             if col not in ['Date', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Rm']]
orth_ex_scaled = df_full[port_cols]

# print(orth_ex_scaled.shape)
# print(port_cols[:5])
# print(df_full['Date'].head().tolist())
# print(df_full['RF'].head())

# \bar{\mu} and \bar{\Sigma}: one for each of the 25 portfolios
mu_bar = orth_ex_scaled.mean().values
Sigma  = orth_ex_scaled.cov(ddof=1).values

# print("mu_bar:", mu_bar)
# print("Sigma:", Sigma)


"""
Naive estimator (eq 17 from pg 276)
resulting naive b's are the weights of the SDF 

m_t = 1 - b_1*F_{1,t} - b_2*F_{2,t} - ... - b_25*F_{25,t}

"""
Sigma_inv = np.linalg.inv(Sigma)
b_naive = -Sigma_inv @ mu_bar
# print(b_naive)


"""
To focus on uncertainty about factor means, the most
important source of fragility in the estimation, we proceed
under the assumption that is known. (p 276)

Here is the implementation of the L2 ONLY shrinkage estimator 
"""

#portfolio values
F = orth_ex_scaled[port_cols].values
T, H = F.shape
# print(F.shape)
# print("T:", T)
# print("H:", H)

mu_bar = F.mean(axis=0)
Sigma = np.cov(F, rowvar=False, ddof=1) + 1e-8 * np.eye(H)

#L2 shrinkage (pg 277)
def l2_shrinkage(mu, Sigma, kappa, T, scale=0.015):
    #equation 22 
    tau = np.trace(Sigma)
    #contributing to R^2?
    gamma = tau / (kappa**2 * T * scale)
    Sigma_reg = Sigma + gamma * np.eye(len(mu))
    b = np.linalg.inv(Sigma_reg) @ mu
    return b

def cross_sectional_r2(mu, Sigma, b):
    pred = Sigma @ b
    ss_res = np.sum((mu - pred)**2)
    ss_tot = np.sum(mu**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

#making 3 folds 
n_folds = 3
fold_size = T // n_folds
folds = []
for i in range(n_folds):
    test_start = i * fold_size
    test_end = min(test_start + fold_size, T)
    train_idx = np.r_[0:test_start, test_end:T]
    test_idx = np.r_[test_start:test_end]
    folds.append((train_idx, test_idx))

#print("\nReplicating Panel (a) from pg 281")

kappa_grid = np.logspace(-2, 1, 60)
oos_r2_a = np.zeros(len(kappa_grid))
oos_se_a = np.zeros(len(kappa_grid))
ins_r2_a = np.zeros(len(kappa_grid))

for i, kappa in enumerate(kappa_grid):
    fold_r2 = []
    for train_idx, test_idx in folds:
        train = orth_ex_scaled.iloc[train_idx]
        test  = orth_ex_scaled.iloc[test_idx]
        
        mu_tr = train.mean().values
        Sigma_tr = train.cov(ddof=1).values + 1e-8 * np.eye(H)
        
        b = l2_shrinkage(mu_tr, Sigma_tr, kappa, len(train))
        
        mu_te = test.mean().values
        Sigma_te = Sigma_tr 
        r2 = cross_sectional_r2(mu_te, Sigma_te, b)
        fold_r2.append(r2)
    
    oos_r2_a[i] = np.mean(fold_r2)
    oos_se_a[i] = np.std(fold_r2) / np.sqrt(n_folds)
    
    # In-sample on full data
    b_full = l2_shrinkage(mu_bar, Sigma, kappa, T)
    ins_r2_a[i] = cross_sectional_r2(mu_bar, Sigma, b_full)

#plotting panel a
plt.figure(figsize=(10, 6))
plt.plot(kappa_grid, ins_r2_a, 'k--', label='In-sample R squared')
plt.plot(kappa_grid, oos_r2_a, 'b-', label='OOS CV R squared')
plt.fill_between(kappa_grid, oos_r2_a - oos_se_a, oos_r2_a + oos_se_a, 
                 color='blue', alpha=0.15)
plt.xscale('log')
plt.xlabel('kappa')
plt.ylabel('R squared')
plt.title('Replicating figure (a) from pg 281')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Peak OOS R² (Panel a): {oos_r2_a.max()} at κ ≈ {kappa_grid[oos_r2_a.argmax()]}")