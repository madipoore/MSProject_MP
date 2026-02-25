#import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import estimatorsv2 as est


#data_choice = "Replication"
data_choice = "50_anomaly"
#data_choice = "Scratch"

if data_choice == "Replication": 
    portfolio_path = "/Users/madisonpoore/Desktop/MSProject/Data/25_Portfolios_5x5_average_value_weighted_returns_monthly.csv"
    factors_path   = "/Users/madisonpoore/Desktop/MSProject/Data/F-F_Research_Data_Factors.csv"

    df_port = pd.read_csv(portfolio_path)
    df_fact = pd.read_csv(factors_path)

    df_port = df_port.rename(columns={df_port.columns[0]: 'Date'})
    df_fact = df_fact.rename(columns={df_fact.columns[0]: 'Date'})
    df_full = df_port.merge(df_fact, on='Date', how='inner')
    df_full = df_full.set_index('Date')
    # print(df_full.head())

if data_choice == "Scratch":
    FILE_PATH = "/Users/madisonpoore/Desktop/ff25_orth_ex_scaled.pkl"
    df_full = pd.read_pickle(FILE_PATH)

if data_choice == "50_anomaly":
    fifty_daily_path = "/Users/madisonpoore/Desktop/MSProject/Data/Instruments/managed_portfolios_anom_d_50.csv"
    df_50 = pd.read_csv(fifty_daily_path)
    df_50['date'] = pd.to_datetime(df_50['date'], format='%m/%d/%Y')
    df_50 = df_50.set_index('date')

    # Drop any empty rows or columns if needed
    df_50 = df_50.dropna(how='all', axis=1)
    df_50 = df_50.dropna(how='all', axis=0)
    df_full = df_50 
    print("50 anomaly portfolios shape:", df_50.shape)
    print("Date range:", df_50.index.min().strftime('%Y-%m-%d'), "to", df_50.index.max().strftime('%Y-%m-%d'))
    print("First 5 columns:", df_50.columns[:5].tolist())
    print("First 5 rows:\n", df_50.head())


def estimate(data, n_folds=3, kappa_min=1e-5, kappa_max=10, num_kappa=100, scale=0.5, objective='CSR2'):
    """
    Driver function that calls helper functions from estimatorsv2.py 
    
    :param data: portfolio return columns with additional columns: Date, RF, Mkt-RF, SMB, HML, Rm
    :param n_folds: number of folds used in the cross validation (3 by default)
    :param kappa_min: lower bound for kappa grid
    :param kappa_max: upper bound for kappa grid
    :param num_kappa: number of points in the kappa grid
    :param scale: scale parameter for estimator (larger = weaker shrinkage)
    :param objective: objective to optimize (currently 'CSR2')
    """

    # Loading & processing data 
    port_cols = [col for col in data.columns 
                 if col not in ['Date', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Rm']]
    orth_ex_scaled = data[port_cols]
    F = orth_ex_scaled.values
    T, H = F.shape

    # Sample covariance and sample mean 
    Sigma  = orth_ex_scaled.cov(ddof=1).values + 1e-8 * np.eye(H)
    print("Sigma diagonal mean:", np.mean(np.diag(Sigma)))
    mu_bar = orth_ex_scaled.mean().values

    # Cross validation setup 
    fold_size = T // n_folds
    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, T)
        train_idx = np.r_[0:test_start, test_end:T]
        test_idx = np.r_[test_start:test_end]
        folds.append((train_idx, test_idx))

    # Kappa grid — no zero (avoids divide-by-zero)
    kappa_grid = np.logspace(np.log10(kappa_min), np.log10(kappa_max), num_kappa)
    print(f"Generated kappa grid: {len(kappa_grid)} points from {kappa_min:.1e} to {kappa_max}")

    oos_r2_a = np.zeros(len(kappa_grid))
    oos_se_a = np.zeros(len(kappa_grid))
    ins_r2_a = np.zeros(len(kappa_grid))

    for i, kappa in enumerate(kappa_grid):
        fold_r2 = []
        for fold_num, (train_idx, test_idx) in enumerate(folds):
            train = orth_ex_scaled.iloc[train_idx]
            test  = orth_ex_scaled.iloc[test_idx]
            
            mu_tr = train.mean().values
            Sigma_tr = train.cov(ddof=1).values + 1e-8 * np.eye(H)
            
            # keeping kappa above a certain threshold 
            safe_kappa = max(kappa, 1e-6)
            b = est.l2_shrinkage(mu_tr, Sigma_tr, safe_kappa, len(train), scale)
            
            mu_te = test.mean().values
            Sigma_te = Sigma
            r2 = est.cross_sectional_r2(mu_te, Sigma_te, b)
            fold_r2.append(r2)
        
        oos_r2_a[i] = np.mean(fold_r2)
        oos_se_a[i] = np.std(fold_r2) / np.sqrt(n_folds)
        
        b_full = est.l2_shrinkage(mu_bar, Sigma, safe_kappa, T, scale)
        ins_r2_a[i] = est.cross_sectional_r2(mu_bar, Sigma, b_full)

    # peaks
    peak_idx = np.argmax(oos_r2_a)
    peak_kappa = kappa_grid[peak_idx]
    peak_r2 = oos_r2_a[peak_idx]
 
    plt.figure(figsize=(10, 6))
    plt.plot(kappa_grid, ins_r2_a, 'k--', linewidth=2, label='In-sample R²')
    plt.plot(kappa_grid, oos_r2_a, 'b-', linewidth=2.5, label='OOS CV R²')
    plt.fill_between(kappa_grid, oos_r2_a - oos_se_a, oos_r2_a + oos_se_a,
                    color='blue', alpha=0.15, label='±1 SE (approx)')
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(0, 1)
    plt.xlabel('κ (prior root expected SR²)')
    plt.ylabel('Cross-sectional R²')
    plt.title('Figure 2 Panel (a) — Pure L2 shrinkage (no sparsity imposed)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    return peak_idx, peak_kappa, peak_r2


estimate(df_full, scale=0.01)


