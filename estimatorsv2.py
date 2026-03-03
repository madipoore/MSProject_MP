import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def l2est(dates, re, market, freq, anomalies, parameters):
    # Default parameters
    parameters = parameters or {}
    default_params = {
        'gridsize': 100,
        'method': 'CV',
        'objective': 'CSR2',
        'ignore_scale': False,
        'kfold': 3,
        'oos_test_date': dates.iloc[-1],
        'freq': freq,
        'rotate_PC': False,
        'demarket_conditionally': False,
        'demarket_unconditionally': True,
        'devol_conditionally': False,
        'devol_unconditionally': True,
        'plot_dof': False,
        'plot_coefpaths': False,
        'plot_objective': True,
        'line_width': 1.5,
        'font_size': 10,
        'L2_max_legends': 20,
        'L2_sort_loc': 'opt',
        'L1_log_scale': True,
        'L2_log_scale': True,
        'legend_loc': 'best',
        'results_export': False,
        'show_plot': True
    }
    parameters = {**default_params, **parameters}

    # Objective direction
    optfunc = min if parameters["objective"] in ['GLS', 'SSE'] else max

    parameters["sObjective"] = 'Cross-sectional $R^2$'

    # Parse oos_test_date
    oos_date = parameters["oos_test_date"]
    if isinstance(oos_date, str):
        tT0 = datetime.strptime(oos_date, "%m/%d/%Y")
    elif isinstance(oos_date, (pd.Timestamp, datetime)):
        tT0 = oos_date if isinstance(oos_date, datetime) else oos_date.to_pydatetime()
    else:
        raise TypeError(f"oos_test_date type: {type(oos_date)}")

    print(f"oos_test_date type: {type(oos_date)}, value: {oos_date}")
    print(f"Parsed tT0: {tT0}")

    # Index setup
    re.index = dates.values
    market.index = dates.values

    if not pd.api.types.is_datetime64_any_dtype(re.index):
        re.index = pd.to_datetime(re.index, errors='coerce')
        market.index = pd.to_datetime(market.index, errors='coerce')

    re = re[re.index.notna()]
    market = market[market.index.notna()]
    re = re.sort_index()
    market = market.sort_index()
    mkt0 = market.copy()

    # De-market unconditionally (keep your demarket function)
    if parameters['demarket_unconditionally']:
        r_train, b_train = demarket(re.loc[:tT0, :], market.loc[:tT0])
        r_test = demarket(re.loc[tT0:, :], market.loc[tT0:], b_train)
        r0 = pd.concat([r_train, r_test], axis=0) if isinstance(r_test, pd.DataFrame) else r_train.copy()
    else:
        r0 = re.copy()

    # De-vol unconditionally (keep)
    if parameters['devol_unconditionally']:
        r0 = r0.divide(r0.std(axis=0), axis=1).multiply(market.std())

    # Train/test split
    mkt = mkt0.loc[:tT0]
    r_train = r0.loc[:tT0, :]
    r_test = r0.loc[tT0:, :]

    T, n = r_train.shape
    parameters['T'] = T
    parameters['n'] = n

    # Moments
    X = regcov(r_train)
    y = np.mean(r_train, axis=0)

    # SVD precompute
    U, D, Q = np.linalg.svd(X, full_matrices=False)
    d = D**2  # eigenvalues

    # Pseudo-inverse setup
    tol = max(X.shape) * np.finfo(float).eps * np.max(d)
    r1 = np.sum(d > tol) + 1
    Q1 = Q[:r1, :]
    s = d[:r1]
    s_inv = 1 / s
    Xinv = Q1.T @ (Q1 * s_inv[:, None])

    parameters['xlbl'] = 'Root Expected SR$^2$ (prior), $\\kappa$'
    parameters['d'] = d
    parameters['Xinv'] = Xinv

    # Kappa → penalty
    kappa2pen = lambda kappa: parameters['freq'] * np.trace(X) / T / (kappa ** 2)

    # Stabilization range (lr loop)
    lr = np.arange(1, 22)
    lm = 1
    z = np.full((n, len(lr)), np.nan)

    for i in lr:
        params_i = parameters.copy()
        params_i['L2pen'] = kappa2pen(2 ** (i - lm))
        z[:, i-1] = ridge_solve(X, y, params_i)[0]  # use helper

    # Stabilize
    rel_change = np.mean(np.abs((z[:, 1:] - z[:, :-1])) / (1 + np.abs(z[:, :-1])), axis=0) > 0.01
    x_rlim_idx = np.nonzero(rel_change)[0]
    x_rlim = 2 ** (x_rlim_idx[-1] - lm + 1) if len(x_rlim_idx) > 0 else 1.0

    # Final grid
    x = np.logspace(np.log10(x_rlim), np.log10(0.01), parameters['gridsize'])
    l = [kappa2pen(val) for val in x]
    lCV = [val / (1 - 1 / parameters['kfold']) for val in l]  # bias correction
    nl = len(l)

    # Outputs
    phi = np.full((n, nl), np.nan)
    se = np.full_like(phi, np.nan)
    objL2 = np.full((nl, 4), np.nan)  # IS, OOS, se, etc.

    for i in range(nl):
        print(f"Grid point {i+1}/{nl}")

        params['L2pen'] = l[i]
        phi[:, i], se[:, i] = ridge_solve(X, y, params, compute_errors=True)

        # Stub CV for now (replace later)
        params['L2pen'] = lCV[i]
        # TODO: cross_validate call
        objL2[i, 1] = 0.25 + np.random.normal(0, 0.02)  # dummy OOS
        objL2[i, 0] = 0.30 + np.random.normal(0, 0.02)  # dummy IS

    # Optimal
    objL2opt = optfunc(objL2[:, 1])
    iL2opt = np.argmax(objL2[:, 1]) if optfunc == max else np.argmin(objL2[:, 1])
    bL2 = phi[:, iL2opt]
    parameters['bL2'] = bL2
    parameters['R2oos'] = objL2opt
    L2optKappa = x[iL2opt]

    # [add your plotting code here if you want to plot objL2 vs x]

    return parameters


def demarket(r, mkt, b=None):
    """
    Demarket function to compute market beta and de-market returns.

    Parameters:
    - r: DataFrame or 2D array of returns.
    - mkt: Series or 1D array of market returns.
    - b: Optional; market beta. If not provided, it will be computed.

    Returns:
    - rme: DataFrame or 2D array of de-marketed returns.
    - b: market beta.
    """

    # If b (beta) is not provided, compute it
    if b is None:
        # Create a design matrix with intercept (column of ones) and market returns
        rhs = np.column_stack([np.ones(mkt.shape[0]), mkt])

        # Solve for beta using least squares
        b, _ = np.linalg.lstsq(rhs, r, rcond=None)[0:2]
        b = b[1:]

    # De-market
    rme = r - np.outer(mkt, b)

    return rme, b

def regcov(r):
    """
    Compute the regularized covariance matrix of r.

    Parameters:
    - r: Input data matrix

    Returns:
    - X: Regularized covariance matrix
    """
    
    # Compute covariance matrix
    X = np.cov(r, rowvar=False)

    # Covariance regularization (with flat Wishart prior)
    T, n = r.shape
    a = n / (n + T)
    X = a * np.trace(X) / n * np.eye(n) + (1 - a) * X

    return X

def l2_shrinkage2(X, y, params, compute_errors=False):
    l = params['L2pen']

    if compute_errors:
        Xinv = np.linalg.inv(X + l * np.eye(X.shape[0]))
        
        b = np.dot(Xinv, y)
        se = np.sqrt(1 / params['T'] * np.diag(Xinv))
    else:
        # Solve a system of linear equations instead if errors are not needed
        b = np.linalg.solve(X + l * np.eye(X.shape[0]), y)
        se = np.full(X.shape[0], np.nan)

    return b, params, se

def cross_validate(FUN, dates, r, params):
    """
    Compute IS/OOS values of the objective function based on the FUN function.
    Implements multiple objectives and validation methods.

    Parameters:
    - FUN: Handle to a function which estimates model parameters.
    - dates: (T x 1) array of dates.
    - r: (T x N) matrix of returns.
    - params: Dictionary that contains extra arguments.

    Returns:
    - obj: (1 x 2) IS and OOS values of the estimated objective function.
    - params: Returns back the params dictionary.
    - obj_folds: ...
    """
    if not callable(FUN):
        raise ValueError('Provided FUN argument is not a callable function.')

    # Select requested method
    if 'method' not in params:
        cross_validate_handler = cross_validate_cv_handler
    else:
        map_cv_method = {
            'CV': cross_validate_cv_handler,
            'ssplit': cross_validate_ssplit_handler,
            # 'bootstrap': cross_validate_bootstrap_handler
        }
        cross_validate_handler = map_cv_method.get(params['method'])

    # Execute selected method
    params['dd'] = dates
    params['ret'] = r
    params['fun'] = FUN
    obj, params, obj_folds = cross_validate_handler(params)

    return obj, params, obj_folds

def cross_validate_ssplit_handler(params):
    """
    Sample split handler for cross-validation.

    Parameters:
    - params: Dictionary with parameters, including 'splitdate', 'dd', etc.

    Returns:
    - obj, params: Results from the bootstrp_handler.
    """
    # Get split date or default to '01JAN2000'
    sd = params.get('splitdate', '01JAN2000')

    # Convert string date to datetime object
    tT0 = datetime.strptime(sd, '%d%b%Y')
    idx_test = [i for i, d in enumerate(params['dd']) if d >= tT0]

    return bootstrp_handler(idx_test, params)

def cross_validate_cv_handler(params):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    - params: dictionary containing the parameters
    
    Returns:
    - obj: (k x 2) array of IS and OOS values of the estimated objective function for each fold
    - params: updated params dictionary
    - obj_folds: (k x 2) array, equal to obj
    
    Note: Requires custom function `bootstrp_handler` and `cvpartition_contiguous`.
    """
    
    # Set k (number of folds) either to provided value or default to 2
    k = params.get('kfold', 2)
    
    cv = cvpartition_contiguous(np.size(params['ret'],0), k)
    
    # Initialize obj to hold IS/OOS stats for each partition
    obj = np.nan * np.zeros((k, 2))
    
    for i in range(k):
        
        idx_test = cv[i]
        if 'cv_idx_test' not in params:
            params['cv_idx_test'] = {}
        params['cv_idx_test'][i] = idx_test
        params['cv_iteration'] = i
        obj[i, :], params = bootstrp_handler(idx_test, params)
        
    # Store estimates for each fold
    obj_folds = obj
    
    # Compute average and standard error of IS/OOS stats across folds
    obj = np.hstack([np.mean(obj, axis=0), np.std(obj, axis=0) / np.sqrt(k)])
    
    # Uncomment and modify the following code if 'SRexpl' objective function is used
    # if params['objective'] == 'SRexpl':
    #     obj = np.sqrt(np.maximum(0, obj))
    
    return obj, params, obj_folds

def bootstrp_handler(idx_test, params):
    
    if 'objective' in params:
        map_bootstrp_obj = {
            #'SSE': bootstrp_obj_SSE,
            #'GLS': bootstrp_obj_HJdist,
            'CSR2': bootstrp_obj_CSR2,
            #'GLSR2': bootstrp_obj_GLSR2,
            #'SRexpl': bootstrp_obj_SRexpl,
            #'SR': bootstrp_obj_SR,
            #'MVU': bootstrp_obj_MVutil
        }
        
        def_bootstrp_obj = map_bootstrp_obj[params['objective']]
    else:
        def_bootstrp_obj = bootstrp_obj_CSR2

    ret = params['ret']
    FUN = params['fun']

    n = ret.shape[0]
    idx = np.setdiff1d(np.arange(n), idx_test)  # difference between two arrays, providing training indices
    n_test = len(idx_test)

    invX = np.nan
    invX_test = np.nan
    res = [np.nan, np.nan]

    if n_test > 0:
        r = ret.iloc[idx, :]
        r_test = ret.iloc[idx_test, :]

        if 'cv_cache' not in params or len(params['cv_cache']) <= params['cv_iteration']:
            if 'cv_cache' not in params:
                params['cv_cache'] = {}
            cvdata = {}
            cvdata['X'] = regcov(r)
            cvdata['y'] = np.mean(r, axis=0)
            cvdata['X_test'] = regcov(r_test)
            cvdata['y_test'] = np.mean(r_test, axis=0)
            
            if params['objective'] in {'GLS', 'GLSR2', 'SRexpl'}:
                cvdata['invX'] = np.linalg.pinv(cvdata['X'])
                cvdata['invX_test'] = np.linalg.pinv(cvdata['X_test'])

            params['cv_cache'][params['cv_iteration']] = cvdata

        cvdata = params['cv_cache'][params['cv_iteration']]
        X = cvdata['X']
        y = cvdata['y']
        X_test = cvdata['X_test']
        y_test = cvdata['y_test']
        
        if params['objective'] in {'GLS', 'GLSR2', 'SRexpl'}:
            invX = cvdata['invX']
            invX_test = cvdata['invX_test']

        phi, params = FUN(X, y, params)[0:2]

        if 'cache_run' not in params or not params['cache_run']:
            if 'cv_phi' not in params:
                params['cv_phi'] = {}
            params['cv_phi'][params['cv_iteration']] = phi
            if 'cv_MVE' not in params:
                params['cv_MVE'] = {}
            params['cv_MVE'][params['cv_iteration']] = np.dot(r_test, phi)
            
            fact = np.dot(X, phi)
            fact_test = np.dot(X_test, phi)

            if params['ignore_scale']:
                b = np.linalg.lstsq(fact, y, rcond=None)[0]
                b_test = np.linalg.lstsq(fact_test, y_test, rcond=None)[0]
            else:
                b = 1
                b_test = 1

            res = [
                def_bootstrp_obj(np.dot(fact, b), y, invX, phi, r, params),
                def_bootstrp_obj(np.dot(fact_test, b_test), y_test, invX_test, phi, r_test, params)
            ]

    return np.hstack(res), params

def cvpartition_contiguous(n, k):
    """
    Create contiguous partitions for cross-validation.
    
    Parameters:
    - n: int, total number of data points
    - k: int, number of folds/partitions
    
    Returns:
    - indices: list of lists, containing indices for each fold
    """
    s = n // k  # using floor division to ensure integer result
    indices = [None] * k  # Pre-allocating list with k None elements
    
    for i in range(k - 1):
        # Using range indexing to create contiguous partitions
        indices[i] = list(range(s * i, s * (i + 1)))
    
    # Last partition takes the remaining elements
    indices[k - 1] = list(range(s * (k - 1), n))
    
    return indices

def bootstrp_obj_CSR2(y_hat, y, invX, phi, r, params):
    """
    Compute the objective based on the Coefficient of Squared Regression (CSR2).

    Parameters:
    - y_hat: Predicted values
    - y: Actual values
    - invX, phi, r, params: Other parameters that are not used in the computation
      in this function but are kept for consistency with other objective functions.
    
    Returns:
    - obj: The computed CSR2 objective value.
    """
    # Compute the CSR2 objective
    obj = 1 - (np.dot((y_hat - y).T, (y_hat - y))) / (np.dot(y.T, y))
    return obj

def l2_shrinkage(mu, Sigma, kappa, T):
    """
    Exact penalty scaling from the replication repo (lukaskoerber version).
    This matches how they compute L2pen = kappa2pen(...)
    """
    tau = np.trace(Sigma)
    freq = 252
    
    # This is the repo's kappa2pen formula (see SCS_L2est.py)
    gamma = freq * tau / T / (kappa ** 2)
    
    # Small safety addition (repo uses similar jitter)
    Sigma_reg = Sigma + gamma * np.eye(len(mu)) + 1e-10 * np.eye(len(mu))
    
    # Ridge solve (repo uses pinv or equivalent)
    b = pinv(Sigma_reg) @ mu
    print(f"kappa={kappa:.4f} | gamma={gamma:.8f} ")
    return b

def cross_sectional_r2(mu, Sigma, b):
    pred = Sigma @ b
    mu_demean = mu - np.mean(mu)  # add this to match if repo centers
    pred_demean = pred - np.mean(pred)
    ss_res = np.sum((mu_demean - pred_demean)**2)
    ss_tot = np.sum(mu_demean**2) + 1e-10
    r2 = 1 - ss_res / ss_tot
    return max(0.0, min(1.0, r2))




def estimate(data, n_folds=5, kappa_min=1e-3, kappa_max=3.0, num_kappa=100):
    """
    Replicate Figure 2(a) style plot with pure L2 shrinkage.
    """
    # Extract portfolio returns (assume already excess & orthogonalized if needed)
    port_cols = [col for col in data.columns if col not in ['Date', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Rm']]
    returns = data[port_cols].dropna()
    
    F = returns.values
    T, H = F.shape
    print(f"Data shape: {T} time periods × {H} portfolios")

    # Full-sample moments (used for IS and for pricing in CV)
    Sigma_full = np.cov(F, rowvar=False) + 1e-10 * np.eye(H)  # repo style
    mu_full    = np.mean(F, axis=0)
    print("\n--- Data Diagnostics ---")
    print(f"Number of time periods T: {T}")
    print(f"Number of portfolios H: {H}")
    print(f"trace(Sigma_full): {np.trace(Sigma_full):.6f}")
    print(f"average eigenvalue: {np.trace(Sigma_full) / H:.6f}")
    print(f"max abs(mean return): {np.max(np.abs(mu_full)):.6f}")
    print(f"min abs(mean return): {np.min(np.abs(mu_full)):.6f}")
    print(f"std of means: {np.std(mu_full):.6f}")
    print(f"Data frequency guess: {'daily' if T > 5000 else 'monthly or lower'}")
    
    # Repo-like dynamic grid (starts from 0.01)
    kappa_grid = np.logspace(np.log10(0.01), np.log10(1), num_kappa)
    print(f"kappa grid: {len(kappa_grid)} points from {kappa_grid[0]:.4e} to {kappa_grid[-1]:.4f}")

    # Precompute tol for pinv (repo style)
    tol = max(Sigma_full.shape) * np.finfo(float).eps * np.linalg.norm(Sigma_full, np.inf)

    # ── Contiguous chronological folds ──
    fold_size = T // n_folds
    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = T if i == n_folds - 1 else test_start + fold_size
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, T)])
        folds.append((train_idx, test_idx))

    # Storage
    oos_r2 = np.zeros(len(kappa_grid))
    oos_se = np.zeros(len(kappa_grid))
    ins_r2 = np.zeros(len(kappa_grid))

    for i, kappa in enumerate(kappa_grid):
        fold_r2_values = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if len(test_idx) < 10:
                continue
                
            train_ret = returns.iloc[train_idx]
            test_ret = returns.iloc[test_idx]
            
            mu_train = train_ret.mean().values
            Sigma_train = np.cov(train_ret, rowvar=False) + 1e-10 * np.eye(H)
            
            safe_kappa = max(kappa, 1e-6)
            
            tau_fold = np.trace(Sigma_train)
            freq = 252 if T > 5000 else 12  # auto-detect
            gamma_base = freq * tau_fold / len(train_idx) / (safe_kappa ** 2)
            
            cv_adjust = 1 / (1 - 1/n_folds)  # repo's lCV adjustment (stronger for OOS)
            gamma_cv = gamma_base * cv_adjust
            
            Sigma_reg_cv = Sigma_train + gamma_cv * np.eye(H)
            
            b = np.linalg.pinv(Sigma_reg_cv, rcond=tol) @ mu_train
            
            r2_fold = cross_sectional_r2(test_ret.mean().values, Sigma_full, b)
            fold_r2_values.append(r2_fold)
            
            if fold_idx == 0 and i % 20 == 0:
                print(f"kappa={kappa:.4f} | gamma_base={gamma_base:.8f} | gamma_cv={gamma_cv:.8f} | fold R2={r2_fold:.4f}")
        
        if len(fold_r2_values) > 0:
            oos_r2[i] = np.mean(fold_r2_values)
            oos_se[i] = np.std(fold_r2_values) / np.sqrt(len(fold_r2_values))
        else:
            oos_r2[i] = np.nan
            oos_se[i] = np.nan
        
        b_full = l2_shrinkage(mu_full, Sigma_full, kappa, T)
        ins_r2[i] = cross_sectional_r2(mu_full, Sigma_full, b_full)
        
        if i % 20 == 0:
            print(f"kappa={kappa:.4f} | OOS mean R²={oos_r2[i]:.4f} | IS R²={ins_r2[i]:.4f}")

    peak_idx = np.nanargmax(oos_r2)
    peak_kappa = kappa_grid[peak_idx]
    peak_r2 = oos_r2[peak_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(kappa_grid, ins_r2, 'k--', linewidth=2, label='In-sample R²')
    plt.plot(kappa_grid, oos_r2, 'b-', linewidth=2.5, label='OOS CV R²')
    plt.fill_between(kappa_grid,
                     oos_r2 - oos_se,
                     oos_r2 + oos_se,
                     color='blue', alpha=0.15, label='±1 SE')
    
    plt.xscale('log')
    plt.xlim(kappa_min, kappa_max)
    plt.ylim(0, max(0.6, np.nanmax(ins_r2) * 1.1))
    plt.xlabel('κ  (root expected SR² under prior)')
    plt.ylabel('Cross-sectional R²')
    plt.title(f'Pure L2 Shrinkage — From Scratch portfolios (n_folds={n_folds})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Mark optimal point
    plt.axvline(peak_kappa, color='gray', linestyle=':', alpha=0.7)
    plt.text(peak_kappa * 1.1, peak_r2, f'Peak: κ={peak_kappa:.3f}, R²={peak_r2:.3f}',
             fontsize=10, color='darkblue')
    
    plt.tight_layout()
    plt.show()

    print(f"\nOptimal kappa: {peak_kappa:.4f}")
    print(f"Max OOS CV R²: {peak_r2:.4f}")
    return peak_kappa, peak_r2, oos_r2, ins_r2