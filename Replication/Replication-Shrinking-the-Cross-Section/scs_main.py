import numpy as np
import pandas as pd
from datetime import datetime
import os
from load_managed_portfolios import load_managed_portfolios
from SCS_L2est import SCS_L2est
import load_emerging

# Options
#dataprovider = 'ff25'
daily = True
interactions = False
rotate_PC = False
withhold_test_sample = False

#dataprovider = 'anom_50'
dataprovider = 'emerging_mkt'

# Sample dates
t0 = datetime(1926, 7, 1)
tN = datetime(2017, 12, 31)

oos_test_date = datetime.strptime('01JAN2005', '%d%b%Y')

# Current run folder
run_folder = datetime.today().strftime('%d%b%Y').upper() + "/"

# Paths
projpath = ''
datapath = os.path.join(projpath, 'Data')
instrpath = os.path.join(datapath, 'Instruments')

# Initialize
if daily:
    freq = 252
    suffix = '_d'
    date_fmt = '%m/%d/%Y'
else:
    freq = 12
    suffix = ''
    date_fmt = '%m/%Y'

np.random.seed(0)

# Default estimation parameters
default_params = {
    'gridsize': 100,
    'contour_levelstep': 0.01,
    'objective': 'CSR2',
    'rotate_PC': False,
    'devol_unconditionally': False,
    'kfold': 3,
    'plot_dof': True,
    'plot_coefpaths': True,
    'plot_objective': True,
    'fig_options': {'fig_sizes': ['width=half'], 'close_after_print': True}
}

# Parameters setup
p = default_params

if interactions:
    p['kfold'] = 2
else:
    p['gridsize'] = 100

if withhold_test_sample:
    p['oos_test_date'] = oos_test_date

# Process original ff25 portfolios if requested
if dataprovider == 'ff25':
    print("=== ENTERED FF25 BLOCK ===")   # ← add this
    from load_ff25 import load_ff25
    from datetime import datetime

    datapath = os.path.join(projpath, 'Data') + '/'
    print(f"Loading from: {datapath}")     # ← add this

    dd, re, mkt, DATA, labels = load_ff25(
        datapath=datapath,
        daily=daily,
        t0=t0,
        tN=tN
    )

    print(f"Loaded {re.shape[1]} portfolios")  # should print 25
    print(f"Date range: {dd.min()} to {dd.max()}")
    print("Sample portfolio names:", labels[:5])  # first 5 names

    anomalies = labels

    if withhold_test_sample:
        p['oos_test_date'] = oos_test_date

    # print("Starting estimation...")
    p = SCS_L2est(dd, re, mkt, freq, anomalies, p)
    # print("Estimation finished.")
    if not interactions:
        pass
        # Assuming functions have been translated
        # dd, re, mkt, DATA, labels = load_ff25(datapath, daily, 0, tN)
        # Followed by processing and estimation logic as in MATLAB

elif dataprovider == 'emerging_mkt':
    print("=== ENTERED EMERGING MARKETS BLOCK ===")
    
    datapath = os.path.join(projpath, 'Data') + '/'
    print(f"Loading from: {datapath}")
    
    # Local dates for emerging (starts 1990, extend to future)
    emerging_t0 = datetime(1990, 1, 1)
    emerging_tN = datetime(2025, 12, 31)  # or datetime.now()
    
    dd, re, mkt, DATA, labels = load_emerging.load_emerging_mkt(
        datapath=datapath,
        daily=daily,
        t0=emerging_t0,   # ← use local
        tN=emerging_tN    # ← use local
    )
    
    anomalies = labels
    
    freq = 12
    
    print(f"Passing to SCS_L2est: {len(anomalies)} portfolios, {len(dd)} dates")
    p = SCS_L2est(dd, re, mkt, freq, anomalies, p)
    print("Estimation finished.")



else:
    # Managed portfolios
    fmask = os.path.join(instrpath, f"managed_portfolios_{dataprovider}{suffix}_*.csv")

    flist = os.listdir(instrpath)
    print("flist:", flist)
    print("list dir:", os.listdir(instrpath))
    filename = os.path.join(instrpath, flist[0].strip())
    # Followed by data loading and estimation as in MATLAB

    p['L1_truncPath'] = True

    if interactions:  # use interactions
        dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, [])
        p = SCS_L2est(dd, re, mkt, freq, anomalies, p)
    else:  # use only raw characteristics (no derived instruments)
        # load data
        dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, ['rX_', 'r2_', 'r3_'])
        
        # estimate
        p = SCS_L2est(dd, re, mkt, freq, anomalies, p)

