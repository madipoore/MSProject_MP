import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import estimatorsv2 as est
from datetime import datetime
import os 

#initial choices
data_choice = "Replication"          # or "Replication" or "Scratch"
daily = True 
t0 = datetime(1926,7,1)
tN = datetime(2017,12,31)
oos_test_date = datetime.strptime('01Jan2005','%d%b%Y')
np.random.seed(0)


if data_choice == "Replication":
    #uses data from data folder (specifically, the 25 portfolio dataset)
    datapath = "/Users/madisonpoore/Desktop/MSProject/Data/"
    from load25 import load_ff25
    run_folder = datetime.today().strftime('%d%b%Y').upper() + "/"
    projpath = ''
    datapath = os.path.join(projpath, 'Data')
    instrpath = os.path.join(datapath, 'instruments')

    if daily:
        freq = 252
        suffix = '_d'
        date_fmt = '%m/%d/%Y'
    else:
        freq = 12
        suffix = ''
        date_fmt = '%m/%Y'
    
    default = {
    'gridsize': 100,
    'contour_levelstep': 0.01,
    'objective': 'CSR2',
    'rotate_PC': False,
    'devol_unconditionally': False,
    'kfold': 3,
    'plot_dof': True,
    'plot_coefpaths': True,
    'plot_objective': True,
    'fig_options': {'fig_sizes': ['width=half'], 'close_after_print': True}}

    p = default
    datapath = "/Users/madisonpoore/Desktop/MSProject/Data/"
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

    print("starting estimate")
    p = est.l2est(dd, re, mkt, freq, anomalies, p)
    print("finished estimate")
elif data_choice == "Scratch":
    #data that i processed in "ff25_data_download.py", which isn't used in replication folder
    FILE_PATH = "/Users/madisonpoore/Desktop/ff25_orth_ex_scaled.pkl"
    df_full = pd.read_pickle(FILE_PATH)
    peak_kappa, peak_r2, oos_r2, ins_r2 = est.estimate(df_full, n_folds=5)

elif data_choice == "50_anomaly":
    #data from Data folder, under "Instruments" folder
    fifty_daily_path = "/Users/madisonpoore/Desktop/MSProject/Data/Instruments/managed_portfolios_anom_d_50.csv"
    df_50 = pd.read_csv(fifty_daily_path)
    df_50['date'] = pd.to_datetime(df_50['date'], format='%m/%d/%Y')
    df_50 = df_50.set_index('date')
    df_50 = df_50.dropna(how='all', axis=1).dropna(how='all', axis=0)
    df_full = df_50
    print("50 anomaly portfolios shape:", df_50.shape)
    print("Date range:", df_50.index.min().strftime('%Y-%m-%d'), "to", df_50.index.max().strftime('%Y-%m-%d'))
    print("First 5 columns:", df_50.columns[:5].tolist())

else: 
    print("invalid data choice")
    print("Options: Scratch, Replication, 50_anomaly")