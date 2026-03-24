import pandas as pd
from datetime import datetime

def load_qqq2(datapath, daily, t0=None, tN=None):
    """
    Load the QQQ 100 quintile portfolios (no momentum version)
    """
    csv_file = "qqq_100_portfolios_quintiles.csv"   # put the exact filename here
    
    full_path = datapath + csv_file if not datapath.endswith('/') else datapath + csv_file
    
    print(f"Loading QQQ portfolios from: {full_path}")
    
    DATA = pd.read_csv(full_path)
    
    # Your CSV uses 'month_end' as the first column
    DATA = DATA.rename(columns={DATA.columns[0]: 'month_end'})
    DATA['month_end'] = pd.to_datetime(DATA['month_end'], errors='coerce')
    DATA = DATA.dropna(subset=['month_end'])
    
    # Optional: start from a clean date (highly recommended)
    start_date = pd.to_datetime('2011-01-01')   # change to '2010-06-01' if you want more months
    DATA = DATA[DATA['month_end'] >= start_date]
    
    # All portfolio columns (should be the remaining 100 columns)
    portfolio_cols = [col for col in DATA.columns if col != 'month_end']
    
    # Convert returns to decimal (some files store as percent)
    for col in portfolio_cols:
        DATA[col] = pd.to_numeric(DATA[col], errors='coerce') / 100.0   # remove /100 if already in decimal
    
    DATA = DATA.dropna(subset=portfolio_cols)   # drop any remaining bad rows
    
    dates = DATA['month_end'].reset_index(drop=True)
    re = DATA[portfolio_cols].reset_index(drop=True)
    mkt = re.mean(axis=1)
    
    labels = portfolio_cols
    
    print(f"Loaded {re.shape[1]} portfolios over {len(dates)} months")
    print(f"Date range: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
    print(f"Final ret shape: {re.shape}")
    print("Sample head:\n", re.head())
    
    return dates, re, mkt, DATA, labels