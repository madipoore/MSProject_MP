import pandas as pd
from datetime import datetime

def load_qqq2(datapath, daily, t0=None, tN=None):
    csv_file = "qqq_new_portfolios.csv"
    
    full_path = datapath + csv_file if not datapath.endswith('/') else datapath + csv_file
    
    print(f"Loading QQQ portfolios from: {full_path}")
    
    DATA = pd.read_csv(full_path)
    DATA = DATA.rename(columns={DATA.columns[0]: 'month_end'})
    DATA['month_end'] = pd.to_datetime(DATA['month_end'], errors='coerce')
    DATA = DATA.dropna(subset=['month_end'])
    
    start_date = pd.to_datetime('2011-01-01')
    DATA = DATA[DATA['month_end'] >= start_date]
    
    portfolio_cols = [col for col in DATA.columns if col != 'month_end']
    
    # portfolio_cols = [
    #     'Size_Quintile3','BM_Quintile3','ProfitGPM_Quintile3','AssetGrowth_Quintile3',
    #     'SalesGrowth_Quintile3','EVtoEBITDA_Quintile3','ROE_Quintile3','ROE_Quintile4',
    #     'ROA_Quintile3','PS_Quintile3','EVSales_Quintile3','FCFYield_Quintile3',
    #     'EPSGrowth_Quintile3','OCFGrowth_Quintile3','NetMargin_Quintile3',
    #     'OpMargin_Quintile3','EBITDAMargin_Quintile3','DebtEquity_Quintile3',
    #     'CurrentRatio_Quintile3','EarnYield_Quintile3','ROIC_Quintile3'
    # ]
    for col in portfolio_cols:
        DATA[col] = pd.to_numeric(DATA[col], errors='coerce') / 100.0
    
    DATA = DATA.dropna(subset=portfolio_cols)
    
    dates = DATA['month_end'].reset_index(drop=True)
    re = DATA[portfolio_cols].reset_index(drop=True)
    mkt = re.mean(axis=1)
    
    labels = portfolio_cols
    
    print(f"Loaded {re.shape[1]} portfolios over {len(dates)} months")
    print(f"Date range: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
    print(f"Final ret shape: {re.shape}")
    print("Sample head:\n", re.head())
    
    return dates, re, mkt, DATA, labels