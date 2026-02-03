import pandas as pd
from datetime import datetime

"""
preparing data for analysis 
reads in two files: FF25_data.csv includes daily returns from 25 portfolios 
F-F_research_data_factors_daily.csv includes the risk free rate


data website: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 


Portfolio names: 
MarketCap BookMarket Ratio 
- example: SMALL LoBM means small market cap, low (book value)/(market value)

Factors File: 
"""

ff25_datapath = "/Users/madisonpoore/Desktop/FF25_Data.csv"
factor_datapath = "/Users/madisonpoore/Desktop/F-F_research_data_factors_daily.csv"


def load_ff25_returns_only(path, nrows=26149, skiprows=19):
    """
    function loading ff25 returns (NO FACTOR DATA)
    """
    #loading
    df = pd.read_csv(path, skiprows=skiprows, nrows=nrows - skiprows)

    #labels & processing
    df = df.rename(columns={"Unnamed: 0": "Date"})

    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")

    #transforms to decimal form (3% -> 0.03)
    ret_cols = df.columns.drop("Date")
    df[ret_cols] = df[ret_cols].astype(float) / 100

    return df.reset_index(drop=True)
df = load_ff25_returns_only(ff25_datapath)

# same processing steps as before
skip_fact = 3
factors = pd.read_csv(factor_datapath, skiprows=skip_fact, low_memory=False)
factors = factors.rename(columns={"Unnamed: 0": "Date"})
factors = factors[pd.to_numeric(factors["Date"], errors="coerce").notna()]
factors["Date"] = pd.to_datetime(factors["Date"], format="%Y%m%d")

for col in factors.columns[1:]:
    factors[col] = factors[col].astype(float) / 100


df = df.merge(factors[["Date", "RF", "Mkt-RF"]], on="Date", how="inner")



"""
We start with an application of our proposed method to
daily returns on the 25 Fama-French ME/BM-sorted (FF25)
portfolios from July 1926 to December2017, which we 
orthogonalize with respect to the Center for Research
in Security Prices (CRSP) value-weighted index return 
using βs estimated in the full sample. (p280)
"""

port_cols = df.columns.drop(['Date', 'RF', 'Mkt-RF'])
ex_ret = df[port_cols].sub(df['RF'], axis=0)
mkt_ex = df['Mkt-RF']
