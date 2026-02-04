import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats

"""
preparing data for analysis 
reads in two files: FF25_data.csv includes daily returns from 25 portfolios 
F-F_research_data_factors_daily.csv includes the risk free rate


data website: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 


Portfolio naming: 
MarketCap BookMarket Ratio 
- example: SMALL LoBM means small market cap, low (book value)/(market value)

Factors File: 
RF: 1-month T Bill Rate (aka, risk free rate)
Mkt-RF: Market rate - risk free rate

Financial Variables: 
beta: coefficient of volatility. that is, when the market changes, how much does a portfolio change? 
- e.g., beta = 1.3 indicates the portfolio moves 30% more than the market (or RF rate here)
    - **portfolio return = alpha + beta * market return + epsilon**
"""

ff25_datapath = "/Users/madisonpoore/Desktop/FF25_Data.csv"
factor_datapath = "/Users/madisonpoore/Desktop/F-F_research_data_factors_daily.csv"
start_date = pd.to_datetime('1926-07-01')
end_date   = pd.to_datetime('2017-12-31')

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

#putting together both dataframes
df = df.merge(factors[["Date", "RF", "Mkt-RF","SMB", "HML"]], on="Date", how="inner")
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)
#print(df.tail())
print(df.head())
"""
We start with an application of our proposed method to
daily returns on the 25 Fama-French ME/BM-sorted (FF25)
portfolios from July 1926 to December2017, which we 
orthogonalize with respect to the Center for Research
in Security Prices (CRSP) value-weighted index return 
using βs estimated in the full sample. (p280)
"""

#names of 25 portfolios
port_cols = df.columns.drop(['Date', 'RF', 'Mkt-RF',"SMB", "HML"])

#excess returns for portfolios (returns - RF)
ex_ret = df[port_cols].sub(df['RF'], axis=0)

#market excess returns
mkt_ex = df['Mkt-RF']

#defining market return "Mkt" from the "RF" variable and "Mkt-RF"
df['Rm'] = df['Mkt-RF'] + df['RF']

#need 25 betas for OLS regression
def calc_beta(asset_returns, market_returns):
    """Super simple beta: cov(asset, market) / var(market)"""
    cov = np.cov(asset_returns, market_returns, ddof=1)[0, 1]
    var_mkt = np.var(market_returns, ddof=1)
    return cov / var_mkt if var_mkt != 0 else np.nan

#OLS regression
def simple_beta(portfolio, market):
    slope, intercept, r_value, p_value, std_err = stats.linregress(market, portfolio)
    return slope  # this is your beta
betas = df[port_cols].apply(lambda col: simple_beta(col, df['Rm']))
print(betas.round(4))