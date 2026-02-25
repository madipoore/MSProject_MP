import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
import sklearn

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
#print(df.head())


"""
In our specification, we focus on under-
standing the factors that help explain these cross-sectional
differences, and we do not explicitly include a market fac-
tor, but we orthogonalize the characteristics-based factors
with respect to the market factor (p275)

orthogonalize raw returns
"""

#calculating market returns 
df['Rm'] = df['Mkt-RF'] + df['RF']

#orthogonalizing returns
port_cols = [col for col in df.columns 
             if col not in ['Date', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Rm']]
orth_raw = df[port_cols].copy()
#initializes betas object
betas = pd.Series(index=port_cols, dtype=float)
for col in port_cols:
    #OLS regression fit 
    slope, _, _, _, _ = stats.linregress(df['Rm'], df[col])
    betas[col] = slope
#print(betas.head())
for col in port_cols:
    orth_raw[col] = df[col] - betas[col] * df['Rm']

#tests for orthogonality... should be really small cov vals 
# for col in port_cols[:5]:
#     cov = np.cov(orth_raw[col], df['Rm'], ddof=1)[0, 1]
#     print(f"{col:12} cov = {cov:.2e}")

#rescaling to std(Mkt-RF)
market_ex_std = df['Mkt-RF'].std(ddof=1)
orth_scaled_raw = orth_raw.multiply(
    market_ex_std / orth_raw.std(ddof=1),
    axis=1
)
print("Std dev after rescaling:")
print(orth_scaled_raw.iloc[:, :5].std(ddof=1))
orth_ex_scaled = orth_scaled_raw.subtract(df['RF'], axis=0)


full_df = pd.concat([
    df[['Date', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Rm']],  # keep original factors + dates
    orth_ex_scaled                                      # add the 25 processed columns
], axis=1)


#export to pk1 file: 
#full_df.to_pickle("/Users/madisonpoore/Desktop/ff25_orth_ex_scaled.pkl")