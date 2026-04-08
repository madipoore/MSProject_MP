import numpy as np
import pandas as pd

# ==============================================
# CONFIG
# ==============================================
INPUT_FILE = "/Users/madisonpoore/Desktop/MSProject/Replication/Replication-Shrinking-the-Cross-Section/Data/qqq_monthly_1997_2017.csv"
OUTPUT_FILE = "qqq_longpanel_new.csv"

# ==============================================
# 1. Load and prepare raw data
# ==============================================
print("Loading raw QQQ data...")
df = pd.read_csv(INPUT_FILE)

df['month_end'] = pd.to_datetime(df['month_end'])
df['month_end'] = df['month_end'] + pd.offsets.MonthEnd(0)

print(f"Full range: {df['month_end'].min().date()} -- {df['month_end'].max().date()} ({len(df):,} rows)")

df = df.sort_values(['symbol', 'month_end']).reset_index(drop=True)

# ==============================================
# 2. Select 19 strong factors + ROIC (no momentum)
# ==============================================
core_cols = [
    'symbol', 'month_end',
    'monthly_return', 'close', 'volume',
    'key_metrics_marketCap',
]

pred_raw_cols = [
    'balance_totalStockholdersEquity',          # B/M
    'ratios_grossProfitMargin',                 # profitability
    'growth_assetGrowth',                       # investment
    'growth_revenueGrowth',                     # sales growth
    'key_metrics_evToEBITDA',                   # valuation
    'key_metrics_returnOnEquity',               # ROE
    'key_metrics_returnOnAssets',               # ROA
    'ratios_priceToSalesRatio',                 # P/S
    'key_metrics_evToSales',                    # EV/Sales
    'key_metrics_freeCashFlowYield',            # FCF yield
    'growth_epsgrowth',                         # EPS growth
    'growth_operatingCashFlowGrowth',           # OCF growth
    'ratios_netProfitMargin',                   # net margin
    'ratios_operatingProfitMargin',             # op margin
    'ratios_ebitdaMargin',                      # EBITDA margin
    'ratios_debtToEquityRatio',                 # leverage
    'ratios_currentRatio',                      # liquidity
    'key_metrics_earningsYield',                # earnings yield
    'key_metrics_returnOnInvestedCapital',      # ROIC ← replacement
]

# Only keep existing columns
all_needed = core_cols + pred_raw_cols
keep_cols = [col for col in all_needed if col in df.columns]
df_slim = df[keep_cols].copy()

print(f"Kept {len(keep_cols)} columns")
print("Columns:", keep_cols)

# ==============================================
# 3. Basic cleaning & excess returns
# ==============================================
df_clean = df_slim[
    (df_slim['close'] > 1) &
    (df_slim['key_metrics_marketCap'] > 0) &
    (df_slim['volume'] > 0) &
    (df_slim['monthly_return'].notna()) &
    (df_slim['monthly_return'].between(-2, 5))
].copy()

df_clean = df_clean.dropna(subset=['key_metrics_marketCap', 'monthly_return'])

print("Shape after basic filters:", df_clean.shape)

df_clean['mktcap_lag'] = df_clean.groupby('symbol')['key_metrics_marketCap'].shift(1)
df_clean = df_clean.dropna(subset=['mktcap_lag'])

print("Shape after mktcap_lag:", df_clean.shape)

# Merge RF
rf_path = '/Users/madisonpoore/Desktop/MSProject/Replication/Replication-Shrinking-the-Cross-Section/Data/F-F_Research_Data_Factors.csv'
rf_df = pd.read_csv(rf_path, skiprows=3, header=None)
rf_df = rf_df.iloc[:, :5]
rf_df.columns = ['date_raw', 'mkt_rf', 'smb', 'hml', 'rf']
rf_df['date_raw'] = pd.to_datetime(rf_df['date_raw'], format='%Y/%m/%d', errors='coerce')
rf_df = rf_df.dropna(subset=['date_raw'])
rf_df['month_end'] = rf_df['date_raw'] + pd.offsets.MonthEnd(0)
rf_df['rf'] = rf_df['rf'] / 100.0
rf_df = rf_df[['month_end', 'rf']]

df_clean = df_clean.merge(rf_df, on='month_end', how='left')
df_clean['rf'] = df_clean['rf'].ffill().fillna(0.0001)
df_clean['ret_exc'] = df_clean['monthly_return'] - df_clean['rf']
df_clean = df_clean.dropna(subset=['ret_exc'])

print("Shape after RF & ret_exc:", df_clean.shape)

# ==============================================
# 4. Forward-fill + compute 19 predictors + ROIC
# ==============================================
ffill_cols = [col for col in pred_raw_cols if col in df_clean.columns]
for col in ffill_cols:
    df_clean[col] = df_clean.groupby('symbol')[col].ffill()

df_clean['size']              = np.log(df_clean['mktcap_lag'])
df_clean['bm']                = df_clean['balance_totalStockholdersEquity'] / df_clean['mktcap_lag']
df_clean['profit_gpm']        = df_clean['ratios_grossProfitMargin']
df_clean['asset_growth']      = df_clean['growth_assetGrowth']
df_clean['sales_growth']      = df_clean['growth_revenueGrowth']
df_clean['ev_ebitda']         = df_clean['key_metrics_evToEBITDA']
df_clean['roe']               = df_clean['key_metrics_returnOnEquity']
df_clean['roa']               = df_clean['key_metrics_returnOnAssets']
df_clean['ps_ratio']          = df_clean['ratios_priceToSalesRatio']
df_clean['ev_sales']          = df_clean['key_metrics_evToSales']
df_clean['fcf_yield']         = df_clean['key_metrics_freeCashFlowYield']
df_clean['eps_growth']        = df_clean['growth_epsgrowth']
df_clean['ocf_growth']        = df_clean['growth_operatingCashFlowGrowth']
df_clean['net_margin']        = df_clean['ratios_netProfitMargin']
df_clean['op_margin']         = df_clean['ratios_operatingProfitMargin']
df_clean['ebitda_margin']     = df_clean['ratios_ebitdaMargin']
df_clean['debt_equity']       = df_clean['ratios_debtToEquityRatio']
df_clean['current_ratio']     = df_clean['ratios_currentRatio']
df_clean['earn_yield']        = df_clean['key_metrics_earningsYield']
df_clean['roic']              = df_clean['key_metrics_returnOnInvestedCapital']

# No momentum — that's the point

# ==============================================
# 5. Imputation & winsorizing
# ==============================================
pred_cols = [
    'size', 'bm', 'profit_gpm', 'asset_growth', 'sales_growth',
    'ev_ebitda', 'roe', 'roa', 'ps_ratio', 'ev_sales', 'fcf_yield',
    'eps_growth', 'ocf_growth', 'net_margin', 'op_margin', 'ebitda_margin',
    'debt_equity', 'current_ratio', 'earn_yield', 'roic'
]

for col in pred_cols:
    df_clean[col] = df_clean.groupby('symbol')[col].transform(lambda x: x.fillna(x.median()))
    df_clean['year'] = df_clean['month_end'].dt.year
    df_clean[col] = df_clean.groupby(['symbol', 'year'])[col].transform(lambda x: x.fillna(x.median()))
    df_clean = df_clean.drop(columns=['year'], errors='ignore')
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Winsorize
for col in pred_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(
            df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)
        )

# Final clean
df_predictors = df_clean.dropna(subset=pred_cols + ['ret_exc', 'mktcap_lag'])

final_cols = ['month_end', 'symbol', 'ret_exc', 'mktcap_lag'] + pred_cols
df_final = df_predictors[final_cols].copy()

df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved long panel with 19 factors + ROIC (no momentum) to: {OUTPUT_FILE}")
print(f"Shape: {df_final.shape} (stock-months)")
print(f"Unique months: {df_final['month_end'].nunique()}")
print(f"Unique symbols: {df_final['symbol'].nunique()}")
print("\nNaN check:\n", df_final[pred_cols].isna().sum())
print("\nHead:\n", df_final.head(10))