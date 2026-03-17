import numpy as np
import pandas as pd

df = pd.read_csv("/Users/madisonpoore/Desktop/MSProject/Replication/Replication-Shrinking-the-Cross-Section/Data/qqq_monthly_dataset.csv")

df['month_end'] = pd.to_datetime(df['month_end'])
df['month_end'] = df['month_end'] + pd.offsets.MonthEnd(0)

end_date = pd.to_datetime('2017-12-31')
df = df[df['month_end'] <= end_date]
print(f"Cropped to {df['month_end'].min().date()} -- {df['month_end'].max().date()} ({len(df)} rows)")

df = df.sort_values(['symbol', 'month_end']).reset_index(drop=True)

required_cols = [
    'symbol', 'month_end',
    'monthly_return', 'close', 'volume',
    'key_metrics_marketCap',
    'balance_totalStockholdersEquity',
    'ratios_grossProfitMargin',
    'growth_assetGrowth',
    'growth_revenueGrowth',
    'key_metrics_evToEBITDA'
]
keep_cols = [col for col in required_cols if col in df.columns]
df_slim = df[keep_cols].copy()

print("Columns kept:", df_slim.columns.tolist())
print("Shape after column selection:", df_slim.shape)

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

print("Shape after mktcap_lag dropna:", df_clean.shape)

rf_path = '/Users/madisonpoore/Desktop/MSProject/Replication/Replication-Shrinking-the-Cross-Section/Data/F-F_Research_Data_Factors.csv'
rf_df = pd.read_csv(rf_path, skiprows=3, header=None)

print("\nRaw RF head (first 10 rows):\n", rf_df.head(10).to_string(index=False))

rf_df = rf_df.iloc[:, :5]
rf_df.columns = ['date_raw', 'mkt_rf', 'smb', 'hml', 'rf']

rf_df['date_raw'] = pd.to_datetime(rf_df['date_raw'], format='%Y/%m/%d', errors='coerce')
rf_df = rf_df.dropna(subset=['date_raw'])

rf_df['month_end'] = rf_df['date_raw'] + pd.offsets.MonthEnd(0)
rf_df['rf'] = rf_df['rf'] / 100.0
rf_df = rf_df[['month_end', 'rf']]

print("RF range after cleaning:", rf_df['month_end'].min().date(), "to", rf_df['month_end'].max().date())
print("Number of RF rows:", len(rf_df))

# Merge
df_clean = df_clean.merge(rf_df, on='month_end', how='left')

df_clean['rf'] = df_clean['rf'].ffill().fillna(0.0001)
df_clean['ret_exc'] = df_clean['monthly_return'] - df_clean['rf']

df_clean = df_clean.dropna(subset=['ret_exc'])

print("Shape after RF merge and ret_exc dropna:", df_clean.shape)

ffill_cols = ['balance_totalStockholdersEquity', 'ratios_grossProfitMargin',
              'growth_assetGrowth', 'growth_revenueGrowth', 'key_metrics_evToEBITDA']

for col in ffill_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean.groupby('symbol')[col].ffill()

df_clean['size'] = np.log(df_clean['mktcap_lag'])
df_clean['bm'] = df_clean['balance_totalStockholdersEquity'] / df_clean['mktcap_lag']
df_clean['profit'] = df_clean['ratios_grossProfitMargin']
df_clean['asset_growth'] = df_clean['growth_assetGrowth']
df_clean['sales_growth'] = df_clean['growth_revenueGrowth']
df_clean['ev_ebitda'] = df_clean['key_metrics_evToEBITDA']

df_clean['mom12m'] = df_clean.groupby('symbol')['monthly_return'].transform(
    lambda x: x.rolling(11, min_periods=6).sum().shift(1)
)

lag_period = 0
lag_cols = ['bm', 'profit', 'asset_growth', 'sales_growth', 'ev_ebitda']
for col in lag_cols:
    df_clean[f'{col}_lag'] = df_clean.groupby('symbol')[col].shift(lag_period)

for col in [f'{c}_lag' for c in lag_cols]:
    df_clean[col] = df_clean.groupby('symbol')[col].transform(lambda x: x.fillna(x.median()))
df_clean[[f'{c}_lag' for c in lag_cols]] = df_clean[[f'{c}_lag' for c in lag_cols]].fillna(
    df_clean[[f'{c}_lag' for c in lag_cols]].median()
)

for col in ['size', 'bm_lag', 'mom12m', 'profit_lag', 'asset_growth_lag',
            'sales_growth_lag', 'ev_ebitda_lag']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(
            df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)
        )

predictor_cols = ['size', 'bm_lag', 'mom12m', 'profit_lag', 'asset_growth_lag',
                  'sales_growth_lag', 'ev_ebitda_lag']

df_predictors = df_clean.dropna(subset=predictor_cols + ['ret_exc', 'mktcap_lag'])
final_cols = ['month_end', 'symbol', 'ret_exc', 'mktcap_lag'] + predictor_cols
df_final = df_predictors[final_cols].copy()
# df_final.to_csv('qqq_long_panel_clean.csv', index=False)
# print("Saved to qqq_long_panel_clean.csv")
# print("\nHead of final:\n", df_final.head())