import pandas as pd
import numpy as np

INPUT_LONG = "/Users/madisonpoore/Desktop/MSProject/Replication/Replication-Shrinking-the-Cross-Section/qqq_longpanel2.csv"
OUTPUT_WIDE = "qqq_new_portfolios.csv"

print("Loading long panel...")
df = pd.read_csv(INPUT_LONG)
df['month_end'] = pd.to_datetime(df['month_end'])

print(f"Loaded {len(df):,} stock-month rows | "
      f"{df['symbol'].nunique()} symbols | "
      f"{df['month_end'].nunique()} months")

# Your 20 factors (no momentum)
predictors = {
    'size': 'Size',
    'bm': 'BM',
    'profit_gpm': 'ProfitGPM',
    'asset_growth': 'AssetGrowth',
    'sales_growth': 'SalesGrowth',
    'ev_ebitda': 'EVtoEBITDA',
    'roe': 'ROE',
    'roa': 'ROA',
    'ps_ratio': 'PS',
    'ev_sales': 'EVSales',
    'fcf_yield': 'FCFYield',
    'eps_growth': 'EPSGrowth',
    'ocf_growth': 'OCFGrowth',
    'net_margin': 'NetMargin',
    'op_margin': 'OpMargin',
    'ebitda_margin': 'EBITDAMargin',
    'debt_equity': 'DebtEquity',
    'current_ratio': 'CurrentRatio',
    'earn_yield': 'EarnYield',
    'roic': 'ROIC'
}

wide = pd.DataFrame(index=sorted(df['month_end'].unique()))
wide.index.name = 'month_end'

def vw_ret(g):
    if len(g) < 2:
        return np.nan
    weights = g['mktcap_lag'] / g['mktcap_lag'].sum()
    return np.average(g['ret_exc'], weights=weights)

print("Forming 100 quintile portfolios (rank-based)...")
for pred_col, name in predictors.items():
    print(f"  {name} ({pred_col})")

    def monthly_quintiles(g):
        valid = g.dropna(subset=[pred_col, 'ret_exc', 'mktcap_lag']).copy()
        if len(valid) < 2:
            return pd.Series({f"{name}_Quintile{q}": np.nan for q in range(1,6)})
        
        # Rank-based quintiles - most robust method
        valid['rank'] = valid[pred_col].rank(pct=True, method='average')
        
        try:
            valid['bucket'] = pd.cut(
                valid['rank'], 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                labels=range(1,6), 
                include_lowest=True,
                right=False
            )
        except Exception:
            # Fallback: simple equal split if rank fails
            print(f"  Warning: binning failed for {name} in {g.name.date()}. Using simple split.")
            valid = valid.sort_values(pred_col)
            valid['bucket'] = np.minimum((np.arange(len(valid)) // max(1, len(valid)//5)) + 1, 5)
        
        result = {}
        for b in range(1, 6):
            subset = valid[valid['bucket'] == b]
            result[f"{name}_Quintile{b}"] = vw_ret(subset) if len(subset) > 0 else np.nan
        
        return pd.Series(result)

    monthly_ports = df.groupby('month_end').apply(monthly_quintiles)
    wide = wide.join(monthly_ports)

wide = wide.reset_index()
wide.to_csv(OUTPUT_WIDE, index=False)

print(f"\nSaved 100-portfolio quintile file: {OUTPUT_WIDE}")
print(f"Shape: {wide.shape} (months × columns)")
print("\nHead preview:\n", wide.head(8).to_string(index=False))
print("\nTail preview:\n", wide.tail(8).to_string(index=False))
print("\nOverall NaN rate:", wide.drop('month_end', axis=1).isna().mean().mean() * 100, "%")