# import pandas as pd
# from datetime import datetime

# def load_ff25(datapath, daily, t0=None, tN=None):
#     # Set default values for t0 and tN if they are not provided
#     if t0 is None:
#         t0 = datetime.min
#     if tN is None:
#         tN = datetime.max

#     # Decide on the file names based on whether data is daily or not
#     if daily:
#         ffact5 = 'F-F_Research_Data_Factors_daily.csv'
#         ff25 = '25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv'
#     else:
#         ffact5 = 'F-F_Research_Data_Factors.csv'
#         ff25 = '25_Portfolios_5x5_average_value_weighted_returns_monthly.csv'

#     # Read DATA from CSV
#     DATA = pd.read_csv(
#     datapath + ffact5,
#     skiprows=3,                         # skip description lines
#     parse_dates=["Date"],
#     dayfirst=False,
#     na_values=['-99.99', ''])
    
#     # Filter rows based on date range
#     DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]
    
#     # Read RET from CSV
#     RET = pd.read_csv(
#     datapath + ff25,
#     skiprows=19,
#     parse_dates=["Date"],
#     dayfirst=False,
#     na_values=['-99.99', ''])
    
#     # Inner join of DATA and RET on 'Date' column
#     DATA = pd.merge(DATA, RET, on='Date', how='inner')

#     # Extract required columns and perform operations
#     dates = DATA['Date']
#     mkt = DATA['Mkt-RF'] / 100     # ← change underscore to hyphen
#     ret = DATA.iloc[:, 5:30].divide(100) - DATA['RF'] / 100
#     labels = RET.columns[1:].tolist()

#     return dates, ret, mkt, DATA, labels
import pandas as pd
from datetime import datetime


def load_ff25(datapath, daily, t0=None, tN=None):
    """
    Loads Fama-French 25 portfolios and factors from your clean CSVs.
    Uses monthly data (daily=False) based on your recent successful loading.
    """
    if t0 is None:
        t0 = datetime.min
    if tN is None:
        tN = datetime.max

    # File names
    if daily:
        ffact5 = 'F-F_Research_Data_Factors_daily.csv'
        ff25   = '25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv'
        date_fmt = '%Y/%m/%d'
    else:
        ffact5 = 'F-F_Research_Data_Factors.csv'
        ff25   = '25_Portfolios_5x5_average_value_weighted_returns_monthly.csv'
        date_fmt = '%Y/%m/%d'  # matches 1926/07/31 format in your files

    # Load factors
    DATA = pd.read_csv(
        datapath + ffact5,
        skiprows=0,
        parse_dates=['Date'],
        date_format=date_fmt,
        na_values=['-99.99', '-999', ''],
    )

    # Filter date range
    DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]

    # Load 25 portfolios
    RET = pd.read_csv(
        datapath + ff25,
        skiprows=0,
        parse_dates=['Date'],
        date_format=date_fmt,
        na_values=['-99.99', '-999', ''],
    )

    # Clean any full-NaN rows/columns
    RET = RET.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # ───── Debug ─────
    print("Factors shape:", DATA.shape)
    print("Factors columns:", DATA.columns.tolist())
    print("Factors first 2 rows:\n", DATA.head(2))

    print("\nRET shape:", RET.shape)
    print("RET columns:", RET.columns.tolist())
    print("RET first 2 rows:\n", RET.head(2))

    # Merge
    DATA = pd.merge(DATA, RET, on='Date', how='inner')

    print("\nMerged shape:", DATA.shape)
    print("Merged columns (all):", DATA.columns.tolist())

    dates = DATA['Date']
    mkt   = DATA['Mkt-RF'] / 100

    # Explicitly select the 25 portfolio columns by name
    portfolio_cols = [
        'SMALL LoBM', 'ME1 BM2', 'ME1 BM3', 'ME1 BM4', 'SMALL HiBM',
        'ME2 BM1', 'ME2 BM2', 'ME2 BM3', 'ME2 BM4', 'ME2 BM5',
        'ME3 BM1', 'ME3 BM2', 'ME3 BM3', 'ME3 BM4', 'ME3 BM5',
        'ME4 BM1', 'ME4 BM2', 'ME4 BM3', 'ME4 BM4', 'ME4 BM5',
        'BIG LoBM', 'ME5 BM2', 'ME5 BM3', 'ME5 BM4', 'BIG HiBM'
    ]

    # Verify all 25 columns exist
    missing = [col for col in portfolio_cols if col not in DATA.columns]
    if missing:
        print("Missing portfolio columns:", missing)
        raise ValueError("Some portfolio columns not found in merged DATA")

    print("Using exactly these 25 portfolios:", len(portfolio_cols))

    # Create ret with exactly these columns
    ret = DATA[portfolio_cols].copy()
    ret = ret.divide(100) - DATA['RF'].values[:, None] / 100  # broadcast RF subtraction

    labels = portfolio_cols

    print(f"\nFinal ret shape: {ret.shape}  ← MUST be (n_months, 25)")
    print("ret columns:", ret.columns.tolist())

    return dates, ret, mkt, DATA, labels