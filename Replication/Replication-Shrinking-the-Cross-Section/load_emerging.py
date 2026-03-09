import pandas as pd
from datetime import datetime


def load_emerging_mkt(datapath, daily, t0=None, tN=None):
    """
    Loads ONLY the emerging market portfolios CSV — no merge, no US factors.
    """
    if t0 is None:
        t0 = datetime.min
    if tN is None:
        tN = datetime.max

    ff25 = 'Emerging_Markets_6_Portfolios_ME_Prior_12_2.csv'
    date_fmt = '%Y%m'  # YYYYMM

    DATA = pd.read_csv(
        datapath + ff25,
        na_values=['-99.99', '-999', '']
    )

    DATA = DATA.rename(columns={DATA.columns[0]: 'Date'})

    DATA['Date'] = pd.to_datetime(DATA['Date'].astype(str), format='%Y%m', errors='coerce')

    DATA = DATA.dropna(subset=['Date'])

    if t0 is not None and tN is not None:
        DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]

    # Portfolio columns — hard-code them to avoid detection issues
    portfolio_cols = [
        'SMALL LoPRIOR', 'ME1 PRIOR2', 'SMALL HiPRIOR',
        'BIG LoPRIOR', 'ME2 PRIOR2', 'BIG HiPRIOR'
    ]

    # Force convert to numeric (this is the key fix)
    for col in portfolio_cols:
        DATA[col] = pd.to_numeric(DATA[col], errors='coerce') / 100  # percent to decimal

    # Drop rows with any NaN in returns
    DATA = DATA.dropna(subset=portfolio_cols)

    dates = DATA['Date']
    ret = DATA[portfolio_cols]
    mkt = ret.mean(axis=1)  # average across portfolios

    labels = portfolio_cols

    print(f"\nFinal ret shape: {ret.shape}")
    print("ret columns:", ret.columns.tolist())
    print("Sample ret head:\n", ret.head())

    return dates, ret, mkt, DATA, labels