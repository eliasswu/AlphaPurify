

import pytest
import pandas as pd
import numpy as np
from alphapurify import FactorAnalyzer

@pytest.fixture(scope="module")
def df():
    np.random.seed(42)
    n_stocks = 100
    start_date = "2024-01-01"
    end_date = "2025-12-31"

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    symbols = [f"stock_{i}" for i in range(1, n_stocks + 1)]
    dfs = []

    for sym in symbols:
        n = len(dates)
        drift = np.random.uniform(0.0001, 0.0005)     
        vol = np.random.uniform(0.01, 0.03)            
        
        eps = np.random.randn(n)
        returns = drift + vol * eps
        
        price = 100 * np.exp(np.cumsum(returns))
        
        df_temp = pd.DataFrame({
            "datetime": dates,
            "symbol": sym,
            "close": price,
            "ret": returns
        })
        
        dfs.append(df_temp)

    df = pd.concat(dfs).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df["future_ret"] = df.groupby("symbol")["ret"].shift(-1)
    noise = np.random.randn(len(df)) * 0.02
    df["factor"] = 0.2 * df["future_ret"] + noise
    df = df.drop(columns=["ret", "future_ret"])
    return df


def test_sheets(df):
    FA = FactorAnalyzer(df,'datetime','symbol','close','factor')
    FA.run()
    res = FA.create_single_fac_full_sheet(return_fig=True)

    assert not res.empty
