import pandas as pd
import numpy as np
from alphapurify.Exposures import PureExposures, PortfolioExposures
import pytest

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

    stock_beta = {
        sym: np.random.uniform(0.5, 1.5) for sym in df["symbol"].unique()
    }
    df["beta"] = df["symbol"].map(stock_beta) + np.random.randn(len(df)) * 0.05

    df["volatility"] = (
        df.groupby("symbol")["ret"]
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    df = df.drop(columns=["ret", "future_ret"])

    return df


def test_sheets(df):
    pe = PortfolioExposures(
        base_df=df(),
        trade_date_col="datetime",
        symbol_col="symbol",
        price_col="close",
        factor_name="factor",
        exposure_cols=["beta", "volatility"],
        bins=5,
        position="ls"   
    )

    pe.run()
    pe.plot_portfolio_exposures()
    pe.plot_portfolio_returns()
    pe.plot_portfolio_exposures_and_returns()

    pure = PureExposures(
        base_df=df(),
        trade_date_col="datetime",
        symbol_col="symbol",
        price_col="close",
        factor_name="factor",
        exposure_cols=["beta", "volatility"]
    )

    # >>> Run analysis
    pure.run()

    # >>> Plot results
    pure.plot_pure_exposures()
    pure.plot_pure_returns()
    pure.plot_pure_exposures_and_returns()

    # >>> Correlation diagnostics
    pure.plot_correlations()
    res = pure
    assert not res.empty