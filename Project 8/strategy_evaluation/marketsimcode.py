import pandas as pd
import numpy as np
from util import get_data

def normalized_portvals(trades, sv=100_000, commission=0.0, impact=0.0):
    portvals = compute_portvals(
        trades, start_val=sv, commission=commission, impact=impact
    )
    return portvals / portvals.iloc[0]

def compute_portvals(orders_df=None, start_val=1000000, commission=9.95, impact=0.005):
    if orders_df is None or orders_df.empty:
        raise ValueError("orders_df is None or empty.")
    
    symbols = [col for col in orders_df.columns if col != "SPY"]

    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    prices_df = get_data(symbols, pd.date_range(start_date, end_date))
    prices_df = prices_df.ffill().bfill()
    prices_df['CASH'] = 1.0

    trade_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    trade_df.loc[trade_df.index[0], "CASH"] = float(start_val)

    for date in orders_df.index:
        if date not in prices_df.index:
            continue
 
        for sym in symbols:
            shares = orders_df.loc[date, sym]
            if shares == 0:
                continue
 
            price = prices_df.loc[date, sym]
            trade_value = shares * price
 
            trade_df.loc[date, sym] += shares
            trade_df.loc[date, "CASH"] -= trade_value
            if commission != 0.0 or impact != 0.0:
                trade_df.loc[date, "CASH"] -= commission + abs(shares) * price * impact

    holdings_df = trade_df.cumsum()
    portvals = (holdings_df * prices_df).sum(axis=1)
    return portvals


def compute_sharpe_ratio(adr, sddr):
    if sddr == 0:
        return 0.0
    return (np.sqrt(252) * (adr / sddr))

def portval_stats(portvals):
    pv_df = portvals.to_frame(name="portfolio")
    _, cr, adr, sddr, sr = compute_portfolio_stats([1.0], pv_df)
    return cr, adr, sddr, sr

def compute_portfolio_stats(allocs, prices):
    norms = prices / prices.iloc[0, :]
    pos_vals = norms * allocs
    port_val = pos_vals.sum(axis=1)
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets.iloc[1:]
    
    cr = (port_val.iloc[-1] / port_val.iloc[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = compute_sharpe_ratio(adr, sddr)
    
    return port_val, cr, adr, sddr, sr

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()