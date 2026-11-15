"""
Student Name: Tan Teck Zhi Edward (replace with your name)
GT User ID: etan74 (replace with your User ID)
GT ID: 904189602 (replace with your GT ID)
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

from indicators import (
    calc_bb, calc_ema,
    calc_macd, calc_rsi,
    calc_stochastic_indicator,
)
from marketsimcode import compute_portfolio_stats, normalized_portvals
import util as ut

class ManualStrategy(object):
    """
    A manual learner that can learn (essenitally human coded rules) a trading policy using the same indicators used in StrategyLearner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000,):
        """
        Add evidence is used here for consistency with the Strategy Learner. In actuality, it is likely that 
        This function does nothnig as the "rules" will have been coded by the developer of the class. If that is the case,
        This method might simply consist of a "pass" statement.

        """
        pass

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        prices_df = ut.get_data([symbol], pd.date_range(sd, ed))[symbol]
        norm_price = prices_df / prices_df.iloc[0]

        raw_close = ut.get_data([symbol], pd.date_range(sd, ed), colname="Close")[symbol]
        adj_ratio = prices_df / raw_close
        high_df = ut.get_data([symbol], pd.date_range(sd, ed), colname="High")[symbol] * adj_ratio
        low_df = ut.get_data([symbol], pd.date_range(sd, ed), colname="Low")[symbol] * adj_ratio

        bb_pct = calc_bb(norm_price, lookback=20)
        ema_20 = calc_ema(norm_price, lookback=20)
        macd_hist = calc_macd(norm_price, fast=12, slow=26, signal=9)
        rsi = calc_rsi(prices_df, lookback=14)
        pct_k = calc_stochastic_indicator(high_df, low_df, prices_df, lookback=14, d_lookback=3)

        ema_pct_diff = (norm_price - ema_20) / ema_20
        ema_signal = pd.Series(0.0, index=prices_df.index)
        ema_signal[ema_pct_diff < -0.08] = 1.0
        ema_signal[ema_pct_diff > 0.05] = -1.0

        macd_signal = pd.Series(0.0, index=prices_df.index)
        macd_signal[macd_hist < -0.015] = 1.0
        macd_signal[macd_hist > 0.005] = -1.0

        rsi_signal = pd.Series(0.0, index=prices_df.index)
        rsi_signal[rsi <= 27] = 1.0
        rsi_signal[rsi >= 68] = -1.0

        bb_signal = pd.Series(0.0, index=prices_df.index)
        bb_signal[bb_pct <= 0.05] = 1.0
        bb_signal[bb_pct >= 0.9] = -1.0

        stoch_signal = pd.Series(0.0, index=prices_df.index)
        stoch_signal[pct_k <= 15] = 1.0
        stoch_signal[pct_k >= 75] = -1.0

        basket = ema_signal + macd_signal + rsi_signal + bb_signal + stoch_signal
        df_trades = pd.Series(0.0, index=prices_df.index)
        current_position = 0.0

        ENTRY_THRESH = 3
        EXIT_THRESH = -3
        for date in df_trades.index:
            trade = 0.0
            b = basket.loc[date]

            if current_position == 0.0:
                if b >= ENTRY_THRESH:
                    trade = 1000.0
                elif b <= EXIT_THRESH:
                    trade = -1000.0
            elif current_position == 1000.0:
                if b <= EXIT_THRESH:
                    trade = -2000.0
            elif current_position == -1000.0:
                if b >= ENTRY_THRESH:
                    trade = 2000.0

            current_position += trade
            df_trades.loc[date] = trade

        return df_trades.to_frame(name=symbol)

    def generate_chart(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000, commission=0.0, impact=0.0, title="", filename=""):
        ms_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        bm_trades = pd.Series(0.0, index=ms_trades.index)
        bm_trades.iloc[0] = 1000.0
        bm_trades = bm_trades.to_frame(name=symbol)

        ms_portvals = normalized_portvals(ms_trades, sv=sv, commission=commission, impact=impact)
        bm_portvals = normalized_portvals(bm_trades, sv=sv, commission=commission, impact=impact)

        long_entries, short_entries = _entry_dates(ms_trades, symbol)
        _plot_trade(ms_portvals, bm_portvals, long_entries, short_entries, title, filename)
        _print_stats(ms_portvals, bm_portvals, symbol, sd, ed, sv, commission, impact, title)


def _entry_dates(trades, symbol):
    holdings = trades[symbol].cumsum()
    long_entries = holdings[holdings == 1000].index
    short_entries = holdings[holdings == -1000].index
    long_entries = [
        d
        for i, d in enumerate(long_entries)
        if i == 0 or holdings.loc[:d].iloc[-2] != 1000
    ]
    short_entries = [
        d
        for i, d in enumerate(short_entries)
        if i == 0 or holdings.loc[:d].iloc[-2] != -1000
    ]
    return long_entries, short_entries

def _plot_trade(ms_norm, bm_norm, long_entries, short_entries, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bm_norm.index, bm_norm.values, color="purple", label="Benchmark")
    ax.plot(ms_norm.index, ms_norm.values, color="red", label="Manual Strategy")

    for d in long_entries:
        ax.axvline(x=d, color="blue", linestyle="--", linewidth=0.8)
    for d in short_entries:
        ax.axvline(x=d, color="black", linestyle="--", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def _portval_stats(portvals):
    pv_df = portvals.to_frame(name="portfolio")
    _, cr, adr, sddr, sr = compute_portfolio_stats([1.0], pv_df)
    return cr, adr, sddr, sr

def _print_stats(tos_pv, bench_pv, symbol, sd, ed, start_val, commission, impact, title):
    cr_t, adr_t, sd_t, sr_t = _portval_stats(tos_pv)
    cr_b, adr_b, sd_b, sr_b = _portval_stats(bench_pv)

    w = 60
    print("=" * w)
    print(f"{title:^{w}}")
    print("=" * w)
    print(f"  Symbol  : {symbol}")
    print(f"  Period  : {sd.strftime('%Y-%m-%d')} to {ed.strftime('%Y-%m-%d')}")
    print(f"  Start   : ${start_val:,.0f}   Commission: ${commission:,.2f}   Impact: {impact:.2%}")
    print("-" * w)
    print(f"{'Metric':<28} {'Strategy':>10}  {'Benchmark':>10}")
    print("-" * w)
    print(f"{'Cumulative Return':<28} {cr_t:>10.6f}  {cr_b:>10.6f}")
    print(f"{'Std Dev Daily Returns':<28} {sd_t:>10.6f}  {sd_b:>10.6f}")
    print(f"{'Mean Daily Return':<28} {adr_t:>10.6f}  {adr_b:>10.6f}")
    print(f"{'Sharpe Ratio (annualised)':<28} {sr_t:>10.6f}  {sr_b:>10.6f}")
    print("=" * w)
    print()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()

if __name__ == "__main__":
    print("One does not simply think up a strategy")