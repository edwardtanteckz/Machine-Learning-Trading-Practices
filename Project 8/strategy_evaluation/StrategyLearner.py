""""""
"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tan Teck Zhi Edward (replace with your name)
GT User ID: etan74 (replace with your User ID)
GT ID: 904189602 (replace with your GT ID)
"""
import datetime as dt
import numpy as np
import pandas as pd

from indicators import (
    calc_bb, calc_ema, calc_rsi,
    calc_macd, calc_stochastic_indicator
)
from marketsimcode import compute_portvals, compute_portfolio_stats
from RTLearner import RTLearner
from BagLearner import BagLearner
import util as ut


STATE_COLUMNS = ["ema", "rsi", "bbp", "macd", "stoch"]

N_DAY_LOOKAHEAD = 5
LEAF_SIZE = 15
BAGS = 23
VOTE_THRESHOLD = 1.0 / 3
VAL_FRACTION = 0.2


class StrategyLearner(object):
    """
    A strategy learner that trades on a classification signal (BagLearner of
    RTLearners) instead of a QLearner. It uses the same indicators as
    ManualStrategy.

    :param verbose: If "verbose" is True, your code can print out information for debugging.
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
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": LEAF_SIZE},
            bags=BAGS,
            boost=False,
            verbose=False,
        )
        self.train_returns = []
        self.validation_returns = []

    def _compute_features(self, symbol, sd, ed, for_training=False):
        lookback_pad = dt.timedelta(days=60)
        lookahead_pad = dt.timedelta(days=int(N_DAY_LOOKAHEAD * 2) + 5) if for_training else dt.timedelta(days=0)
        dates = pd.date_range(sd - lookback_pad, ed + lookahead_pad)
        prices_df = ut.get_data([symbol], dates)[symbol]
        norm_price = prices_df / prices_df.iloc[0]

        raw_close = ut.get_data([symbol], dates, colname="Close")[symbol]
        adj_ratio = prices_df / raw_close
        high_df = ut.get_data([symbol], dates, colname="High")[symbol] * adj_ratio
        low_df = ut.get_data([symbol], dates, colname="Low")[symbol] * adj_ratio

        ema = calc_ema(norm_price, lookback=20)
        ema_pct_diff = norm_price / ema
        rsi = calc_rsi(prices_df, lookback=14)
        bb_pct = calc_bb(norm_price, lookback=20)
        macd_hist = calc_macd(norm_price, fast=12, slow=26, signal=9)
        pct_k = calc_stochastic_indicator(high_df, low_df, prices_df, lookback=14, d_lookback=3)

        features = pd.DataFrame({
            "ema": ema_pct_diff,
            "rsi": rsi,
            "bbp": bb_pct,
            "macd": macd_hist,
            "stoch": pct_k,
        })
        if for_training:
            fwd_return = (prices_df.shift(-N_DAY_LOOKAHEAD) / prices_df) - 1.0
            features["fwd_return"] = fwd_return

        return features.loc[(features.index >= pd.Timestamp(sd)) & (features.index <= pd.Timestamp(ed))]

    def _make_labels(self, fwd_return, avg_price):
        commission_frac = (self.commission / (1000 * avg_price)) if self.commission else 0.0
        breakeven = 2 * self.impact + commission_frac
        ybuy = breakeven * 1.5 + 0.01
        ysell = -ybuy

        y = pd.Series(0.0, index=fwd_return.index)
        y[fwd_return > ybuy] = 1.0
        y[fwd_return < ysell] = -1.0
        return y

    def _predict_actions(self, feat):
        preds = self.learner.query(feat[STATE_COLUMNS].values)
        actions = np.zeros(len(preds), dtype=int)
        actions[preds > VOTE_THRESHOLD] = 1
        actions[preds < -VOTE_THRESHOLD] = -1
        return actions

    def _trades_from_actions(self, actions, index, symbol):
        df_trades = pd.Series(0.0, index=index)
        holdings = 0
        for i, action in enumerate(actions):
            idx = index[i]
            if action == 1 and holdings < 1000:
                df_trades.loc[idx] = 1000.0 if holdings == 0 else 2000.0
                holdings = 1000
            elif action == -1 and holdings > -1000:
                df_trades.loc[idx] = -1000.0 if holdings == 0 else -2000.0
                holdings = -1000
        return df_trades

    def _score_cr(self, feat, symbol, sv):
        actions = self._predict_actions(feat)
        trades = self._trades_from_actions(actions, feat.index, symbol).to_frame(name=symbol)
        portvals = compute_portvals(
            trades, start_val=sv, commission=self.commission, impact=self.impact
        )
        portvals = portvals / portvals.iloc[0]
        _, cr, _, _, _ = compute_portfolio_stats([1.0], portvals.to_frame(name="portfolio"))
        return cr

    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=100_000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        feat = self._compute_features(symbol, sd, ed, for_training=True)
        feat = feat.dropna(subset=STATE_COLUMNS + ["fwd_return"])

        split_idx = int(len(feat) * (1 - VAL_FRACTION))
        train_feat = feat.iloc[:split_idx]
        val_feat = feat.iloc[split_idx:]

        avg_price = ut.get_data([symbol], pd.date_range(sd, ed))[symbol].mean()
        y_train = self._make_labels(train_feat["fwd_return"], avg_price)

        data_x = train_feat[STATE_COLUMNS].values
        data_y = y_train.values

        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": LEAF_SIZE},
            bags=BAGS,
            boost=False,
            verbose=False,
        )
        self.learner.add_evidence(data_x, data_y)

        train_cr = self._score_cr(train_feat, symbol, sv)
        self.train_returns.append(train_cr)

        if len(val_feat) > 0:
            val_cr = self._score_cr(val_feat, symbol, sv)
            self.validation_returns.append(val_cr)
            if self.verbose:
                print(f"Train CR: {train_cr:.4f} | Val CR (held-out tail): {val_cr:.4f}")
        elif self.verbose:
            print(f"Train CR: {train_cr:.4f} | (no validation slice -- window too short)")

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
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
        feat = self._compute_features(symbol, sd, ed, for_training=False)
        feat = feat.dropna(subset=STATE_COLUMNS)

        actions = self._predict_actions(feat)
        df_trades = self._trades_from_actions(actions, feat.index, symbol)
        return df_trades.to_frame(name=symbol)


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