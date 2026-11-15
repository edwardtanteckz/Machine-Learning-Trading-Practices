import datetime as dt
import matplotlib.pyplot as plt

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import normalized_portvals, portval_stats

SYMBOL = "JPM"
SV = 100_000
COMMISSION = 9.95
IMPACT = 0.005

IN_SD = dt.datetime(2008, 1, 1)
IN_ED = dt.datetime(2009, 12, 31)
OUT_SD = dt.datetime(2010, 1, 1)
OUT_ED = dt.datetime(2011, 12, 31)

def _benchmark_trades(trades_like):
    bm = trades_like.copy()
    bm.iloc[:, :] = 0.0
    bm.iloc[0, 0] = 1000.0
    return bm

def _plot_comparison(ms_norm, sl_norm, bm_norm, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(bm_norm.index, bm_norm.values, color="purple", label="Benchmark")
    ax.plot(ms_norm.index, ms_norm.values, color="red", label="Manual Strategy")
    ax.plot(sl_norm.index, sl_norm.values, color="green", label="Strategy Learner")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

def _print_stats(manual_pv, learner_pv, bench_pv, symbol, sd, ed, sv, commission, impact, title):
    cr_m, adr_m, sd_m, sr_m   = portval_stats(manual_pv)
    cr_l, adr_l, sd_l, sr_l   = portval_stats(learner_pv)
    cr_b, adr_b, sd_b, sr_b   = portval_stats(bench_pv)

    w = 60
    print("=" * w)
    print(f"{title:^{w}}")
    print("=" * w)
    print(f"  Symbol  : {symbol}")
    print(f"  Period  : {sd.strftime('%Y-%m-%d')} to {ed.strftime('%Y-%m-%d')}")
    print(f"  Start   : ${sv:,.0f}   Commission: ${commission:,.2f}   Impact: {impact:.2%}")
    print("-" * w)
    print(f"{'Metric':<28} {'Manual':>10}  {'Strategy':>10}  {'Benchmark':>8}")
    print("-" * w)
    print(f"{'Cumulative Return':<28} {cr_m:>10.6f}  {cr_l:>10.6f}  {cr_b:>8.6f}")
    print(f"{'Std Dev Daily Returns':<28} {sd_m:>10.6f}  {sd_l:>10.6f}  {sd_b:>8.6f}")
    print(f"{'Mean Daily Return':<28} {adr_m:>10.6f}  {adr_l:>10.6f}  {adr_b:>8.6f}")
    print(f"{'Sharpe Ratio (annualised)':<28} {sr_m:>10.6f}  {sr_l:>10.6f}  {sr_b:>8.6f}")
    print("=" * w)
    print()

def run_experiment1():
    ms = ManualStrategy(verbose=False, impact=IMPACT, commission=COMMISSION)
    ms_in_trades = ms.testPolicy(symbol=SYMBOL, sd=IN_SD, ed=IN_ED, sv=SV)
    ms_out_trades = ms.testPolicy(symbol=SYMBOL, sd=OUT_SD, ed=OUT_ED, sv=SV)

    sl = StrategyLearner(verbose=False, impact=IMPACT, commission=COMMISSION)
    sl.add_evidence(symbol=SYMBOL, sd=IN_SD, ed=IN_ED, sv=SV)
    sl_in_trades = sl.testPolicy(symbol=SYMBOL, sd=IN_SD, ed=IN_ED, sv=SV)
    sl_out_trades = sl.testPolicy(symbol=SYMBOL, sd=OUT_SD, ed=OUT_ED, sv=SV)

    ms_in_norm = normalized_portvals(ms_in_trades, sv=SV, commission=COMMISSION, impact=IMPACT)
    sl_in_norm = normalized_portvals(sl_in_trades, sv=SV, commission=COMMISSION, impact=IMPACT)
    bm_in_norm = normalized_portvals(_benchmark_trades(ms_in_trades), sv=SV, commission=COMMISSION, impact=IMPACT)
    _plot_comparison(
        ms_in_norm, sl_in_norm, bm_in_norm,
        "Experiment 1: Manual Strategy vs Strategy Learner vs Benchmark - JPM (In-Sample)",
        "images/experiment1_in_sample.png",
    )
    _print_stats(ms_in_norm, sl_in_norm, bm_in_norm, SYMBOL, IN_SD, IN_ED, SV, COMMISSION, IMPACT,
                 "Manual Strategy vs Strategy Learner vs Benchmark (In-sample)")

    ms_out_norm = normalized_portvals(ms_out_trades, sv=SV, commission=COMMISSION, impact=IMPACT)
    sl_out_norm = normalized_portvals(sl_out_trades, sv=SV, commission=COMMISSION, impact=IMPACT)
    bm_out_norm = normalized_portvals(_benchmark_trades(ms_out_trades), sv=SV, commission=COMMISSION, impact=IMPACT)
    _plot_comparison(
        ms_out_norm, sl_out_norm, bm_out_norm,
        "Experiment 1: Manual Strategy vs Strategy Learner vs Benchmark - JPM (Out-of-Sample)",
        "images/experiment1_out_of_sample.png",
    )
    _print_stats(ms_out_norm, sl_out_norm, bm_out_norm, SYMBOL, OUT_SD, OUT_ED, SV, COMMISSION, IMPACT,
                 "Manual Strategy vs Strategy Learner vs Benchmark (Out-of-sample)")


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()