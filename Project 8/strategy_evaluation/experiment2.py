import datetime as dt
import matplotlib.pyplot as plt

from StrategyLearner import StrategyLearner
from marketsimcode import normalized_portvals, portval_stats

SYMBOL = "JPM"
SV = 100_000
IN_SD = dt.datetime(2008, 1, 1)
IN_ED = dt.datetime(2009, 12, 31)
COMMISSION = 0.00
IMPACT_VALUES = [0.000, 0.005, 0.010, 0.025, 0.050, 0.100]


def _count_trades(trades_df, symbol):
    return int((trades_df[symbol] != 0).sum())

def _run_for_impact(impact):
    learner = StrategyLearner(verbose=False, impact=impact, commission=COMMISSION)
    learner.add_evidence(symbol=SYMBOL, sd=IN_SD, ed=IN_ED, sv=SV)
    trades = learner.testPolicy(symbol=SYMBOL, sd=IN_SD, ed=IN_ED, sv=SV)

    portvals = normalized_portvals(trades, sv=SV, commission=COMMISSION, impact=impact)
    cr, adr, sddr, sr = portval_stats(portvals)
    return {
        "impact": impact,
        "cr": cr,
        "adr": adr,
        "sddr": sddr,
        "sr": sr,
        "n_trades": _count_trades(trades, SYMBOL),
        "portvals": portvals,
    }

def _plot_metric(results, key, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    impacts = [r["impact"] for r in results]
    values = [r[key] for r in results]

    ax.plot(impacts, values, marker="o", color="darkorange")
    ax.set_xlabel("Market Impact")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def _plot_portval_overlay(results, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        color = cmap(i / max(len(results) - 1, 1))
        ax.plot(
            r["portvals"].index, r["portvals"].values,
            label=f"impact={r['impact']:.3f}", color=color,
        )

    ax.set_title("Experiment 2: Strategy Learner In-Sample Portfolio Value vs Impact - JPM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _print_summary(results):
    w = 60
    print("=" * w)
    print(f"{'Impact Sensitivity - JPM (In-Sample)':^{w}}")
    print("=" * w)
    print(f"  Symbol  : {SYMBOL}")
    print(f"  Period  : {IN_SD.strftime('%Y-%m-%d')} to {IN_ED.strftime('%Y-%m-%d')}")
    print(f"  Start   : ${SV:,.0f}   Commission: ${COMMISSION:,.2f}")
    print("-" * w)
    print(f"{'Impact':>8} {'Cum Return':>12} {'Sharpe':>10} {'# Trades':>10}")
    print("-" * w)
    for r in results:
        print(f"{r['impact']:>8.3f} {r['cr']:>12.6f} {r['sr']:>10.6f} {r['n_trades']:>10d}")
    print("=" * w)
    print()


def run_experiment2():
    results = [_run_for_impact(impact) for impact in IMPACT_VALUES]

    _print_summary(results)
    _plot_metric(
        results, "cr", "Cumulative Return",
        "Experiment 2: Impact vs Cumulative Return - JPM (In-Sample)",
        "images/experiment2_impact_vs_return.png",
    )
    _plot_metric(
        results, "n_trades", "Number of Trades",
        "Experiment 2: Impact vs Number of Trades - JPM (In-Sample)",
        "images/experiment2_impact_vs_trades.png",
    )
    _plot_portval_overlay(results, "images/experiment2_portval_overlay.png")


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()