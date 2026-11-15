import datetime as dt
import numpy as np
import random

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ManualStrategy import ManualStrategy
from experiment1 import run_experiment1
from experiment2 import run_experiment2

GT_ID = 904189602
SYMBOL = "JPM"
SV = 100_000
COMMISSION = 9.95
IMPACT = 0.005

IN_SD = dt.datetime(2008, 1, 1)
IN_ED = dt.datetime(2009, 12, 31)
OUT_SD = dt.datetime(2010, 1, 1)
OUT_ED = dt.datetime(2011, 12, 31)

def run_manual_strategy(): 
    ms = ManualStrategy(verbose=False, impact=IMPACT, commission=COMMISSION)

    ms.generate_chart(
        symbol=SYMBOL,
        sd=IN_SD,
        ed=IN_ED,
        sv=SV,
        commission=COMMISSION,
        impact=IMPACT,
        title="Manual Strategy vs Benchmark - JPM (In-Sample)",
        filename="images/manual_strategy_in_sample.png",
    )
    ms.generate_chart(
        symbol=SYMBOL,
        sd=OUT_SD,
        ed=OUT_ED,
        sv=SV,
        commission=COMMISSION,
        impact=IMPACT,
        title="Manual Strategy vs Benchmark - JPM (Out-of-Sample)",
        filename="images/manual_strategy_out_of_sample.png",
    )

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "etan74"

def study_group():
    return author()

if __name__ == "__main__":
    random.seed(GT_ID)
    np.random.seed(GT_ID)

    run_manual_strategy()
    w=60
    print()
    print("=" * w)
    print(f"{'Experiment 1':^{w}}")
    print("=" * w)
    run_experiment1()
    print()
    print("=" * w)
    print(f"{'Experiment 2':^{w}}")
    print("=" * w)
    run_experiment2()