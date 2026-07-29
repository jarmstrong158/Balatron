"""One-off (dec-093): is an engine piece ever AFFORDABLE when the agent shops?

The forced-engine pilot fired (141 forced buys, 122 forced rerolls) but did NOT
force: only 23% of what it bought was xmult, 36% was economy and 34% additive
scaling — 0.53 xmult per run. The forcing takes the highest-tier AFFORDABLE
piece, so if engines are priced out it degenerates into "buy cheap filler
earlier than the planner would", which is strictly worse than banking.

This measures the affordability wall directly: what the agent holds at shop
entry versus what each tier actually costs.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import re
import statistics
import sys

import engine_forcing as ef
from environment.action_space import _api_key_to_name

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/eval_forced_pilot2.log"


def main() -> None:
    log = open(LOG, encoding="utf-8", errors="ignore").read()

    money = [int(m) for m in re.findall(r"SHOP-RAW\] money=\$(\d+)", log)]
    if not money:
        print("no SHOP-RAW money lines found")
        return
    q = sorted(money)
    print(f"money at shop entry (n={len(money)}): "
          f"median ${statistics.median(money):.0f}  mean ${statistics.mean(money):.1f}")
    for p in (10, 25, 50, 75, 90):
        print(f"    p{p}: ${q[min(int(len(q) * p / 100), len(q) - 1)]}")

    costs = collections.defaultdict(list)
    for key, cost in re.findall(r'"key": "(j_[a-z_0-9]+)", "cost": \{"buy": (\d+)', log):
        tier = ef._tier(_api_key_to_name(key))
        costs[tier].append(int(cost))

    print("\ncost of shop jokers actually seen, by engine tier:")
    for t in sorted(costs, reverse=True):
        v = costs[t]
        print(f"  tier {t}: n={len(v):4d}  median ${statistics.median(v):.0f}  "
              f"mean ${statistics.mean(v):.1f}")

    # THE number: how often could the agent afford a tier-5 engine?
    med = statistics.median(money)
    t5 = costs.get(5, [])
    if t5:
        aff = sum(c <= med for c in t5) / len(t5)
        print(f"\nAt the MEDIAN bankroll (${med:.0f}), the agent can afford "
              f"{aff:.0%} of the tier-5 engines it sees.")
        for bank in (5, 10, 15, 20, 25):
            print(f"    at ${bank:2d}: {sum(c <= bank for c in t5)/len(t5):5.0%} affordable")


if __name__ == "__main__":
    main()
