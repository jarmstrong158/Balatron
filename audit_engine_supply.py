"""One-off (dec-093): is engine acquisition limited by POLICY or by SUPPLY?

Three radically different shop policies were run on the same 60 seeds and the
same checkpoint:

    control            planner ranking, banks when already clearing
    mode 1 (spend)     buy the best AFFORDABLE engine piece, reroll to hunt
    mode 2 (bank)      refuse everything below tier 4, hold money, buy only a
                       real engine

If engine count were policy-limited, these should differ a lot. This script
counts, per run: tier-5 engines SEEN in shops, how many of those were affordable
at the bankroll actually held, and how many were ACQUIRED.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import re
import statistics
import sys

import engine_forcing as ef
from environment.action_space import _api_key_to_name

ARMS = [
    ("control", "logs/eval_ctrl.log",
     r"\[SHOP\] (?:NN=planner|PLANNER) buy: (.+?) \(slot"),
    ("mode1-spend", "logs/eval_forced_pilot2.log",
     r"FORCE-ENGINE buy: (.+?) \(slot"),
    ("mode2-bank", "logs/eval_bank.log",
     r"FORCE-ENGINE buy: (.+?) \(slot"),
]
RUNS = 60


def main() -> None:
    print(f"{'arm':<13} {'t5 seen/run':>11} {'affordable/run':>14} "
          f"{'acquired/run':>12} {'capture':>8}")
    for label, path, buypat in ARMS:
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            print(f"{label:<13} (missing {path})")
            continue

        # Every shop joker offered, with its price, paired to the bankroll the
        # agent held when that shop was entered.
        money_at = [int(m) for m in re.findall(r"SHOP-RAW\] money=\$(\d+)", txt)]
        median_bank = statistics.median(money_at) if money_at else 0

        seen = affordable = 0
        for key, cost in re.findall(
                r'"key": "(j_[a-z_0-9]+)", "cost": \{"buy": (\d+)', txt):
            if ef._tier(_api_key_to_name(key)) == 5:
                seen += 1
                if int(cost) <= median_bank:
                    affordable += 1

        acquired = sum(
            1 for n in re.findall(buypat, txt) if ef._tier(n.strip()) == 5)

        cap = acquired / affordable if affordable else float("nan")
        print(f"{label:<13} {seen/RUNS:>11.2f} {affordable/RUNS:>14.2f} "
              f"{acquired/RUNS:>12.2f} {cap:>8.0%}")

    print("\ncapture = acquired / affordable. Near 100% means the agent is "
          "already taking essentially every engine it can pay for, and no shop "
          "policy can raise the count — the ceiling is supply x affordability.")


if __name__ == "__main__":
    main()
