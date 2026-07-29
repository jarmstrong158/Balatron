"""One-off (dec-093): consistent per-arm acquisition accounting.

Earlier passes in this session got two numbers wrong and both errors flattered a
story I was already telling:

  1. The "$6 median bankroll / engines are unaffordable" cliff was measured on
     MODE 1's log — the arm that had spent its own bankroll down. That median was
     a CONSEQUENCE of the arm's behaviour, not the agent's natural state.
  2. Mode 2's "0.92 buys/run" counted only FORCE-ENGINE buys and was compared
     against control's TOTAL, making it look as though banking had suppressed
     buying when it had not.

This counts every joker-buy path in every arm the same way, and reads the
bankroll per arm rather than borrowing one arm's number for another.
"""
import re
import statistics

import engine_forcing as ef

RUNS = 60
BUY_PATTERNS = [
    r"\[SHOP\] (?:NN=planner|PLANNER) buy: (.+?) \(slot",
    r"FORCE-ENGINE buy: (.+?) \(slot",
    r"\[SHOP\] REDIRECT pack buy . joker buy: (.+?)\s*$",
    r"\[SHOP\] Redirecting buy . (.+?) \(delta",
    r"\[SHOP\] Buying (.+?) \(delta=0",
]
ARMS = [("control", "logs/eval_ctrl.log"),
        ("mode1-spend", "logs/eval_forced_pilot2.log"),
        ("mode2-bank", "logs/eval_bank.log")]


def main() -> None:
    print(f"{'arm':<13} {'med $':>6} {'buys/run':>9} {'t5/run':>7} "
          f"{'t5 share':>9} {'t5 afford/run':>14} {'capture':>8}")
    for label, path in ARMS:
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            print(f"{label:<13} (missing)")
            continue

        money = [int(m) for m in re.findall(r"SHOP-RAW\] money=\$(\d+)", txt)]
        med = statistics.median(money) if money else 0

        names = []
        for pat in BUY_PATTERNS:
            names += [n.strip() for n in re.findall(pat, txt, re.M)]
        tiers = [ef._tier(n) for n in names]
        t5 = sum(t == 5 for t in tiers)

        # engines OFFERED that the arm could pay for at its own median bankroll
        afford = sum(
            1 for key, cost in re.findall(
                r'"key": "(j_[a-z_0-9]+)", "cost": \{"buy": (\d+)', txt)
            if ef._tier(_name(key)) == 5 and int(cost) <= med)

        cap = t5 / afford if afford else float("nan")
        print(f"{label:<13} {med:>6.0f} {len(names)/RUNS:>9.2f} {t5/RUNS:>7.2f} "
              f"{t5/max(len(names),1):>9.0%} {afford/RUNS:>14.2f} {cap:>8.0%}")


def _name(key: str) -> str:
    from environment.action_space import _api_key_to_name
    return _api_key_to_name(key)


if __name__ == "__main__":
    main()
