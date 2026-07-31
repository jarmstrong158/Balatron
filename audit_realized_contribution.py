"""dec-095: rank jokers by what they REALLY contribute, not by the schema flag.

dec-093's forced-engine arms scored jokers with `_tier`, which calls any joker
carrying `xmult=True` a tier-5 engine. That set includes Blackboard (X3 only if
every card held in hand is black), Loyalty Card (every 6th hand) and Cavendish
(a 1-in-1000 lottery) alongside genuine engines. Forcing that NOMINAL share from
12% to 19% made outcomes worse, and the engine hypothesis could not be tested
because the instrument could not see which "engines" were actually firing.

play_quality.jsonl now carries a `realized` list per played hand: for every joker
held, what it contributed to THAT hand. This aggregates it into the two numbers
that matter per joker:

    fire rate     fraction of hands where it contributed anything at all
    mean xmult    average multiplicative contribution WHEN HELD (dead = 1.0)

A joker with xmult=True in the schema but a fire rate near zero is exactly the
card the old instrument could not distinguish from an engine.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import json
import os
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "play_quality.jsonl")
MIN_HANDS = 25          # below this the rates are noise


def main() -> None:
    if not os.path.exists(LOG):
        print(f"missing {LOG}")
        return

    held = collections.Counter()
    fired = collections.Counter()
    xmults = collections.defaultdict(list)
    chips = collections.defaultdict(list)
    mults = collections.defaultdict(list)
    rows = withr = 0

    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows += 1
            rz = r.get("realized")
            if not rz:
                continue
            withr += 1
            for e in rz:
                j = e.get("joker") or "?"
                held[j] += 1
                x = float(e.get("xmult") or 1.0)
                c = float(e.get("chips") or 0.0)
                m = float(e.get("mult") or 0.0)
                xmults[j].append(x)
                chips[j].append(c)
                mults[j].append(m)
                if c or m or abs(x - 1.0) > 1e-9:
                    fired[j] += 1

    print(f"{rows} played hands, {withr} carrying a realised breakdown")
    if not withr:
        print("\nNo `realized` rows yet — dec-095 logging only starts on the next\n"
              "trainer restart. Re-run once the trainer has played some hands.")
        return

    print(f"\n{'joker':<24}{'held':>7}{'fire%':>8}{'mean x':>9}"
          f"{'mean chips':>12}{'mean mult':>11}")
    def key(j):
        return (statistics.mean(xmults[j]), fired[j] / held[j])
    for j in sorted((j for j in held if held[j] >= MIN_HANDS), key=key, reverse=True):
        n = held[j]
        print(f"{j:<24}{n:>7}{fired[j]/n:>8.0%}{statistics.mean(xmults[j]):>9.3f}"
              f"{statistics.mean(chips[j]):>12.1f}{statistics.mean(mults[j]):>11.1f}")

    # THE comparison: nominal tier vs realised behaviour.
    try:
        import engine_forcing as ef
    except Exception:
        return
    print("\nJokers the SCHEMA calls tier-5 engines, ranked by how often they "
          "actually fire:")
    t5 = [j for j in held if held[j] >= MIN_HANDS and ef._tier(j) == 5]
    if not t5:
        print("  (none seen enough times yet)")
        return
    for j in sorted(t5, key=lambda j: fired[j] / held[j]):
        n = held[j]
        flag = "  <-- nominal engine, rarely fires" if fired[j] / n < 0.34 else ""
        print(f"  {j:<24}{n:>6} held{fired[j]/n:>7.0%} fire"
              f"{statistics.mean(xmults[j]):>8.2f}x{flag}")


if __name__ == "__main__":
    main()
