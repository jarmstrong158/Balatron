r"""
Standalone simulation: Bloodstone flush-draw EV vs play-what-you-have.

Deals 100 random 8-card hands from a standard 52-card deck with Bloodstone
in the joker slot, then runs find_best_discard + find_best_hands to compare
flush-draw EV against the best playable hand.

Usage:
    cd C:\Users\jarms\repos\balatron
    python scripts\sim_bloodstone.py
"""

import random
import sys
import os
from collections import Counter

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.hand_eval import (
    find_best_hands,
    find_best_discard,
    plan_optimal_action,
    compute_joker_scoring,
    estimate_score,
    classify_hand,
    _draw_probability,
    _round_strategy,
    card_suit,
    card_rank,
    card_chips,
    CARD_CHIP_VALUES,
    BASE_HAND_SCORES,
)

# ---------------------------------------------------------------------------
# Card / deck helpers
# ---------------------------------------------------------------------------

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["H", "D", "C", "S"]
SUIT_NAMES = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}


def make_card(rank: str, suit: str) -> dict:
    """Build an API-format card dict."""
    return {
        "value": {"rank": rank, "suit": suit},
        "modifier": {},
        "state": {},
        "cost": {"sell": 0, "buy": 0},
    }


def make_deck() -> list[dict]:
    """Standard 52-card deck."""
    return [make_card(r, s) for s in SUITS for r in RANKS]


def make_bloodstone_joker() -> dict:
    """Bloodstone joker in API format."""
    return {
        "key": "j_bloodstone",
        "joker_key": "j_bloodstone",
        "set": "JOKER",
        "value": {"effect": "1 in 2 chance for played cards with Heart suit to give X1.5 Mult"},
        "modifier": {},
        "state": {},
        "cost": {"sell": 4, "buy": 8},
    }


def make_gamestate(hand: list[dict], deck: list[dict], jokers: list[dict],
                   blind_target: float = 600.0, hands_left: int = 4,
                   discards_left: int = 3) -> dict:
    """Build a minimal gamestate dict."""
    return {
        "state": "SELECTING_HAND",
        "ante_num": 2,
        "money": 10,
        "hand": {"cards": hand, "count": len(hand)},
        "cards": {"cards": deck, "count": len(deck)},
        "jokers": {"cards": jokers, "count": len(jokers), "limit": 5},
        "consumables": {"cards": [], "count": 0, "limit": 2},
        "round": {
            "hands_left": hands_left,
            "discards_left": discards_left,
            "blind_target": blind_target,
            "chips": 0,
        },
        "blinds": {},
        "hands": {
            # Level 1 defaults
            "High Card":        {"level": 1, "chips": 5, "mult": 1, "played": 0},
            "Pair":             {"level": 1, "chips": 10, "mult": 2, "played": 5},
            "Two Pair":         {"level": 1, "chips": 20, "mult": 2, "played": 2},
            "Three of a Kind":  {"level": 1, "chips": 30, "mult": 3, "played": 1},
            "Straight":         {"level": 1, "chips": 30, "mult": 4, "played": 1},
            "Flush":            {"level": 1, "chips": 35, "mult": 4, "played": 3},
            "Full House":       {"level": 1, "chips": 40, "mult": 4, "played": 0},
            "Four of a Kind":   {"level": 1, "chips": 60, "mult": 7, "played": 0},
            "Straight Flush":   {"level": 1, "chips": 100, "mult": 8, "played": 0},
            "Five of a Kind":   {"level": 1, "chips": 120, "mult": 12, "played": 0},
            "Flush House":      {"level": 1, "chips": 140, "mult": 14, "played": 0},
            "Flush Five":       {"level": 1, "chips": 160, "mult": 16, "played": 0},
        },
        "used_vouchers": [],
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(n_trials: int = 100, hand_size: int = 8, seed: int = 42):
    rng = random.Random(seed)

    jokers = [make_bloodstone_joker()]

    flush_draw_wins = 0
    play_now_wins = 0
    ties = 0
    flush_draw_evs: list[float] = []
    play_now_evs: list[float] = []
    margins: list[float] = []

    print("=" * 100)
    print(f"BLOODSTONE FLUSH-DRAW SIMULATION — {n_trials} hands, {hand_size}-card hand")
    print("Jokers: Bloodstone (50% x1.5 per Hearts scored, during_card)")
    print("=" * 100)
    print()

    for trial in range(n_trials):
        # Reset round strategy cache so each trial is independent
        _round_strategy["fingerprint"] = None
        _round_strategy["committed_type"] = None

        # Shuffle and deal
        full_deck = make_deck()
        rng.shuffle(full_deck)
        hand = full_deck[:hand_size]
        deck = full_deck[hand_size:]

        gs = make_gamestate(hand, deck, jokers)

        # Count hearts in hand
        hand_suits = Counter(card_suit(c) for c in hand)
        hearts_in_hand = hand_suits.get("Hearts", 0)
        hearts_in_deck = sum(1 for c in deck if card_suit(c) == "Hearts")

        # --- Best playable hand right now ---
        top_hands = find_best_hands(hand, jokers, gs, top_n=3)
        best_play = top_hands[0] if top_hands else None
        best_play_score = best_play["estimated_score"] if best_play else 0.0
        best_play_type = best_play["hand_type"] if best_play else "None"

        # --- Best discard strategy ---
        discard = find_best_discard(hand, deck, jokers, gs)
        discard_ev = discard["expected_score"]
        discard_strategy = discard["strategy"]

        # --- plan_optimal_action decision ---
        plan = plan_optimal_action(hand, deck, jokers, gs)
        plan_action = plan["action"]

        # Categorize
        is_flush_draw = "flush" in discard_strategy.lower()
        flush_ev = discard_ev if is_flush_draw else 0.0

        # Find the flush-specific EV even if it wasn't the winner
        # Re-run discard to get all candidates
        flush_specific_ev = 0.0
        non_flush_best_ev = 0.0

        # We need to peek inside find_best_discard to get per-strategy EVs.
        # Reconstruct flush draw EV manually.
        suit_groups: dict[str, list[int]] = {}
        for i, c in enumerate(hand):
            s = card_suit(c)
            suit_groups.setdefault(s, []).append(i)

        hearts_indices = suit_groups.get("Hearts", [])
        deck_size = len(deck)
        deck_suit_counts = Counter(card_suit(c) for c in deck)

        if len(hearts_indices) >= 3:
            needed = 5 - len(hearts_indices)
            available = deck_suit_counts.get("Hearts", 0)
            num_draws = min(hand_size - len(hearts_indices), 5)

            if needed <= 0:
                # Already have flush
                flush_cards = [hand[i] for i in hearts_indices[:5]]
                ht, si = classify_hand(flush_cards)
                flush_specific_ev = estimate_score(ht, flush_cards, si, jokers, gs)
            elif needed <= num_draws and deck_size > 0:
                p_flush = _draw_probability(available, deck_size, needed, num_draws)

                # Score if flush completes
                flush_info = gs["hands"]["Flush"]
                flush_chips = flush_info["chips"]
                flush_mult = flush_info["mult"]
                kept_chips = sum(card_chips(hand[i]) for i in hearts_indices)
                avg_deck_chip = sum(card_chips(c) for c in deck) / max(deck_size, 1)
                drawn_chips = avg_deck_chip * needed

                kept = [hand[i] for i in hearts_indices]
                filler = [c for c in deck if card_suit(c) == "Hearts"][:needed]
                if len(filler) < needed:
                    filler.extend([kept[-1]] * (needed - len(filler)))
                projected = (kept + filler)[:5]
                j_chips, j_mult, j_xmult = compute_joker_scoring(
                    "Flush", projected, list(range(len(projected))),
                    jokers, gs, base_mult=flush_mult,
                )
                flush_hit_score = (flush_chips + kept_chips + drawn_chips + j_chips) * (flush_mult + j_mult) * j_xmult

                # Score if flush misses
                kept_cards = [hand[i] for i in hearts_indices]
                ht, si = classify_hand(kept_cards)
                fallback = estimate_score(ht, kept_cards, si, jokers, gs)

                flush_specific_ev = p_flush * flush_hit_score + (1 - p_flush) * fallback
            else:
                p_flush = 0.0
                flush_hit_score = 0.0
                fallback = 0.0
        else:
            needed = 5 - len(hearts_indices)
            p_flush = 0.0
            flush_hit_score = 0.0
            fallback = 0.0

        margin = flush_specific_ev - best_play_score
        margins.append(margin)
        flush_draw_evs.append(flush_specific_ev)
        play_now_evs.append(best_play_score)

        if flush_specific_ev > best_play_score * 1.01:
            flush_draw_wins += 1
        elif best_play_score > flush_specific_ev * 1.01:
            play_now_wins += 1
        else:
            ties += 1

        # Print detailed output for interesting hands
        interesting = (
            hearts_in_hand >= 3  # Has flush draw potential
            or trial < 10        # First 10 always
            or abs(margin) > 100  # Big margin
        )

        if interesting:
            hand_str = " ".join(f"{card_rank(c)}{card_suit(c)[0]}" for c in hand)
            print(f"Hand #{trial:3d}: {hand_str}")
            print(f"  Hearts: {hearts_in_hand} in hand, {hearts_in_deck} in deck")
            print(f"  Best play: {best_play_type:20s} -> score = {best_play_score:>10.1f}")
            if len(hearts_indices) >= 3:
                print(f"  Flush draw: need {needed}, outs={deck_suit_counts.get('Hearts', 0)}, "
                      f"P(flush)={p_flush:.3f}")
                print(f"    If hit:  {flush_hit_score:>10.1f}")
                print(f"    If miss: {fallback:>10.1f}")
                print(f"    EV:      {flush_specific_ev:>10.1f}")
            else:
                print(f"  Flush draw: only {hearts_in_hand} Hearts (need >=3)")
                print(f"    EV:      {flush_specific_ev:>10.1f}")
            print(f"  Margin (flush_draw - play_now): {margin:>+10.1f}  "
                  f"{'YES FLUSH DRAW' if margin > 0 else 'NO PLAY NOW'}")
            print(f"  find_best_discard chose: {discard_strategy} (EV={discard_ev:.1f})")
            print(f"  plan_optimal_action chose: {plan_action}")
            print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Trials: {n_trials}")
    print(f"  Flush draw wins:  {flush_draw_wins:3d} ({flush_draw_wins/n_trials*100:.1f}%)")
    print(f"  Play now wins:    {play_now_wins:3d} ({play_now_wins/n_trials*100:.1f}%)")
    print(f"  Ties (<1%):       {ties:3d} ({ties/n_trials*100:.1f}%)")
    print()

    # Stats for hands with >=3 Hearts (actual flush draw candidates)
    flush_candidate_trials = [i for i in range(n_trials)
                              if flush_draw_evs[i] > 0]
    if flush_candidate_trials:
        fc_flush_evs = [flush_draw_evs[i] for i in flush_candidate_trials]
        fc_play_evs = [play_now_evs[i] for i in flush_candidate_trials]
        fc_margins = [margins[i] for i in flush_candidate_trials]
        fc_wins = sum(1 for m in fc_margins if m > 0)
        print(f"  Among hands with >=3 Hearts ({len(flush_candidate_trials)} trials):")
        print(f"    Flush draw preferred: {fc_wins}/{len(flush_candidate_trials)} "
              f"({fc_wins/len(flush_candidate_trials)*100:.1f}%)")
        print(f"    Avg flush draw EV:    {sum(fc_flush_evs)/len(fc_flush_evs):>10.1f}")
        print(f"    Avg play now EV:      {sum(fc_play_evs)/len(fc_play_evs):>10.1f}")
        print(f"    Avg margin:           {sum(fc_margins)/len(fc_margins):>+10.1f}")
        print(f"    Max margin:           {max(fc_margins):>+10.1f}")
        print(f"    Min margin:           {min(fc_margins):>+10.1f}")
    else:
        print("  No hands had >=3 Hearts — bad luck on the deal.")

    print()

    # Breakdown by hearts count in hand
    print("  Breakdown by Hearts in hand:")
    for h_count in range(0, 9):
        subset = [(flush_draw_evs[i], play_now_evs[i], margins[i])
                  for i in range(n_trials)
                  if Counter(card_suit(c) for c in make_deck()[:0]).get("Hearts", 0) == 0  # placeholder
                  ]
        # Actually redo with real counts
        pass

    # Redo breakdown properly
    rng2 = random.Random(seed)
    hearts_buckets: dict[int, list[tuple[float, float, float]]] = {}
    for trial in range(n_trials):
        full_deck = make_deck()
        rng2.shuffle(full_deck)
        hand = full_deck[:hand_size]
        h_count = sum(1 for c in hand if card_suit(c) == "Hearts")
        hearts_buckets.setdefault(h_count, []).append(
            (flush_draw_evs[trial], play_now_evs[trial], margins[trial])
        )

    print(f"  {'Hearts':>6s}  {'Count':>5s}  {'Avg Flush EV':>12s}  {'Avg Play EV':>12s}  {'Avg Margin':>12s}  {'Flush Wins':>10s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    for h_count in sorted(hearts_buckets.keys()):
        bucket = hearts_buckets[h_count]
        n = len(bucket)
        avg_f = sum(f for f, _, _ in bucket) / n
        avg_p = sum(p for _, p, _ in bucket) / n
        avg_m = sum(m for _, _, m in bucket) / n
        wins = sum(1 for _, _, m in bucket if m > 0)
        print(f"  {h_count:>6d}  {n:>5d}  {avg_f:>12.1f}  {avg_p:>12.1f}  {avg_m:>+12.1f}  {wins:>5d}/{n}")

    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_sim(n_trials=n)
