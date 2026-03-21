"""Standalone scoring validator for the Balatro RL agent.

Each test case defines a hand, joker config, and manually verified expected score.
Validates the full scoring pipeline: card chips, joker triggers, retriggers, etc.

Balatro scoring formula (level 1 hands):
    score = (hand_chips + scored_card_chips + joker_chips) × (hand_mult + joker_mult) × joker_xmult

Run: python -m tests.test_scoring
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.hand_eval import (
    estimate_score, compute_joker_scoring, classify_hand,
    BASE_HAND_SCORES, CARD_CHIP_VALUES, _build_api_key_cache,
)

# Ensure cache is built
_build_api_key_cache()


def _card(rank: str, suit: str, enhancement: str = "", edition: str = "") -> dict:
    """Create a card dict matching the BalatroBot API format."""
    suit_short = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    c = {
        "value": {"rank": rank, "suit": suit_short.get(suit, suit)},
        "modifier": {},
        "id": id(object()),  # unique id
    }
    if enhancement:
        c["modifier"]["enhancement"] = enhancement
    if edition:
        c["modifier"]["edition"] = edition
    return c


def _joker(name: str, key: str = "", edition: str = "") -> dict:
    """Create a joker dict. key is the Balatro internal key (j_xxx)."""
    if not key:
        key = "j_" + name.lower().replace(" ", "_").replace(".", "").replace("!", "").replace("'", "").replace("-", "_")
    j = {"joker_key": key, "key": key, "modifier": {}}
    if edition:
        j["modifier"]["edition"] = edition
    return j


def _gamestate(hand_levels: dict[str, tuple[int, int]] | None = None,
               held_cards: list[dict] | None = None) -> dict:
    """Create a minimal gamestate dict."""
    gs = {"hands": {}, "hand": {"cards": held_cards or []}}
    if hand_levels:
        for ht, (chips, mult) in hand_levels.items():
            gs["hands"][ht] = {"chips": chips, "mult": mult}
    return gs


def run_test(name: str, cards: list[dict], jokers: list[dict],
             gamestate: dict, expected_score: float,
             tolerance: float = 0.01) -> bool:
    """Run a single scoring test case. Returns True if passed."""
    hand_type, scoring_indices = classify_hand(cards)

    actual = estimate_score(
        hand_type, cards, scoring_indices, jokers, gamestate
    )

    passed = abs(actual - expected_score) / max(expected_score, 1) <= tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         Hand: {hand_type}, scoring indices: {scoring_indices}")
    print(f"         Cards: {[(c['value']['rank'], c['value']['suit']) for c in cards]}")
    if jokers:
        print(f"         Jokers: {[j.get('joker_key', '?') for j in jokers]}")
    print(f"         Expected: {expected_score:.0f}  Actual: {actual:.0f}")
    if not passed:
        # Debug: show joker breakdown
        j_chips, j_mult, j_xmult = compute_joker_scoring(
            hand_type, cards, scoring_indices, jokers, gamestate,
            base_mult=BASE_HAND_SCORES.get(hand_type, (5, 1))[1]
        )
        print(f"         Joker breakdown: chips={j_chips:.1f} mult={j_mult:.1f} xmult={j_xmult:.2f}")
        base_c, base_m = BASE_HAND_SCORES.get(hand_type, (5, 1))
        scoring_chips = sum(CARD_CHIP_VALUES.get(cards[i]["value"]["rank"], 0) for i in scoring_indices)
        print(f"         Base: chips={base_c} mult={base_m} card_chips={scoring_chips}")
    print()
    return passed


def main():
    results = []

    # ================================================================
    # Test 1: Pair of Kings, no jokers, level 1
    # Pair: 10 chips, 2 mult. Kings = 10+10 = 20 card chips.
    # Only the 2 Kings are scoring. Kickers (5, 3, 7) contribute nothing.
    # Score = (10 + 20) * 2 = 60
    # ================================================================
    cards = [
        _card("K", "Hearts"), _card("K", "Spades"),
        _card("5", "Clubs"), _card("3", "Diamonds"), _card("7", "Hearts"),
    ]
    gs = _gamestate()
    results.append(run_test("Pair of Kings, no jokers", cards, [], gs, 60))

    # ================================================================
    # Test 2: Flush of Hearts, no jokers, level 1
    # Flush: 35 chips, 4 mult. All 5 cards score.
    # Cards: A(11)+K(10)+Q(10)+J(10)+9(9) = 50
    # Score = (35 + 50) * 4 = 340
    # ================================================================
    cards = [
        _card("A", "Hearts"), _card("K", "Hearts"), _card("Q", "Hearts"),
        _card("J", "Hearts"), _card("9", "Hearts"),
    ]
    results.append(run_test("Flush of Hearts, no jokers", cards, [], gs, 340))

    # ================================================================
    # Test 3: Pair of 10s with Lusty Joker (Hearts suit joker)
    # Pair of 10s (neither is Hearts), kickers are 5H, 3D, 7C
    # Pair: 10 chips, 2 mult. Scoring cards: two 10s = 10+10 = 20
    # Lusty Joker: +3 mult per SCORING Heart. 10s are not Hearts → +0.
    # Kicker 5H is NOT scoring → does NOT trigger Lusty.
    # Score = (10 + 20) * 2 = 60
    # ================================================================
    cards = [
        _card("T", "Spades"), _card("T", "Diamonds"),
        _card("5", "Hearts"), _card("3", "Diamonds"), _card("7", "Clubs"),
    ]
    jokers = [_joker("Lusty Joker", "j_lusty_joker")]
    results.append(run_test("Pair of 10s, Lusty Joker (no scoring Hearts)", cards, jokers, gs, 60))

    # ================================================================
    # Test 4: Pair of 10s (Hearts) WITH Lusty Joker
    # Pair: 10 chips, 2 mult. Both 10s are Hearts and scoring.
    # Lusty: +3 mult per scoring Heart = +6.
    # Score = (10 + 20) * (2 + 6) = 240
    # ================================================================
    cards = [
        _card("T", "Hearts"), _card("T", "Hearts"),
        _card("5", "Clubs"), _card("3", "Diamonds"), _card("7", "Spades"),
    ]
    jokers = [_joker("Lusty Joker", "j_lusty_joker")]
    results.append(run_test("Pair of 10s (Hearts), Lusty Joker", cards, jokers, gs, 240))

    # ================================================================
    # Test 5: Two Pair (10s and 5s) with Photograph
    # Two Pair: 20 chips, 2 mult. Scoring cards: 10, 10, 5, 5 (4 cards).
    # Kicker 7 is NOT scoring.
    # Photograph: x2 on first scoring face card. 10s are NOT face cards.
    # → Photograph contributes nothing.
    # Card chips (scoring only): 10+10+5+5 = 30
    # Score = (20 + 30) * 2 = 100
    # ================================================================
    cards = [
        _card("T", "Hearts"), _card("T", "Spades"),
        _card("5", "Clubs"), _card("5", "Diamonds"), _card("7", "Hearts"),
    ]
    jokers = [_joker("Photograph", "j_photograph")]
    results.append(run_test("Two Pair 10s/5s, Photograph (no face cards)", cards, jokers, gs, 100))

    # ================================================================
    # Test 6: Two Pair (Kings and 5s) with Photograph
    # Two Pair: 20 chips, 2 mult. Scoring: K, K, 5, 5.
    # Card chips: 10+10+5+5 = 30
    # Photograph: x2 on first scoring face card → Kings are face cards → x2.
    # Photograph is during_card xmult. Using ordering correction:
    #   bonus_mult = base_mult * (during_xmult - 1) = 2 * (2 - 1) = 2
    #   xmult_product = after_xmult = 1.0
    #   total = (20 + 30) * (2 + 2) * 1.0 = 200
    # ================================================================
    cards = [
        _card("K", "Hearts"), _card("K", "Spades"),
        _card("5", "Clubs"), _card("5", "Diamonds"), _card("7", "Hearts"),
    ]
    jokers = [_joker("Photograph", "j_photograph")]
    results.append(run_test("Two Pair Ks/5s, Photograph (face cards score)", cards, jokers, gs, 200))

    # ================================================================
    # Test 7: Pair of Kings with Hanging Chad (no other jokers)
    # Pair: 10 chips, 2 mult. Scoring: K, K = 20 chips.
    # Hanging Chad: retrigger first scoring card 2 extra times.
    # Hanging Chad itself has score_effect: none (retrigger only).
    # First K scores 3 times (original + 2 retriggers) = 10*3 = 30 chips.
    # Second K scores 1 time = 10 chips. Total card chips = 40.
    # Score = (10 hand + 40 card) × 2 mult = 100
    # ================================================================
    cards = [
        _card("K", "Hearts"), _card("K", "Spades"),
        _card("5", "Clubs"), _card("3", "Diamonds"), _card("7", "Hearts"),
    ]
    jokers = [_joker("Hanging Chad", "j_hanging_chad")]
    results.append(run_test("Pair of Kings, Hanging Chad only", cards, jokers, gs, 100))

    # ================================================================
    # Test 8: Flush of Clubs with Gluttonous Joker
    # Flush: 35 chips, 4 mult. All 5 cards score.
    # Cards: K(10)+Q(10)+J(10)+9(9)+8(8) = 47 card chips.
    # Gluttonous: +3 mult per scoring Club = 5 × 3 = +15.
    # Score = (35 + 47) * (4 + 15) = 82 * 19 = 1558
    # ================================================================
    cards = [
        _card("K", "Clubs"), _card("Q", "Clubs"), _card("J", "Clubs"),
        _card("9", "Clubs"), _card("8", "Clubs"),
    ]
    jokers = [_joker("Gluttonous Joker", "j_gluttenous_joker")]
    results.append(run_test("Flush of Clubs, Gluttonous Joker", cards, jokers, gs, 1558))

    # ================================================================
    # Test 9: Three of a Kind (Jacks) with Photograph and Lusty Joker
    # Three of a Kind: 30 chips, 3 mult. Scoring: J, J, J (3 cards).
    # Kickers: 5H, 3D are NOT scoring.
    # Card chips (scoring only): 10+10+10 = 30.
    # Photograph: x2 on first scoring face card → J is face → x2.
    # Lusty Joker: +3 mult per scoring Heart.
    #   If Jacks are J♥, J♠, J♦ → 1 scoring Heart → +3.
    # Kicker 5♥ is NOT scoring → does NOT trigger Lusty.
    #
    # ordering correction: base_mult=3, during_xmult=2.0
    #   bonus_mult = 3*(2-1) + 3 (lusty) = 6
    #   xmult = 1.0
    # Score = (30 + 30) * (3 + 6) * 1.0 = 60 * 9 = 540
    # ================================================================
    cards = [
        _card("J", "Hearts"), _card("J", "Spades"), _card("J", "Diamonds"),
        _card("5", "Hearts"), _card("3", "Diamonds"),
    ]
    jokers = [_joker("Lusty Joker", "j_lusty_joker"), _joker("Photograph", "j_photograph")]
    results.append(run_test("Trips Jacks, Lusty+Photograph (1 scoring Heart)", cards, jokers, gs, 540))

    # ================================================================
    # Test 10: High Card with Joker (base joker, +4 mult)
    # High Card: 5 chips, 1 mult. Only highest card scores.
    # Cards played: A♠, 7♣, 3♦, 2♥, 4♠. Highest = Ace.
    # Scoring card chips: A = 11.
    # Joker: +4 mult (any_hand_played trigger).
    # Score = (5 + 11) * (1 + 4) = 16 * 5 = 80
    # ================================================================
    cards = [
        _card("A", "Spades"), _card("7", "Clubs"), _card("3", "Diamonds"),
        _card("2", "Hearts"), _card("4", "Spades"),
    ]
    jokers = [_joker("Joker", "j_joker")]
    results.append(run_test("High Card Ace, base Joker (+4 mult)", cards, jokers, gs, 80))

    # ================================================================
    # Test 11: Straight with Devious Joker (+100 chips on Straights... wait)
    # Actually Devious Joker gives +9 mult if hand contains a Straight? No.
    # Let me check: Devious Joker is specific_hand_type=Straight, chip_value=100.
    # Wait - schema says trigger_hand_type="Straight", chip_value=100, score_effect=["chips"].
    # So Devious gives +100 chips when playing a Straight.
    # Straight: 30 chips, 4 mult. All 5 score.
    # Cards: 5+6+7+8+9 = 5+6+7+8+9 = 35 card chips.
    # Score = (30 + 35 + 100) * 4 = 165 * 4 = 660
    # ================================================================
    cards = [
        _card("5", "Hearts"), _card("6", "Clubs"), _card("7", "Diamonds"),
        _card("8", "Spades"), _card("9", "Hearts"),
    ]
    jokers = [_joker("Devious Joker", "j_devious")]
    results.append(run_test("Straight 5-9, Devious Joker (+100 chips)", cards, jokers, gs, 660))

    # ================================================================
    # Test 12: Flush of Hearts with Greedy Joker (Diamonds, wrong suit)
    # Flush: 35 chips, 4 mult. All 5 score, all Hearts.
    # Greedy Joker: +3 mult per scoring Diamond. No Diamonds → +0.
    # Cards: A(11)+K(10)+Q(10)+5(5)+3(3) = 39
    # Score = (35 + 39) * 4 = 74 * 4 = 296
    # ================================================================
    cards = [
        _card("A", "Hearts"), _card("K", "Hearts"), _card("Q", "Hearts"),
        _card("5", "Hearts"), _card("3", "Hearts"),
    ]
    jokers = [_joker("Greedy Joker", "j_greedy_joker")]
    results.append(run_test("Flush Hearts, Greedy Joker (wrong suit)", cards, jokers, gs, 296))

    # ================================================================
    # Test 13: Four of a Kind (7s) with Scary Face (+7 chips per face card)
    # Four of a Kind: 60 chips, 7 mult. Scoring: four 7s (4 cards).
    # 7s are NOT face cards → Scary Face contributes nothing.
    # Kicker K♥ is NOT scoring → doesn't trigger even though it's face.
    # Card chips (scoring): 7+7+7+7 = 28
    # Score = (60 + 28) * 7 = 88 * 7 = 616
    # ================================================================
    cards = [
        _card("7", "Hearts"), _card("7", "Spades"), _card("7", "Diamonds"),
        _card("7", "Clubs"), _card("K", "Hearts"),
    ]
    jokers = [_joker("Scary Face", "j_scary_face")]
    results.append(run_test("Quads 7s, Scary Face (no scoring face cards)", cards, jokers, gs, 616))

    # ================================================================
    # Test 14: Four of a Kind (Qs) with Scary Face (+30 chips per face card)
    # Four of a Kind: 60 chips, 7 mult. Scoring: four Qs.
    # Qs ARE face cards → Scary Face: +30 chips × 4 = +120 chips.
    # Card chips: 10+10+10+10 = 40
    # Score = (60 + 40 + 120) * 7 = 220 * 7 = 1540
    # ================================================================
    cards = [
        _card("Q", "Hearts"), _card("Q", "Spades"), _card("Q", "Diamonds"),
        _card("Q", "Clubs"), _card("3", "Hearts"),
    ]
    jokers = [_joker("Scary Face", "j_scary_face")]
    results.append(run_test("Quads Qs, Scary Face (4 scoring face cards)", cards, jokers, gs, 1540))

    # ================================================================
    # Test 15: Pair with FOIL edition on kicker (should NOT add chips)
    # Pair of 5s. Kickers: A♠(FOIL), 3♦, 7♣.
    # Pair: 10 chips, 2 mult. Scoring: 5, 5 = 10 card chips.
    # FOIL on Ace kicker: should NOT add 50 chips (kicker not scoring).
    # Score = (10 + 10) * 2 = 40
    # ================================================================
    cards = [
        _card("5", "Hearts"), _card("5", "Spades"),
        _card("A", "Spades", edition="FOIL"), _card("3", "Diamonds"), _card("7", "Clubs"),
    ]
    results.append(run_test("Pair of 5s, FOIL on kicker (should not add chips)", cards, [], gs, 40))

    # ================================================================
    # Test 16: Pair with FOIL edition on scoring card
    # Pair of 5s (one is FOIL). Kickers: A♠, 3♦, 7♣.
    # Pair: 10 chips, 2 mult. Scoring: 5, 5 = 10 card chips.
    # FOIL on scoring 5: +50 chips.
    # Score = (10 + 10 + 50) * 2 = 140
    # ================================================================
    cards = [
        _card("5", "Hearts", edition="FOIL"), _card("5", "Spades"),
        _card("A", "Spades"), _card("3", "Diamonds"), _card("7", "Clubs"),
    ]
    results.append(run_test("Pair of 5s, FOIL on scoring card", cards, [], gs, 140))

    # ================================================================
    # Summary
    # ================================================================
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("FAILURES DETECTED — scoring pipeline has bugs")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
