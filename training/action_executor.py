"""Action-translation layer — extracted from training/train.py (06-14
monolith decoupling).

Turns a chosen action (policy- or heuristic-decided) plus the live game state
into BalatroBot API calls, and runs the shop/pack auto-actions. Coupled to the
Trainer only through two values now held here: ``policy_authority`` (does the
network own play/discard + which-joker-to-buy?) and the internal
``_shop_block_count``. Everything else operates on the EnvSession passed in.
The heavier helpers are imported lazily inside the methods (unchanged), so this
module's top-level imports are intentionally minimal.
"""

import asyncio
from typing import Optional

import numpy as np

from environment.hand_eval import (
    find_best_discard, find_best_hands, estimate_score_for_hand_type,
    plan_optimal_action, compute_optimal_joker_order,
    plan_consumable_use, optimize_play_order, mouth_should_dig, needle_should_dig,
)


def _find_weakest_sellable_joker(
    jokers_raw: list[dict],
    raw_state: dict,
    *,
    exclude_indices: set[int] | None = None,
) -> tuple[int, float]:
    """Find the weakest joker that is safe to sell.

    Skips: eternal, negative edition, MUST_BUY (Blueprint/Brainstorm),
    retrigger jokers, and copy jokers.

    Returns (index, value) of weakest sellable joker, or (-1, inf) if none.
    """
    from environment.action_space import (
        _estimate_joker_value, _api_key_to_name, MUST_BUY_JOKERS,
    )
    from data.jokers import JOKERS

    weakest_idx = -1
    weakest_val = float("inf")
    # Engine protection (dec-027): prefer to sell a NON-xmult joker. The
    # current-snapshot value estimator undervalues an under-leveled xmult
    # scaler (its worth is future compounding), so without this it would pick
    # the freshly-bought 2nd xmult as "weakest" and churn the engine away.
    weakest_nonx_idx = -1
    weakest_nonx_val = float("inf")
    exclude = exclude_indices or set()

    for i, j in enumerate(jokers_raw):
        if i in exclude:
            continue
        mod = j.get("modifier", {})
        if isinstance(mod, dict):
            if mod.get("eternal", False):
                continue
            if mod.get("edition", "") == "NEGATIVE":
                continue

        # Never sell MUST_BUY jokers
        jk = j.get("joker_key", "") or j.get("key", "")
        name = _api_key_to_name(jk)
        if name in MUST_BUY_JOKERS:
            continue

        is_xmult = False
        # Never sell retrigger or copy jokers
        if name and name in JOKERS:
            schema = JOKERS[name]
            if schema.get("retrigger_effect") or schema.get("copy"):
                continue
            is_xmult = bool(schema.get("xmult") or schema.get("xmult_scaling")
                            or schema.get("scaling_type") == "xmult")

        val = _estimate_joker_value(j, jokers_raw, raw_state)
        if val < weakest_val:
            weakest_val = val
            weakest_idx = i
        if not is_xmult and val < weakest_nonx_val:
            weakest_nonx_val = val
            weakest_nonx_idx = i

    # Sell a non-xmult joker if any is sellable; only fall back to selling an
    # xmult joker when the whole roster is xmult.
    if weakest_nonx_idx >= 0:
        return weakest_nonx_idx, weakest_nonx_val
    return weakest_idx, weakest_val




class ActionExecutor:
    """Translates actions into API calls; runs shop/pack auto-actions."""

    def __init__(self, policy_authority: bool = True):
        self.policy_authority = policy_authority
        self._shop_block_count = 0

    def _planner_pick_joker(self, jokers_raw: list, raw_state: dict,
                            money: float, default_idx: int) -> int:
        """Pillar 2 (dec-034 PLANNING): among affordable, valid shop jokers, pick
        the one the planner says advances the build deepest (multi-ante
        survivability), instead of blindly buying the policy's targeted slot.
        Falls back to the policy's pick (default_idx) on any error or empty set —
        never crashes the shop loop."""
        try:
            from environment.planner import build_value
            from environment.action_space import (
                BAD_JOKERS, _api_key_to_name, _is_non_joker_card,
            )
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            best_idx, best_val = default_idx, None
            for si, sc in enumerate(shop_cards):
                if _is_non_joker_card(sc):
                    continue
                cost = sc.get("cost", {})
                if (cost.get("buy", 999) if isinstance(cost, dict) else 999) > money:
                    continue
                scn = _api_key_to_name(sc.get("joker_key", "") or sc.get("key", ""))
                if scn and scn in BAD_JOKERS:
                    continue
                val = build_value(sc, jokers_raw, raw_state)
                if best_val is None or val > best_val:
                    best_val, best_idx = val, si
            return best_idx
        except Exception as e:
            print(f"[SHOP] planner pick failed ({e}) -> NN slot {default_idx}",
                  flush=True)
            return default_idx

    def _planner_pick_swap(self, jokers_raw: list, raw_state: dict, money: float):
        """Pillar 2 full-slot planning (dec-034): slots are full, so evaluate
        sell-one + buy-one SWAPS by resulting build survivability and return
        (sell_idx, buy_idx) for the best IMPROVING swap, or None (don't downgrade
        the build). THE PLATEAU ZONE — by ante ~2-3 slots fill, and the planner
        was previously inert there (heuristic weakest-sell + the policy's buy), so
        planning did nothing exactly where it matters most. Survivability handles
        engine protection implicitly (selling a key joker tanks the score)."""
        try:
            from environment.planner import build_survivability
            from environment.action_space import (
                _is_non_joker_card, _api_key_to_name, BAD_JOKERS, _joker_sell_value,
            )
            base = build_survivability(jokers_raw, raw_state)
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            best = None  # (survivability, sell_idx, buy_idx)
            for bi, sc in enumerate(shop_cards):
                if _is_non_joker_card(sc):
                    continue
                scn = _api_key_to_name(sc.get("joker_key", "") or sc.get("key", ""))
                if scn and scn in BAD_JOKERS:
                    continue
                bc = sc.get("cost", {})
                buy_cost = bc.get("buy", 999) if isinstance(bc, dict) else 999
                for si, owned in enumerate(jokers_raw):
                    mod = owned.get("modifier", {})
                    if isinstance(mod, dict) and (
                            mod.get("eternal") or mod.get("edition") == "NEGATIVE"):
                        continue  # unsellable — protect
                    sell_price = _joker_sell_value(owned)
                    if buy_cost > money + sell_price:
                        continue
                    roster = [j for k, j in enumerate(jokers_raw) if k != si] + [sc]
                    surv = build_survivability(roster, raw_state)
                    if best is None or surv > best[0]:
                        best = (surv, si, bi)
            if best is not None and best[0] > base + 1e-6:
                return best[1], best[2]
            return None
        except Exception as e:
            print(f"[SHOP] planner swap failed ({e})", flush=True)
            return None

    # Reroll to assemble the build when the best shop joker barely advances it
    # (dec-034 Pillar 2). dec-060 (save->spike): raised 0.12 -> 0.25 so the agent
    # hunts for a REAL engine when the shop is merely MEDIOCRE, not just barren —
    # spending surplus above the interest reserve on finding power instead of
    # wasting it on weak jokers. The dec-057 audit found the agent buys junk and
    # never builds the power to clear power-check bosses; this is the "don't waste
    # money" half of save->spike (the boss spend-down below is the "spike").
    PLANNER_REROLL_THRESHOLD = 0.25

    def _planner_reroll_ok(self, env, raw_state: dict, money: float) -> bool:
        """Is a planner-driven reroll safe right now? Only on genuine SURPLUS
        above the interest floor, with enough left to buy after, under a per-shop
        cap — so seeking build pieces can't drain the economy."""
        reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
        if money < reroll_cost + 4:           # keep enough to buy after rerolling
            return False
        vouchers = raw_state.get("used_vouchers", [])
        v = set(vouchers) if isinstance(vouchers, list) else set()
        floor = 125 if "v_money_tree" in v else 50 if "v_seed_money" in v else 25
        # dec-042: in the early death-zone antes the agent is chronically BELOW the
        # $25 interest floor (median $4-17 at antes 1-5), which made this gate
        # unsatisfiable in 65-79% of those shops — exactly when it must reroll to
        # FIND an xmult engine (the proven death cause). Relax the floor early so
        # seeking the build-defining engine isn't blocked by interest optimization;
        # the per-shop reroll cap below still prevents draining the economy.
        try:
            ante = int(raw_state.get("ante_num", raw_state.get("ante", 1)) or 1)
        except (TypeError, ValueError):
            ante = 1
        if ante <= 5:
            floor = min(floor, 5)
        # dec-060 (save->spike, THE SPIKE): before a HARD boss (dec-059
        # boss-aware difficulty), spend the war chest down to find/buy the power
        # to clear it — money is worthless if the run dies at this gate, so the
        # interest reserve the agent saved IS for exactly this moment. Relax the
        # floor so deep-ante shops before a Wall/Needle/Flint/etc. can reroll for
        # engines instead of protecting a reserve for a next ante it won't reach.
        try:
            from environment.planner import upcoming_boss, boss_difficulty
            if boss_difficulty(upcoming_boss(raw_state)) >= 1.5:
                floor = min(floor, 10)
        except Exception:
            pass
        if money - reroll_cost < floor:        # only spend surplus above interest
            return False
        if getattr(env, "shop_rerolls", 0) >= 3:   # cap planner rerolls per shop
            return False
        return True

    def planner_recommended_action(self, raw_state: dict,
                                   action_mask: np.ndarray) -> Optional[np.ndarray]:
        """The build planner's preferred action for THIS state as a 14-dim tensor,
        or None if the planner has no opinion here (confidence-gate deferral
        target, dec-061 — INFERENCE/EVAL only, never called on the training path).

        Pure REUSE of the dec-034 planner seam: the planner only has authority in
        the SHOP (which-joker), so deferral returns a ``buy_joker`` action there
        (when legal). Executing that routes the WHICH-joker choice through the
        existing planner (``_planner_pick_joker`` open-slot / ``_planner_pick_swap``
        full-slot / reroll-to-assemble) exactly as when the policy itself samples
        buy_joker — no planning is reimplemented. Returns None on non-shop states,
        without ``policy_authority`` (heuristic-drives-everything legacy), or when
        buy_joker is illegal (no affordable buyable joker) — in those cases the
        gate abstains and the policy's own sample stands."""
        from environment.action_space import ACTION_BUY_JOKER
        if not self.policy_authority:
            return None
        if raw_state.get("state", "") != "SHOP":
            return None
        if action_mask[ACTION_BUY_JOKER] <= 0:
            return None
        a = np.zeros(14, dtype=np.float32)
        a[0] = float(ACTION_BUY_JOKER)  # planner picks the slot at execution time
        return a

    def _encode_executed_action(self, api_method: str,
                                api_params: Optional[dict],
                                sampled_action: np.ndarray
                                ) -> Optional[np.ndarray]:
        """Map the EXECUTED API call back to a 14-dim action tensor.

        The heuristic layer routinely overrides the sampled action (buy
        redirects, reroll->buy conversion, leave-guard force-buys, planner
        choosing play vs discard). Storing the sampled action with the
        overridden action's outcome trains the policy heads on structured
        noise — PPO credits results to choices that never executed. This
        re-encodes what actually ran so the gradient matches reality.

        Returns None when the call has no action-tensor equivalent
        (gamestate no-op, pack handling, menu) or the index falls outside
        the head's target range — the caller keeps the sampled action.
        """
        params = api_params or {}
        a = sampled_action.copy()
        target: Optional[int] = None

        if api_method == "play":
            atype = 0
        elif api_method == "discard":
            atype = 1
        elif api_method == "buy":
            if "card" in params:
                if params["card"] > 2:
                    return None
                atype, target = 2, 0 + params["card"]
            elif "voucher" in params:
                if params["voucher"] > 1:
                    return None
                atype, target = 3, 3 + params["voucher"]
            elif "pack" in params:
                if params["pack"] > 1:
                    return None
                atype, target = 4, 5 + params["pack"]
            else:
                return None
        elif api_method == "sell":
            if params.get("joker", 0) > 4:
                return None
            atype, target = 5, 7 + params.get("joker", 0)
        elif api_method == "reroll":
            atype = 7
        elif api_method == "use":
            if params.get("consumable", 0) > 1:
                return None
            atype, target = 8, 12 + params.get("consumable", 0)
            # Card bits matter only for type 8 — set them from the real
            # targets so the (gated) card log-prob describes what ran.
            a[1:13] = 0.0
            for ci in params.get("cards", []) or []:
                if 0 <= ci < 12:
                    a[1 + ci] = 1.0
        elif api_method == "select":
            atype = 9
        elif api_method == "skip":
            atype = 10
        elif api_method == "next_round":
            atype = 13
        else:
            return None  # gamestate / pack / menu — no tensor equivalent

        a[0] = float(atype)
        if target is not None:
            a[13] = float(target)
        # No-target actions keep the sampled target — the conditioned
        # target distribution is uniform for them, so it cannot matter.
        return a

    def _action_to_api_call(self, env, action: np.ndarray,
                            raw_state: dict) -> tuple[str, Optional[dict]]:
        """Convert sampled action tensor to BalatroBot API call.

        Action format: [type(1), cards(12), target(1)]
        All card references are 0-based positional indices.
        """
        action_type = int(action[0])
        card_selections = action[1:13]
        target_idx = int(action[13])

        # Play or Discard — strategic advisor decides based on full math:
        # blind target, joker synergies, deck composition, draw probabilities
        if action_type in (0, 1):
            hand_cards = raw_state.get("hand", {}).get("cards", [])
            deck_cards = raw_state.get("cards", {}).get("cards", [])
            jokers_raw = raw_state.get("jokers", {}).get("cards", [])

            # Inject scaling tracker values so compute_joker_scoring knows
            # the accumulated values for scaling jokers (Square, Ride the Bus, etc.)
            env.game.inject_scaling_values(jokers_raw)

            # REAL-RL MODE: respect the POLICY's play-vs-discard choice. The
            # heuristic only computes the best CARDS for that chosen action
            # (tactical scoring math, not a judgment call). The strategic
            # decision of WHEN to discard vs play is the policy's to learn.
            if self.policy_authority:
                try:
                    if action_type == 0:  # PLAY (policy chose it)
                        # dec-052: The Mouth setup-override. The Mouth locks the
                        # round to the first hand TYPE played; if our strong
                        # committed hand isn't assembled yet, dig for it (discard)
                        # instead of locking a weak type. Tactical legality guard,
                        # so it overrides the policy's PLAY here (executed action is
                        # what PPO records). Highest single deep-death source (74%).
                        # dec-053 adds The Needle (63%): only 1 hand all blind, so
                        # dig to maximize it instead of playing a weak hand now.
                        if (mouth_should_dig(hand_cards, jokers_raw, raw_state)
                                or needle_should_dig(hand_cards, jokers_raw, raw_state)):
                            advice = find_best_discard(hand_cards, deck_cards,
                                                       jokers_raw, raw_state)
                            dig = list(advice["discard_indices"])[:5]
                            if dig:
                                return "discard", {"cards": dig}
                        top = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
                        cards = list(top[0]["card_indices"])[:5] if top else []
                        if not cards and hand_cards:
                            cards = [0]
                        optimal_order = optimize_play_order(cards, hand_cards, jokers_raw)
                        if optimal_order != sorted(optimal_order):
                            played_set = set(optimal_order)
                            non_played = [i for i in range(len(hand_cards))
                                          if i not in played_set]
                            env.pending_hand_rearrange = list(optimal_order) + non_played
                            env.pending_hand_rearrange_fallback = list(optimal_order)
                            cards = list(range(len(optimal_order)))
                        else:
                            cards = optimal_order
                        env.pending_rearrange = (hand_cards, deck_cards)
                        return "play", {"cards": cards}
                    else:  # DISCARD (policy chose it)
                        advice = find_best_discard(hand_cards, deck_cards,
                                                   jokers_raw, raw_state)
                        discard_indices = list(advice["discard_indices"])[:5]
                        if not discard_indices and hand_cards:
                            discard_indices = [0]
                        return "discard", {"cards": discard_indices}
                except Exception as exc:
                    print(f"[WARN] RL card-selection failed ({exc}) — "
                          f"falling back to planner", flush=True)
                    # fall through to the legacy planner below

            try:
                plan = plan_optimal_action(hand_cards, deck_cards, jokers_raw, raw_state)
                action = plan["action"]
                cards = plan["cards"]

                if action == "play":
                    cards = cards[:5]
                    if not cards and hand_cards:
                        cards = [0]
                    optimal_order = optimize_play_order(cards, hand_cards, jokers_raw)
                    # Rearrange hand so played cards appear in optimal scoring
                    # order (face card first for Photograph, etc.).
                    # Balatro scores left-to-right by hand position, so we must
                    # physically reorder the hand before playing.
                    if optimal_order != sorted(optimal_order):
                        played_set = set(optimal_order)
                        non_played = [i for i in range(len(hand_cards)) if i not in played_set]
                        # New hand order: optimal play order first, then non-played
                        new_hand_order = list(optimal_order) + non_played
                        # Store rearrange request — executed before play in the main loop
                        env.pending_hand_rearrange = new_hand_order
                        # Store original indices as fallback if rearrange fails
                        env.pending_hand_rearrange_fallback = list(optimal_order)
                        # After rearrange, played cards will be at indices 0..N-1
                        cards = list(range(len(optimal_order)))
                    else:
                        cards = optimal_order
                    # Store hand/deck cards for joker reordering before play
                    env.pending_rearrange = (hand_cards, deck_cards)
                    return "play", {"cards": cards}
                else:
                    cards = list(cards)[:5]
                    if not cards and hand_cards:
                        cards = [0]
                    return "discard", {"cards": cards}

            except Exception as exc:
                import traceback
                print(f"[WARN] plan_optimal_action CRASHED: {exc}", flush=True)
                traceback.print_exc()
                # Fallback: play or discard based on network's original choice
                if action_type == 0:
                    try:
                        top_hands = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
                        selected = list(top_hands[0]["card_indices"])[:5] if top_hands else [0]
                    except Exception:
                        selected = [0] if hand_cards else []
                    selected = optimize_play_order(selected, hand_cards, jokers_raw)
                    return "play", {"cards": selected}
                else:
                    try:
                        advice = find_best_discard(hand_cards, deck_cards, jokers_raw, raw_state)
                        discard_indices = list(advice["discard_indices"])[:5]
                    except Exception:
                        discard_indices = [0] if hand_cards else []
                    return "discard", {"cards": discard_indices}

        # ================================================================
        # SHOP / BLIND-SELECT ACTIONS — heuristic-guarded
        # The RL network picks the action type, but we verify every
        # buy/sell/reroll against the scoring engine before executing.
        # Bad decisions are redirected to safer alternatives.
        # ================================================================

        game_state = raw_state.get("state", "")

        # Actions 2-7, 11-13 are shop-only. Block them outside SHOP.
        # Actions 8 (use consumable), 9 (select blind), 10 (skip blind)
        # can work outside SHOP and must NOT be blocked here.
        SHOP_ONLY_ACTIONS = {2, 3, 4, 5, 6, 7, 11, 12, 13}
        if action_type in SHOP_ONLY_ACTIONS and game_state != "SHOP":
            return "gamestate", None

        jokers_raw = raw_state.get("jokers", {}).get("cards", [])
        joker_count = len(jokers_raw)
        joker_limit = raw_state.get("jokers", {}).get("limit", 5)
        money = raw_state.get("money", 0)
        current_score = estimate_score_for_hand_type(jokers_raw, raw_state)

        # ── Pending upgrade buy: after selling a joker for an upgrade,
        # force-buy the target on the very next shop action ──
        pending_buy = env.pending_upgrade_buy
        if pending_buy is not None and game_state == "SHOP":
            env.pending_upgrade_buy = None
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            p_idx = pending_buy
            if p_idx < len(shop_cards):
                p_cost = shop_cards[p_idx].get("cost", {}).get("buy", 999)
                if p_cost <= money and joker_count < joker_limit:
                    print(f"[SHOP] Completing upgrade: buying shop card {p_idx} "
                          f"(${p_cost})", flush=True)
                    return "buy", {"card": p_idx}
                else:
                    print(f"[SHOP] Pending upgrade buy failed: "
                          f"cost=${p_cost} money=${money} "
                          f"slots={joker_count}/{joker_limit}", flush=True)

        # Buy joker (target 0-2 = shop card index)
        if action_type == 2:
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            if target_idx >= len(shop_cards):
                return "gamestate", None  # invalid index

            card = shop_cards[target_idx]
            from environment.action_space import (
                _estimate_joker_value, _joker_is_scoring, _joker_sell_value,
                MUST_BUY_JOKERS, BAD_JOKERS, _api_key_to_name, _is_non_joker_card,
            )
            cost = card.get("cost", {}).get("buy", 999)
            if cost > money:
                return "gamestate", None  # can't afford

            # Non-joker cards (planets/tarots/Hermit) — buy directly if mask allowed it
            if _is_non_joker_card(card):
                card_key = card.get("key", "")
                card_set = card.get("set", "").upper()
                cons_count = len(raw_state.get("consumables", {}).get("cards", []))
                cons_limit = raw_state.get("consumables", {}).get("limit", 2)
                if cons_count >= cons_limit:
                    return "gamestate", None  # no consumable slots
                card_label = card.get("label") or card_key
                print(f"[SHOP] Buying consumable: {card_label} "
                      f"(${cost}, {card_set})", flush=True)
                return "buy", {"card": target_idx}

            joker_key = card.get("joker_key", "") or card.get("key", "")
            name = _api_key_to_name(joker_key)

            # Always buy must-buy jokers (Blueprint, Brainstorm, etc.)
            if name in MUST_BUY_JOKERS:
                if joker_count >= joker_limit:
                    # Slots full — sell weakest to make room
                    from environment.action_space import _joker_sell_value as _jsv2
                    weakest_i, _ = _find_weakest_sellable_joker(
                        jokers_raw, raw_state)
                    if weakest_i >= 0:
                        sell_price = _jsv2(jokers_raw[weakest_i])
                        if cost <= money + sell_price:
                            weak_name = jokers_raw[weakest_i].get("label", "?")
                            print(f"[SHOP] Selling {weak_name} (idx {weakest_i}) "
                                  f"to buy MUST-BUY {name}", flush=True)
                            env.pending_upgrade_buy = target_idx
                            return "sell", {"joker": weakest_i}
                    return "gamestate", None  # can't make room
                return "buy", {"card": target_idx}

            # Never buy bad jokers
            if name in BAD_JOKERS:
                return "gamestate", None

            # Check edition (negative bypasses slot limit, poly/holo = high value)
            from environment.action_space import HIGH_VALUE_JOKERS
            shop_mod = card.get("modifier", {})
            shop_edition = shop_mod.get("edition", "") if isinstance(shop_mod, dict) else ""
            is_negative = shop_edition == "NEGATIVE"
            is_high_value = (name in HIGH_VALUE_JOKERS or
                             shop_edition in ("POLYCHROME", "HOLO", "NEGATIVE"))

            if joker_count < joker_limit or is_negative:
                # REAL-RL MODE: an open slot and the policy targeted an
                # affordable, non-bad, non-must-buy joker (all checked above) —
                # buy THAT one. Which joker to buy is a judgment call the policy
                # owns; don't override it with a heuristic "best" scan.
                if self.policy_authority:
                    # PLANNER picks WHICH joker (dec-034 Pillar 2): the policy
                    # decided to BUY (tempo/economy judgment it owns), but which
                    # joker most advances the build is a lookahead question — rank
                    # affordable shop jokers by multi-ante survivability instead of
                    # blindly buying the policy's slot. This is where reacting
                    # becomes planning. PPO records the EXECUTED buy, so the policy
                    # distills toward the planner's picks over time.
                    pick_idx = self._planner_pick_joker(
                        jokers_raw, raw_state, money, target_idx)
                    shop_cards = raw_state.get("shop", {}).get("cards", [])
                    pick_name = ""
                    if 0 <= pick_idx < len(shop_cards):
                        pk = shop_cards[pick_idx]
                        pick_name = _api_key_to_name(
                            pk.get("joker_key", "") or pk.get("key", ""))
                    # Reroll-to-assemble: if the best the planner found barely
                    # advances the build (shop is barren) AND we have a true open
                    # slot + surplus, reroll for a real engine piece instead of
                    # buying junk — the strong-player move that gets engines online
                    # (the audit's early-death cost). Gated so it can't drain econ.
                    if joker_count < joker_limit:
                        try:
                            from environment.planner import build_value as _bval
                            pick_card = shop_cards[pick_idx] if 0 <= pick_idx < len(shop_cards) else None
                            pick_val = _bval(pick_card, jokers_raw, raw_state) if pick_card else 0.0
                        except Exception:
                            pick_val = 1.0  # on error, just buy
                        if (pick_val < self.PLANNER_REROLL_THRESHOLD
                                and self._planner_reroll_ok(env, raw_state, money)):
                            env.shop_rerolls = env.shop_rerolls + 1
                            print(f"[SHOP] PLANNER reroll (barren shop, best "
                                  f"d-surv={pick_val:.2f}, open slot)", flush=True)
                            return "reroll", None
                    tag = "PLANNER" if pick_idx != target_idx else "NN=planner"
                    print(f"[SHOP] {tag} buy: {pick_name or joker_key} "
                          f"(slot {pick_idx}, NN wanted {target_idx})", flush=True)
                    return "buy", {"card": pick_idx}

                # Open slot — scan ALL shop jokers and buy the best one,
                # not just whichever one the NN happened to pick.
                shop_cards_all = raw_state.get("shop", {}).get("cards", [])
                best_idx = -1
                best_delta = -999999
                best_is_high = False
                best_is_scoring = False
                best_name = ""
                for si, sc in enumerate(shop_cards_all):
                    if _is_non_joker_card(sc):
                        continue
                    sc_cost = sc.get("cost", {}).get("buy", 999)
                    if sc_cost > money:
                        continue
                    sc_key = sc.get("joker_key", "") or sc.get("key", "")
                    sc_name = _api_key_to_name(sc_key)
                    if sc_name and sc_name in BAD_JOKERS:
                        continue
                    sc_mod = sc.get("modifier", {})
                    sc_ed = sc_mod.get("edition", "") if isinstance(sc_mod, dict) else ""
                    sc_high = (sc_name in HIGH_VALUE_JOKERS if sc_name else False) or \
                              sc_ed in ("POLYCHROME", "HOLO", "NEGATIVE")
                    sc_scoring = _joker_is_scoring(sc)
                    sc_delta = _estimate_joker_value(sc, jokers_raw, raw_state)
                    # Priority: must-buy > high-value > highest delta > scoring > economy
                    sc_priority = sc_delta
                    if sc_name and sc_name in MUST_BUY_JOKERS:
                        sc_priority = 999999
                    elif sc_high:
                        sc_priority = max(sc_priority, 10000)
                    elif sc_scoring and sc_delta <= 0:
                        sc_priority = max(sc_priority, 1)  # scoring beats economy
                    if sc_priority > best_delta:
                        best_delta = sc_priority
                        best_idx = si
                        best_is_high = sc_high
                        best_is_scoring = sc_scoring
                        best_name = sc_name or sc_key

                if best_idx >= 0 and (best_delta > 0 or best_is_high or best_is_scoring):
                    if best_idx != target_idx:
                        print(f"[SHOP] Redirecting buy → {best_name} "
                              f"(delta={best_delta:.0f}, better than NN pick)",
                              flush=True)
                    return "buy", {"card": best_idx}
                elif best_idx >= 0 and best_delta == 0:
                    # Economy joker, better than empty slot
                    print(f"[SHOP] Buying {best_name} (delta=0, "
                          f"scoring={best_is_scoring}) — filling empty slot",
                          flush=True)
                    return "buy", {"card": best_idx}
                else:
                    # Nothing worth buying
                    print(f"[SHOP] BLOCKED buy {name or joker_key} "
                          f"(no viable joker in shop)", flush=True)
                    return "gamestate", None
            else:
                # Slots full. PLANNER full-slot planning (dec-034 Pillar 2): under
                # policy_authority, the PLANNER evaluates sell+buy SWAPS by build
                # survivability and picks the best improving one — instead of the
                # old "sell the heuristic-weakest, buy the policy's slot". This is
                # the plateau zone (slots fill by ante ~2-3), where the planner was
                # previously inert. If no swap improves the build, don't downgrade.
                if self.policy_authority:
                    swap = self._planner_pick_swap(jokers_raw, raw_state, money)
                    if swap is None:
                        return "gamestate", None  # no swap improves the build
                    sell_idx, buy_idx = swap
                    sell_nm = jokers_raw[sell_idx].get("label", "?")
                    shop_cards = raw_state.get("shop", {}).get("cards", [])
                    buy_nm = ""
                    if 0 <= buy_idx < len(shop_cards):
                        bk = shop_cards[buy_idx]
                        buy_nm = _api_key_to_name(
                            bk.get("joker_key", "") or bk.get("key", ""))
                    print(f"[SHOP] PLANNER swap: sell {sell_nm} -> buy "
                          f"{buy_nm or buy_idx} (NN wanted slot {target_idx})",
                          flush=True)
                    env.pending_upgrade_buy = buy_idx
                    return "sell", {"joker": sell_idx}

                # ---- non-authority heuristic path (legacy) ----
                weakest_idx, _ = _find_weakest_sellable_joker(
                    jokers_raw, raw_state)
                if weakest_idx < 0:
                    return "gamestate", None  # all protected, no room

                sell_price = _joker_sell_value(jokers_raw[weakest_idx])
                if cost > money + sell_price:
                    return "gamestate", None  # can't afford even after sell

                swapped = [j for i, j in enumerate(jokers_raw) if i != weakest_idx]
                swapped.append(card)
                swap_score = estimate_score_for_hand_type(swapped, raw_state)
                weak_name = jokers_raw[weakest_idx].get("label", "?")
                shop_name = name or joker_key

                # Legacy heuristic gate (non-authority mode only)
                swap_threshold = 1.05 if is_high_value else 1.1
                if swap_score > current_score * swap_threshold:
                    print(f"[SHOP] Selling {weak_name} to upgrade to {shop_name} "
                          f"(score {current_score:.0f} → {swap_score:.0f})",
                          flush=True)
                    env.pending_upgrade_buy = target_idx
                    return "sell", {"joker": weakest_idx}
                else:
                    return "gamestate", None

        # Buy voucher (target 3-4 -> voucher index 0-1)
        if action_type == 3:
            v_idx = target_idx - 3 if target_idx >= 3 else target_idx
            shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
            if v_idx < 0 or v_idx >= len(shop_vouchers):
                return "gamestate", None
            vcost = shop_vouchers[v_idx].get("cost", {}).get("buy", 999)
            if vcost > money:
                return "gamestate", None

            # Voucher tiers — must match auto_buy_vouchers logic
            CRITICAL_VOUCHERS = {
                "v_grabber", "v_nacho_tong",       # +1 hand
                "v_wasteful", "v_recyclomancy",     # +1 discard
                "v_paint_brush", "v_palette",       # +1 hand size
                "v_seed_money", "v_money_tree",     # interest cap
                "v_antimatter",                     # +1 joker slot
            }
            GOOD_VOUCHERS = {
                "v_hieroglyph", "v_petroglyph",     # -1 ante (nice but not critical)
                "v_overstock", "v_overstock_plus",  # +1 shop slot
                "v_directors_cut",                  # reroll boss blind
            }
            vkey = shop_vouchers[v_idx].get("key", "")

            if vkey not in CRITICAL_VOUCHERS and vkey not in GOOD_VOUCHERS:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"not valuable", flush=True)
                return "gamestate", None

            # Economy check: don't buy if it drops an interest tier
            # (critical vouchers bypass this)
            remaining_after = money - vcost
            current_tiers = min(money // 5, 5)
            after_tiers = min(max(remaining_after, 0) // 5, 5)
            if after_tiers < current_tiers and vkey not in CRITICAL_VOUCHERS:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"would lose interest tier ({current_tiers} → {after_tiers})",
                      flush=True)
                return "gamestate", None

            # GOOD (non-critical) vouchers need $10 cushion
            if vkey in GOOD_VOUCHERS and money < vcost + 10:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"not enough cushion (${money})", flush=True)
                return "gamestate", None

            return "buy", {"voucher": v_idx}

        # Buy pack (target 5-6 -> pack index 0-1)
        if action_type == 4:
            p_idx = target_idx - 5 if target_idx >= 5 else target_idx
            shop_packs = raw_state.get("packs", {}).get("cards", [])
            if p_idx < 0 or p_idx >= len(shop_packs):
                return "gamestate", None
            cost = shop_packs[p_idx].get("cost", {}).get("buy", 999)
            if cost > money:
                return "gamestate", None

            # Guard: if there's a buyable joker in shop, buy that instead
            pack_key = shop_packs[p_idx].get("key", "")
            is_free = cost <= 0
            if not is_free and joker_count < joker_limit:
                from environment.action_space import (
                    _joker_is_scoring as _pack_jis,
                    _api_key_to_name as _pack_aktn,
                    _is_non_joker_card as _pack_nonj,
                )
                shop_joker_cards = raw_state.get("shop", {}).get("cards", [])
                best_joker_idx = -1
                for sji, sjc in enumerate(shop_joker_cards):
                    if _pack_nonj(sjc):
                        continue
                    sj_cost = sjc.get("cost", {}).get("buy", 999)
                    if sj_cost > money:
                        continue
                    if _pack_jis(sjc):
                        best_joker_idx = sji
                        break
                if best_joker_idx >= 0:
                    sj_key = shop_joker_cards[best_joker_idx].get("key", "")
                    sj_name = _pack_aktn(sj_key) or sj_key
                    print(f"[SHOP] REDIRECT pack buy → joker buy: {sj_name}",
                          flush=True)
                    return "buy", {"card": best_joker_idx}

            # Block standard packs
            if "standard" in pack_key:
                return "gamestate", None

            return "buy", {"pack": p_idx}

        # Sell joker (target 7-11 -> joker index 0-4)
        if action_type == 5:
            j_idx = target_idx - 7 if target_idx >= 7 else target_idx
            if j_idx >= joker_count:
                return "gamestate", None  # invalid index

            # Never sell your only joker
            if joker_count <= 1:
                return "gamestate", None

            from environment.action_space import (
                _estimate_joker_value, _api_key_to_name,
                _joker_sell_value, _safe_modifier,
                MUST_BUY_JOKERS,
            )

            joker = jokers_raw[j_idx]
            mod = _safe_modifier(joker)

            # Hard blocks: eternal, negative, MUST_BUY jokers
            if mod.get("eternal", False):
                return "gamestate", None
            ed = mod.get("edition", "") if isinstance(mod, dict) else ""
            if ed == "NEGATIVE":
                return "gamestate", None
            joker_key = joker.get("joker_key", "") or joker.get("key", "")
            name = _api_key_to_name(joker_key) or joker_key
            if name in MUST_BUY_JOKERS:
                return "gamestate", None

            # Check if there's a better joker available in shop to replace
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            has_upgrade = False
            sell_price = _joker_sell_value(joker)
            for sc in shop_cards:
                from environment.action_space import _is_non_joker_card
                if _is_non_joker_card(sc):
                    continue
                sc_key = sc.get("joker_key", "") or sc.get("key", "")
                sc_name = _api_key_to_name(sc_key)
                if not sc_name:
                    continue
                sc_cost = sc.get("cost", {}).get("buy", 999)
                if sc_cost <= money + sell_price:
                    swapped = [j for i, j in enumerate(jokers_raw) if i != j_idx]
                    swapped.append(sc)
                    swap_score = estimate_score_for_hand_type(swapped, raw_state)
                    if swap_score > current_score * 1.05:
                        has_upgrade = True
                        print(f"[SHOP] Selling {name} (idx {j_idx}) — "
                              f"shop has {sc_name} as upgrade "
                              f"(score {current_score:.0f} -> {swap_score:.0f}, "
                              f"+{(swap_score/max(current_score,1)-1)*100:.0f}%)",
                              flush=True)
                        break

            if not has_upgrade:
                return "gamestate", None

            return "sell", {"joker": j_idx}

        # Sell consumable (target 12-13 -> consumable index 0-1)
        if action_type == 6:
            c_idx = target_idx - 12 if target_idx >= 12 else target_idx
            consumables = raw_state.get("consumables", {}).get("cards", [])
            if c_idx >= len(consumables):
                return "gamestate", None
            # Block selling planets (always useful) and hermit (money doubler)
            c_set = consumables[c_idx].get("set", "")
            c_key = consumables[c_idx].get("key", "")
            if c_set == "PLANET" or c_key in ("c_hermit", "c_temperance"):
                return "gamestate", None
            return "sell", {"consumable": c_idx}

        # Reroll — cap at 3 per shop normally, but uncap when desperate
        if action_type == 7:
            reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
            if money < reroll_cost:
                return "gamestate", None

            # SHOP AUTHORITY (06-21, Phase A): the reroll->force-buy override is
            # REMOVED — the policy now owns reroll-vs-buy. It used to be forced to
            # buy the heuristic's pick whenever a buyable scoring/high joker was
            # present, which stripped build-direction control (dec-023 audit: the
            # heuristic ceiling, not the gradient, is the wall). Hard-legality
            # (affordability, above) and the anti-softlock reroll cap (below) stay.

            # SIMPLE RULE: after rerolling, can you still buy a joker?
            # Cheapest jokers cost ~$2. If money after reroll < $4, don't bother.
            min_joker_cost = 4
            money_after = money - reroll_cost
            if money_after < min_joker_cost:
                return "gamestate", None

            # Track rerolls per shop
            env.shop_rerolls = env.shop_rerolls + 1

            # Reroll cap: only spend surplus above interest floor.
            # Interest floor = $25 base, $50 with Seed Money, $125 with Money Tree.
            # Owned vouchers are a flat key list under "used_vouchers" —
            # raw["vouchers"] is the SHOP's voucher stock and has no "owned".
            vouchers = raw_state.get("used_vouchers", [])
            v_set = set(vouchers) if isinstance(vouchers, list) else set()
            if "v_money_tree" in v_set:
                interest_floor = 125
            elif "v_seed_money" in v_set:
                interest_floor = 50
            else:
                interest_floor = 25
            surplus = max(money - interest_floor, 0)
            affordable = surplus // max(reroll_cost, 1)
            # Allow 1 peek reroll even with 0 surplus, but cap at 8
            max_rerolls = max(1, min(affordable, 8))
            if env.shop_rerolls > max_rerolls:
                return "gamestate", None

            return "reroll", None

        # Use consumable (target 12-13 -> consumable index 0-1)
        if action_type == 8:
            c_idx = target_idx - 12 if target_idx >= 12 else target_idx
            # dec-042: bounds-guard like the sell path (line ~794). When no
            # consumable target is legal the conditioned head can sample a garbage
            # index (uniform over a fully-masked target vector), and this path had
            # NO check — firing use{consumable: 0-11} at the API. That wasted ~427
            # actions (320 index-out-of-range + spin loops). No-op instead.
            consumables = raw_state.get("consumables", {}).get("cards", [])
            if c_idx < 0 or c_idx >= len(consumables):
                return "gamestate", None
            game_state = raw_state.get("state", "")
            if game_state == "SELECTING_HAND":
                hand_cards = raw_state.get("hand", {}).get("cards", [])
                target_cards = [
                    i for i, sel in enumerate(card_selections)
                    if sel > 0.5 and i < len(hand_cards)
                ]
                if target_cards:
                    return "use", {"consumable": c_idx, "cards": target_cards}
            return "use", {"consumable": c_idx}

        # Select blind
        if action_type == 9:
            return "select", None

        # Skip blind — harvest high-value free-supply tags (dec-042). Honors the
        # policy's skip choice for non-boss blinds carrying a worthwhile tag
        # (planets/jokers/leveling/money); keyword-matched, never on the boss.
        if action_type == 10:
            blinds = raw_state.get("blinds", {})
            for b in (blinds.values() if isinstance(blinds, dict) else []):
                if isinstance(b, dict) and b.get("status") in ("SELECT", "CURRENT"):
                    if b.get("type") == "BOSS":
                        break
                    tag_name = b.get("tag_name", "") or ""
                    if any(k in tag_name for k in (
                            "Investment", "Meteor", "Celestial", "Buffoon",
                            "Orbital", "Charm", "Ethereal", "Economy", "Coupon")):
                        return "skip", None
                    break
            # No worthwhile tag — select instead
            return "select", None

        # Select pack card (target 14-18 -> pack card index 0-4)
        if action_type == 11:
            pk_idx = target_idx - 14 if target_idx >= 14 else target_idx
            # Tarot/Spectral pack cards may need hand card targets
            pack_cards = raw_state.get("pack", {}).get("cards", [])
            card_set = pack_cards[0].get("set", "") if pack_cards else ""
            if card_set in ("TAROT", "SPECTRAL"):
                return "pack", {"card": pk_idx, "targets": [0]}
            return "pack", {"card": pk_idx}

        # Skip pack
        if action_type == 12:
            return "pack", {"skip": True}

        # End shop -> next_round
        if action_type == 13:
            # SHOP AUTHORITY (06-21, Phase A): the leave->force-buy override is
            # REMOVED — the policy now owns leave-vs-buy (banking money is a real
            # strategic choice). It used to be force-bought a scoring joker /
            # must-buy consumable whenever it tried to leave with an open slot,
            # stripping build + economy control (dec-023 audit). Hard-legality is
            # preserved downstream; never-sell-copy guards remain in the sell path.

            # ── Riff-Raff optimization ──
            # If we have Riff-Raff, it creates 2 Common Jokers when the next
            # blind is selected — but only if there are empty joker slots.
            # Sell any genuinely weak jokers before leaving the shop to let
            # Riff-Raff fill those slots with new randoms.
            # SAFETY: only sell if we can still beat the upcoming blind without
            # the joker we're selling.
            has_riff_raff = False
            riff_raff_idx = -1
            for ji, j in enumerate(jokers_raw):
                jk = (j.get("joker_key", "") or j.get("key", "")).lower()
                if "riff" in jk and "raff" in jk:
                    has_riff_raff = True
                    riff_raff_idx = ji
                    break

            if has_riff_raff and joker_count >= joker_limit and joker_count > 1:
                # Joker slots are full — Riff-Raff can't fire.
                # Find the weakest non-essential joker and sell it to make room.

                # Figure out the upcoming blind target so we don't sell
                # ourselves into a loss.
                blinds_data = raw_state.get("blinds", {})
                next_blind_target = 0
                if isinstance(blinds_data, dict):
                    for bkey in ("small", "big", "boss"):
                        b = blinds_data.get(bkey, {})
                        if isinstance(b, dict) and b.get("status") == "UPCOMING":
                            next_blind_target = b.get("score", 0)
                            break

                # Current scoring power (per hand × default 4 hands)
                hands_available = raw_state.get("round", {}).get("hands_left", 4)
                if hands_available <= 0:
                    hands_available = 4  # default for next round
                current_total_score = estimate_score_for_hand_type(
                    jokers_raw, raw_state) * hands_available

                worst_idx, worst_val = _find_weakest_sellable_joker(
                    jokers_raw, raw_state,
                    exclude_indices={riff_raff_idx})

                # Only sell if the joker is genuinely weak (below threshold).
                # A random Common joker averages ~4.0 value — sell if worse.
                if worst_idx >= 0 and worst_val < 5.0:
                    # Check scoring power WITHOUT this joker — can we still win?
                    jokers_without = [j for i, j in enumerate(jokers_raw)
                                      if i != worst_idx]
                    score_without = estimate_score_for_hand_type(
                        jokers_without, raw_state) * hands_available

                    if next_blind_target > 0 and score_without < next_blind_target:
                        pass  # Can't sell — would lose next blind
                    else:
                        return "sell", {"joker": worst_idx}

            # Reset reroll counter for next shop visit
            env.shop_rerolls = 0
            return "next_round", None

        return "gamestate", None

    def _log_play_for_joker_order(self, env, raw_state: dict,
                                   intended_order: list[str] | None):
        """Log joker order details for the current play action."""
        from environment.hand_eval import (
            _api_key_to_name, classify_hand, _resolve_copy_source, JOKERS
        )

        # Get current joker order from API state
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        confirmed_names = []
        for j in joker_cards:
            jk = j.get("key", "")
            jn = _api_key_to_name(jk) or jk
            confirmed_names.append(jn)

        # Determine order match
        order_matched = None
        if intended_order is not None:
            order_matched = intended_order == confirmed_names

        # Figure out what hand is being played
        hand_cards = raw_state.get("hand", {}).get("cards", [])
        played_card_strs = []
        hand_type = "Unknown"
        if hand_cards:
            # Best guess: use the first 5 cards (the play hasn't happened yet,
            # but we stored the planned cards in _pending data)
            card_strs = [
                f"{c.get('value', '?')}{c.get('suit', '?')[0]}"
                for c in hand_cards[:5]
            ]
            played_card_strs = card_strs

        # Resolve Brainstorm copy target
        brainstorm_copies = None
        for idx, j in enumerate(joker_cards):
            jk = j.get("key", "")
            jn = _api_key_to_name(jk)
            if jn == "Brainstorm":
                schema = JOKERS.get(jn, {})
                target_dir = schema.get("copy_target", "left")
                copy_src = _resolve_copy_source(joker_cards, idx, target_dir)
                if copy_src is not None:
                    ck = copy_src.get("key", "")
                    brainstorm_copies = _api_key_to_name(ck) or ck
                else:
                    brainstorm_copies = "NONE (no valid target)"
                break

        env.joker_logger.log_play(
            hand_type=hand_type,
            played_cards=played_card_strs,
            intended_order=intended_order,
            confirmed_order=confirmed_names,
            brainstorm_copies=brainstorm_copies,
            order_matched=order_matched,
        )

    async def _auto_rearrange_jokers(self, env, raw_state: dict,
                                      hand_cards: list[dict] | None = None,
                                      deck_cards: list[dict] | None = None
                                      ) -> list[str] | None:
        """Automatically reorder jokers for optimal scoring.

        Called after any action that changes the joker lineup (buy, sell, pack swap)
        or before playing a hand. When hand/deck cards are available, uses them
        for accurate scoring. Otherwise builds representative hands from deck.

        Returns list of joker names in the new order, or None if no change.
        """
        from environment.hand_eval import _api_key_to_name

        jokers = raw_state.get("jokers", {}).get("cards", [])
        if len(jokers) <= 1:
            return None

        try:
            new_order = compute_optimal_joker_order(
                jokers, gamestate=raw_state,
                hand_cards=hand_cards, deck_cards=deck_cards
            )
            if new_order is not None:
                joker_names = []
                for idx in new_order:
                    if idx < len(jokers):
                        jk = jokers[idx].get("key", "")
                        jn = _api_key_to_name(jk) or jk
                        joker_names.append(jn)
                await env.game.execute_action("rearrange", {"jokers": new_order})
                return joker_names
        except Exception as e:
            print(f"[WARN] Joker rearrange failed: {e}", flush=True)
        return None

    async def _auto_buy_vouchers(self, env, raw_state: dict) -> dict:
        """Auto-buy vouchers in the shop when affordable.

        Most vouchers are strong upgrades. Skip only the ones that add
        more tarot/celestial cards to packs (low value for the bot).

        Timing awareness: if we're in the first shop of an ante (after
        Small Blind) and the voucher isn't critical, defer the purchase
        to a later shop if buying now would cost us interest income.
        """
        # Vouchers to skip — extra tarot/celestial pack cards aren't useful
        SKIP_VOUCHERS = {
            "v_tarot_merchant",   # Tarot cards appear more in shop
            "v_tarot_tycoon",     # Even more tarots
            # dec-042: planet vouchers UN-blacklisted. Leveling (planets) is a
            # binding constraint (committed hand stuck ~level 2 through ante 8),
            # so vouchers that raise planet shop-rate are valuable, not "low value".
            "v_omen_globe",       # Spectral cards in Arcana packs
        }

        shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
        if not shop_vouchers:
            return raw_state

        money = raw_state.get("money", 0)

        # Check interest cap for safe spending
        owned_vouchers = raw_state.get("used_vouchers", [])
        v_set = set(owned_vouchers) if isinstance(owned_vouchers, list) else set()
        if "v_money_tree" in v_set:
            interest_cap = 25   # $25 max interest (from $125)
            interest_cap_money = 125
        elif "v_seed_money" in v_set:
            interest_cap = 10   # $10 max interest (from $50)
            interest_cap_money = 50
        else:
            interest_cap = 5    # $5 max interest (from $25)
            interest_cap_money = 25

        # Determine which shop this is within the ante by checking
        # blind statuses. If Small Blind is defeated but Big Blind
        # is not, we're in the first shop (more shops coming).
        blinds = raw_state.get("blinds", {})
        small_status = ""
        big_status = ""
        boss_status = ""
        if isinstance(blinds, dict):
            small_info = blinds.get("small", {})
            big_info = blinds.get("big", {})
            boss_info = blinds.get("boss", {})
            small_status = small_info.get("status", "") if isinstance(small_info, dict) else ""
            big_status = big_info.get("status", "") if isinstance(big_info, dict) else ""
            boss_status = boss_info.get("status", "") if isinstance(boss_info, dict) else ""

        # shops_remaining: how many more shop visits after this one
        # After Small Blind: Big + Boss shops remain = 2
        # After Big Blind: Boss shop remains = 1
        # After Boss Blind: next ante = 0
        if boss_status == "DEFEATED":
            shops_remaining = 0
        elif big_status == "DEFEATED":
            shops_remaining = 1
        else:
            shops_remaining = 2

        # Only truly game-changing vouchers bypass interest protection
        CRITICAL_VOUCHERS = {
            "v_grabber", "v_nacho_tong",       # +1 hand (huge)
            "v_wasteful", "v_recyclomancy",     # +1 discard (huge)
            "v_paint_brush", "v_palette",       # +1 joker slot (huge)
            "v_seed_money", "v_money_tree",     # interest cap increase (pays for itself)
            "v_antimatter",                     # +1 joker slot (huge)
        }

        for v_idx, voucher in enumerate(shop_vouchers):
            key = voucher.get("key", "")
            cost = voucher.get("cost", {}).get("buy", 999)

            if key in SKIP_VOUCHERS:
                continue

            if cost > money:
                continue

            # Don't buy if it would drop us below an interest tier
            remaining_after = money - cost
            current_tiers = min(money // 5, interest_cap)
            after_tiers = min(max(remaining_after, 0) // 5, interest_cap)
            if after_tiers < current_tiers and key not in CRITICAL_VOUCHERS:
                continue

            # ── Economy guard for non-critical vouchers ──
            # Vouchers are unique per shop visit (don't persist), so we
            # can't defer. But we CAN refuse to buy non-critical vouchers
            # that cost us too much economy.  Only buy if we keep $10+
            # after purchase (preserve interest and shop flexibility).
            if key not in CRITICAL_VOUCHERS:
                if remaining_after < 10:
                    continue

            # Buy it
            try:
                # print(f"[SHOP] AUTO-BUY voucher {v_idx} ({key}) for ${cost}", flush=True)
                await env.game.execute_action("buy", {"voucher": v_idx})
                await asyncio.sleep(0.3)
                # Re-fetch state after purchase
                try:
                    raw_state = await env.game.fetch_gamestate()
                    money = raw_state.get("money", 0)
                except Exception:
                    pass
            except Exception as e:
                print(f"[SHOP] Voucher buy failed: {e}", flush=True)

        return raw_state

    async def _auto_use_consumables(self, env, raw_state: dict) -> dict:
        """Automatically use consumable cards when heuristics say it's optimal.

        Called at the start of SELECTING_HAND. Uses Planet cards immediately,
        Tarot cards with smart targeting, and economy cards at right timing.
        Returns the (possibly updated) raw_state after any consumable use.
        """
        # Loop: may use multiple consumables in sequence (e.g., Planet then Tarot)
        max_uses = 2  # At most 2 consumable slots
        for _ in range(max_uses):
            consumables = raw_state.get("consumables", {}).get("cards", [])
            if not consumables:
                break

            try:
                action = plan_consumable_use(raw_state)
                if action is None:
                    break

                await env.game.execute_action("use", action)
                # Re-fetch state after use (cards may have changed)
                raw_state = await env.game.fetch_gamestate()
            except Exception:
                break  # consumable use is best-effort

        return raw_state
