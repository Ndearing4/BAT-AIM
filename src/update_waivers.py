"""
Waiver and IL management — apply waiver moves and manage IL slots.
"""

import requests

from config import HITTER_SLOTS, IL_SLOTS, ROSTER_SIZE, IL_ELIGIBLE_STATUSES
from espn_client import espn_add_drop, espn_activate_from_il, make_move, submit
from llm import ask_gemini, parse_waiver_response
from prompts import build_il_cleanup_prompt


def can_fill_all_slots(hitters: list, slots: list[str]) -> bool:
    """Return True if `hitters` can fill every slot in `slots` without conflicts.

    Uses augmenting-path bipartite matching. Each hitter may fill at most one
    slot; a slot is fillable if the hitter's eligibleSlots contains its name.
    """
    player_for_slot = [None] * len(slots)

    def augment(slot_idx: int, visited: set) -> bool:
        for p_idx, player in enumerate(hitters):
            if slots[slot_idx] not in player.eligibleSlots:
                continue
            try:
                taken = player_for_slot.index(p_idx)
            except ValueError:
                taken = None
            if taken is None:
                player_for_slot[slot_idx] = p_idx
                return True
            if taken not in visited:
                visited.add(taken)
                if augment(taken, visited):
                    player_for_slot[slot_idx] = p_idx
                    return True
        return False

    return all(augment(i, {i}) for i in range(len(slots)))


def find_il_issues(team) -> tuple[list, list, int]:
    """Identify players needing IL management.

    Returns:
        stuck:         players in IL slots who are no longer IL-eligible (must move out)
        stashable:     IL-eligible players NOT currently in an IL slot (can move in)
        open_il_count: IL slots that will be available after clearing stuck players
    """
    stuck = []
    stashable = []
    occupied_il = 0

    for p in team.roster:
        in_il = p.lineupSlot == "IL"
        status = getattr(p, "injuryStatus", "ACTIVE")
        has_il_designation = status in IL_ELIGIBLE_STATUSES

        if in_il:
            occupied_il += 1
            if not has_il_designation:
                stuck.append(p)
        elif has_il_designation and "IL" in p.eligibleSlots:
            stashable.append(p)

    open_il_count = IL_SLOTS - occupied_il + len(stuck)
    return stuck, stashable, open_il_count


def apply_waivers(team, league, raw_response: str, undroppable: set[str] = None) -> list[dict]:
    """Parse Gemini's waiver recommendations and apply moves to ESPN.

    Returns a list of failure dicts for moves that could not be executed.
    Each dict has keys: add, drop, error.
    """
    moves = parse_waiver_response(raw_response)
    failures = []

    if not moves:
        print("✅ No waiver moves needed this cycle.")
        return failures

    available = league.free_agents(size=200)
    scoring_period = league.current_week

    for move in moves:

        add_name  = move["add"]
        drop_name = move["drop"]
        reason    = move["reason"]

        if undroppable and drop_name in undroppable:
            msg = f"{drop_name} is undroppable (ESPN restriction)"
            failures.append({"add": add_name, "drop": drop_name, "error": msg})
            print(f"⛔ Rejected: {msg}. Skipping.")
            continue

        add_player  = next((p for p in available if p.name == add_name), None)
        drop_player = next((p for p in team.roster if p.name == drop_name), None)

        if not add_player or not drop_player:
            msg = f"player not found on {'free agents' if not add_player else 'roster'}"
            failures.append({"add": add_name, "drop": drop_name, "error": msg})
            print(f"⚠️  Could not execute move: Add {add_name} / Drop {drop_name} — {msg}.")
            continue

        pitcher_slots  = {"SP", "RP", "P"}
        post_move_hitters = [
            p for p in team.roster
            if not set(p.eligibleSlots) & pitcher_slots and p.name != drop_name
        ] + [add_player]
        if not can_fill_all_slots(post_move_hitters, HITTER_SLOTS):
            msg = f"dropping {drop_name} would leave a slot unfillable that {add_name} cannot cover"
            failures.append({"add": add_name, "drop": drop_name, "error": msg})
            print(f"⛔ Rejected: {msg}.")
            continue

        try:
            espn_add_drop(add_player, drop_player, scoring_period)
            print(f"✅ Added {add_name}, dropped {drop_name}. Reason: {reason}")
        except requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response is not None else str(e)
            msg = f"ESPN rejected ({e.response.status_code}): {error_body}"
            failures.append({"add": add_name, "drop": drop_name, "error": msg})
            print(f"⚠️  {msg}")

    return failures


def manage_il(team, league, undroppable: set[str] = None):
    """Handle IL slot management before other roster moves.

    Phase 1 (Cleanup): Move IL-ineligible players off IL to bench.
                       If the non-IL roster is full, ask Gemini who to drop first.
    Phase 2 (Stash):   Move IL-eligible bench/active players into open IL slots
                       to free roster spots for waiver pickups.
    """
    stuck, stashable, open_il_count = find_il_issues(team)

    if not stuck and not stashable:
        print("✅ No IL management needed.")
        return

    scoring_period = league.current_week
    roster_by_name = {p.name: p for p in team.roster}

    # ── Phase 1: Clear IL-ineligible players ─────────────────────────────
    if stuck:
        print(f"IL Phase 1: {len(stuck)} player(s) no longer IL-eligible, must move off IL...")

        for player in stuck:
            non_il_count = sum(1 for p in team.roster if p.lineupSlot != "IL")
            non_il_capacity = ROSTER_SIZE - IL_SLOTS

            if non_il_count >= non_il_capacity:
                # Roster full — must drop someone to make room
                print(f"   Roster full — asking Gemini who to drop for {player.name}...")
                prompt = build_il_cleanup_prompt(team, player, undroppable)
                raw = ask_gemini(prompt, mode="il_cleanup")
                print(raw)

                drop_name = None
                for line in raw.splitlines():
                    stripped = line.strip()
                    if stripped.upper().startswith("DROP:"):
                        drop_name = stripped[5:].strip()
                        break

                if not drop_name:
                    print(f"⚠️  Gemini did not suggest a drop — skipping {player.name}")
                    continue

                drop_p = roster_by_name.get(drop_name)
                if not drop_p:
                    print(f"⚠️  Player '{drop_name}' not found on roster — skipping")
                    continue

                # Atomic: drop player + move IL player to bench in one transaction
                drop_slot = drop_p.lineupSlot
                print(f"   Dropping {drop_name} ({drop_slot}), activating {player.name}: IL → BE")
                espn_activate_from_il(player, drop_p, drop_slot, "BE", scoring_period)
                team.roster = [p for p in team.roster if p.name != drop_name]
                del roster_by_name[drop_name]
                player.lineupSlot = "BE"
            else:
                # Roster has room — just move off IL
                print(f"   {player.name}: IL → BE")
                submit([make_move(player, "IL", "BE")], scoring_period, "IL cleanup")
                player.lineupSlot = "BE"

    # ── Phase 2: Stash IL-eligible players ───────────────────────────────
    if stashable and open_il_count > 0:
        current_il = sum(1 for p in team.roster if p.lineupSlot == "IL")
        available = IL_SLOTS - current_il
        stash_count = min(len(stashable), available)

        if stash_count > 0:
            print(f"IL Phase 2: stashing {stash_count} IL-eligible player(s)...")
            for player in stashable[:stash_count]:
                from_slot = player.lineupSlot
                print(f"   {player.name}: {from_slot} → IL")
                submit([make_move(player, from_slot, "IL")], scoring_period, "IL stash")
                player.lineupSlot = "IL"

    print("✅ IL management complete.")
