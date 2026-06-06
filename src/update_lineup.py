"""
Lineup management — apply hitter and pitcher lineup decisions to ESPN.
"""

from config import HITTER_SLOTS, SLOT_PRIORITY
from espn_client import make_move, submit
from llm import parse_lineup_response


def apply_lineup(team, league, raw_response: str, raw_pitcher_response: str):
    """Apply lineup decisions in three ordered phases to avoid slot conflicts.

    Phase 1 — Bench outgoing players first, freeing active slots.
    Phase 2 — Reposition active players who need a different slot
              (e.g. Jazz 2B→3B). Uses BE as a waypoint for triangular swaps.
    Phase 3 — Promote bench players into the newly freed active slots.
    Pitcher pass — Apply LLM pitcher rotation decision: bench non-selected
              pitchers first, then promote selected starters into P slots.
    """

    decision    = parse_lineup_response(raw_response)
    bench_names = set(decision["bench"])
    # ACTIVE list from the LLM covers hitters only — pitchers are handled
    # separately in the pitcher pass below.
    active_names = set(decision["active"])

    roster_by_name       = {p.name: p for p in team.roster}
    current_slot_by_name = {p.name: p.lineupSlot for p in team.roster}
    scoring_period       = league.current_week

    # ── Helpers ────────────────────────────────────────────────────────────
    pitcher_slots = {"SP", "RP", "P"}
    active_slots  = set(SLOT_PRIORITY)  # everything except BE / IL

    def is_pitcher(player) -> bool:
        return bool(set(player.eligibleSlots) & pitcher_slots)

    total_OF_slots = HITTER_SLOTS.count("OF")
    total_P_slots  = sum(1 for slot in current_slot_by_name.values() if slot in pitcher_slots)

    def find_open_slot(player, occupied: list[str]) -> str | None:
        of_used = sum(1 for s in occupied if s == "OF")
        p_used  = sum(1 for s in occupied if s in pitcher_slots)
        for slot in SLOT_PRIORITY:
            if slot not in player.eligibleSlots:
                continue
            if slot == "OF":
                if of_used < total_OF_slots:
                    return slot
            elif slot in pitcher_slots:
                if p_used < total_P_slots:
                    return slot
            elif slot not in occupied:
                return slot
        return None


    # Track which active slots are occupied as we go (list to allow duplicate slot names)
    occupied: list[str] = [
        slot for name, slot in current_slot_by_name.items()
        if slot in active_slots
    ]


    # ── Phase 1: bench outgoing hitters ────────────────────────────────────
    print("Phase 1: benching outgoing players...")
    phase1 = []
    for name in bench_names:
        player = roster_by_name.get(name)
        if not player or is_pitcher(player):
            continue
        from_slot = current_slot_by_name[name]
        if from_slot == "BE":
            continue  # already benched
        phase1.append(make_move(player, from_slot, "BE"))
        try: occupied.remove(from_slot)
        except ValueError: pass
        current_slot_by_name[name] = "BE"
        print(f"   {name}: {from_slot} → BE")
    submit(phase1, scoring_period, "bench outgoing")

    # ── Phase 2: reposition active hitters who need a different slot ────────
    print("Phase 2: repositioning active players...")
    phase2 = []
    for name in active_names:
        player = roster_by_name.get(name)
        if not player or is_pitcher(player):
            continue
        current = current_slot_by_name.get(name, "BE")
        # Determine best target slot for this player
        temp_occupied = list(occupied)
        if current != "BE":
            try: temp_occupied.remove(current)
            except ValueError: pass
        target = find_open_slot(player, temp_occupied)
        if target is None:
            print(f"⚠️  No open slot for {name}, leaving on bench.")
            continue
        if current == target:
            continue  # already in the right place
        if current != "BE" and target in occupied:
            # Triangular conflict — move via BE as a waypoint
            phase2.append(make_move(player, current, "BE"))
            phase2.append(make_move(player, "BE", target))
            print(f"   {name}: {current} → BE → {target} (via waypoint)")
        else:
            phase2.append(make_move(player, current, target))
            print(f"   {name}: {current} → {target}")
        try: occupied.remove(current)
        except ValueError: pass
        occupied.append(target)
        current_slot_by_name[name] = target
    submit(phase2, scoring_period, "reposition active")

    # ── Phase 3: promote bench players into active slots ────────────────────
    print("Phase 3: promoting bench players...")
    phase3 = []
    for name in active_names:
        player = roster_by_name.get(name)
        if not player or is_pitcher(player):
            continue
        if current_slot_by_name.get(name) != "BE":
            continue  # already placed in phase 2
        target = find_open_slot(player, occupied)
        if target is None:
            print(f"⚠️  No open slot for {name}, leaving on bench.")
            continue
        phase3.append(make_move(player, "BE", target))
        occupied.append(target)
        current_slot_by_name[name] = target
        print(f"   {name}: BE → {target}")
    submit(phase3, scoring_period, "promote from bench")

    # ── Pitcher pass: bench non-starters, promote starters ─────────────────
    print("Pitcher pass: optimising P slots...")
    pitchers = [p for p in team.roster if is_pitcher(p)]

    pitcher_decision = parse_lineup_response(raw_pitcher_response)
    starter_names    = set(pitcher_decision["rotation"])
    starters     = [p for p in pitchers if p.name in starter_names]
    non_starters = [p for p in pitchers if p.name not in starter_names]

    # Bench any non-starters currently in active P slots
    pitcher_phase1 = []
    for p in non_starters:
        slot = current_slot_by_name.get(p.name, "BE")
        if slot != "BE":
            pitcher_phase1.append(make_move(p, slot, "BE"))
            try: occupied.remove(slot)
            except ValueError: pass
            current_slot_by_name[p.name] = "BE"
            print(f"   {p.name}: {slot} → BE (not starting)")
    submit(pitcher_phase1, scoring_period, "bench non-starting pitchers")

    # Promote starters sitting on the bench into open P slots
    pitcher_phase2 = []
    for p in starters:
        if current_slot_by_name.get(p.name) == "BE":
            target = find_open_slot(p, occupied)
            if target:
                pitcher_phase2.append(make_move(p, "BE", target))
                occupied.append(target)
                current_slot_by_name[p.name] = target
                print(f"   {p.name}: BE → {target} (starting today)")
            else:
                print(f"⚠️  No open P slot for starter {p.name}")
    submit(pitcher_phase2, scoring_period, "promote starting pitchers")

    print(f"✅ Lineup complete. Reasoning: {decision['reasoning']}")
