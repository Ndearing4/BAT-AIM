"""
LLM prompt builders for the AI Fantasy Baseball Manager.
"""

import json
from collections import Counter
from datetime import date

from config import GM_PERSONA, HITTER_SLOTS, TEAM_ID


def serialize_player(p, include_stats=True) -> dict:
    """Convert ESPN player object to a clean dict for the AI."""
    d = {
        "name":     p.name,
        "position": p.eligibleSlots,
        "team":     p.proTeam,
        "injured":  p.injured,
        "status":   getattr(p, "injuryStatus", "ACTIVE"),
        "total_points":     getattr(p, "total_points", 0),
        "projected_points": getattr(p, "projected_total_points", 0),
    }
    if include_stats and hasattr(p, "stats"):
        season = p.stats.get(0, {})
        breakdown = season.get("breakdown", {})
        games = breakdown.get("G", 0)
        points = season.get("points", getattr(p, "total_points", 0))
        if games and games > 0:
            d["ppg"] = round(points / games, 2)
        else:
            d["ppg"] = 0
        d["season_stats"] = breakdown
    return d


def build_lineup_prompt(team, league) -> str:
    roster = [serialize_player(p) for p in team.roster]

    pitcher_slots = {"SP", "RP", "P"}
    slot_order    = ["C", "1B", "2B", "3B", "SS", "OF", "UTIL"]

    hitters = [p for p in team.roster if not set(p.eligibleSlots) & pitcher_slots]

    # Use hard-coded HITTER_SLOTS config instead of deriving from current roster
    # state, which is unreliable when players happen to be sitting on the bench.
    total_hitter_slots  = len(HITTER_SLOTS)
    total_bench_slots   = len(hitters) - total_hitter_slots
    total_pitcher_slots = sum(1 for p in team.roster if p.lineupSlot in pitcher_slots)

    active_slot_counts = Counter(HITTER_SLOTS)
    slot_summary = ", ".join(
        f"{active_slot_counts[slot]}x {slot}" if active_slot_counts.get(slot, 0) > 1 else slot
        for slot in slot_order
        if active_slot_counts.get(slot, 0) > 0
    )

    # Get this week's matchup opponent
    current_week = league.currentMatchupPeriod
    matchup = next(
        (m for m in league.box_scores(current_week)
         if m.home_team.team_id == TEAM_ID or m.away_team.team_id == TEAM_ID),
        None
    )
    opponent_info = ""
    if matchup:
        opp = matchup.away_team if matchup.home_team.team_id == TEAM_ID else matchup.home_team
        opponent_info = f"This week's opponent: {opp.team_name} (record: {opp.wins}-{opp.losses})"

    # Build hitter table: name | eligible positions | has game today
    hitter_lines = []
    for p in hitters:
        eligible = [s for s in slot_order if s in p.eligibleSlots]
        playing  = "✓" if not p.injured and getattr(p, "injuryStatus", "ACTIVE") == "ACTIVE" else "✗"
        hitter_lines.append(f"  {p.name:<25} eligible: {', '.join(eligible):<20} playing today: {playing}")
    hitter_table = "\n".join(hitter_lines)

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points
{opponent_info}

Available hitter slots: {slot_summary}
(Pitchers fill {total_pitcher_slots} P slots and will be handled separately — do NOT include them.)

Hitters on your roster:
{hitter_table}

Full roster stats (for reference):
{json.dumps(roster, indent=2)}

Task: Choose exactly {total_hitter_slots} hitters to start and exactly {total_bench_slots} to bench.

DECISION PRIORITY (follow this order strictly):
1. POINTS FIRST: Rank all hitters by expected points today. Use season PPG and recent
   performance (last 15 days) as your primary guide. Start the {total_hitter_slots}
   highest-scoring hitters. A high-PPG player must NEVER be benched in favour of a
   lower-PPG player just because the lower-PPG player is more positionally versatile.
2. GAME TODAY: Among similar-PPG players, prefer those with a game today (marked ✓)
   over those without (marked ✗).
3. SLOT FIT (constraint, not a selection criterion): After choosing your top
   {total_hitter_slots} hitters by points, verify they can be legally assigned to the
   available slots. Only if there is a genuine slot conflict (e.g. two C-only players
   for one C slot) should you swap the lowest-PPG conflicting player to the bench.
   UTIL is a universal slot — use it to keep high-PPG players active.

Rules you MUST follow:
- Your {total_hitter_slots} ACTIVE hitters must be assignable to the available slots above with no conflicts.
  For example, if there is only 1x 1B slot, you cannot start two pure-1B players — one must be eligible
  for another slot (UTIL) or be benched.
- Every hitter must appear in exactly one list — {total_hitter_slots} ACTIVE, {total_bench_slots} BENCH.
- Do NOT factor in positional versatility when deciding who starts. A player eligible at
  only one position is just as startable as a multi-position player if their PPG is higher.

Respond with exactly this structure and nothing else:
ACTIVE: Player Name, Player Name, ... (exactly {total_hitter_slots} names)
BENCH: Player Name, Player Name, ... (exactly {total_bench_slots} names)
REASONING: One or two sentences explaining the key decisions.

Only use player names exactly as they appear above. Do not add any other text.
"""


def build_pitcher_prompt(team) -> str:
    pitcher_slots = {"SP", "RP", "P"}
    pitchers = [p for p in team.roster if set(p.eligibleSlots) & pitcher_slots]

    total_P_slots     = sum(1 for p in pitchers if p.lineupSlot in pitcher_slots)
    total_bench_slots = len(pitchers) - total_P_slots

    pitcher_lines = []
    for p in pitchers:
        roles  = [s for s in ["SP", "RP"] if s in p.eligibleSlots]
        status = getattr(p, "injuryStatus", "ACTIVE")
        healthy = "✓" if not p.injured else "✗"
        pitcher_lines.append(
            f"  {p.name:<25} role: {', '.join(roles):<8} healthy: {healthy}  status: {status}"
        )
    pitcher_table = "\n".join(pitcher_lines)
    stats = [serialize_player(p) for p in pitchers]

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points

You have {total_P_slots} active pitcher slots and {total_bench_slots} bench spots.

Pitchers on your roster:
{pitcher_table}

Full pitcher stats (for reference):
{json.dumps(stats, indent=2)}

Task: Choose exactly {total_P_slots} pitchers for the active rotation and exactly {total_bench_slots} to bench.
Prefer SP that are projected to start today, otherwise select RP. Prioritize healthy pitchers with strong recent stats and projected points.

Respond with exactly this structure and nothing else:
ROTATION: Player Name, Player Name, ... (exactly {total_P_slots} names)
PBENCH: Player Name, Player Name, ... (exactly {total_bench_slots} names)
REASONING: One or two sentences explaining the key decisions.

Only use player names exactly as they appear above. Do not add any other text.
"""


def build_waiver_prompt(team, league, undroppable: set[str] = None,
                        prior_failures: list[dict] = None) -> str:
    pitcher_slots = {"SP", "RP", "P"}
    hitters       = [p for p in team.roster if not set(p.eligibleSlots) & pitcher_slots]
    roster        = [serialize_player(p) for p in team.roster]
    free_agents   = [serialize_player(p) for p in league.free_agents(size=25)]

    anchor_lines = []
    for slot in set(HITTER_SLOTS):
        eligible = [p for p in hitters if slot in p.eligibleSlots]
        if len(eligible) == 1:
            anchor_lines.append(f"  {slot}: {eligible[0].name} (only eligible player — do NOT drop)")
    anchor_section = (
        "Positional anchors — dropping these creates an unfillable hole:\n" + "\n".join(anchor_lines)
        if anchor_lines else "No positional anchors — all slots have multiple eligible players."
    )

    if undroppable:
        undrop_list = ", ".join(sorted(undroppable))
        undrop_section = (
            f"UNDROPPABLE players (ESPN restriction — these CANNOT be dropped under any "
            f"circumstances):\n  {undrop_list}"
        )
    else:
        undrop_section = ""

    if prior_failures:
        failure_lines = []
        for f in prior_failures:
            failure_lines.append(f"  - ADD {f['add']} / DROP {f['drop']} FAILED: {f['error']}")
        failure_section = (
            "PREVIOUS ATTEMPTS THAT FAILED (do NOT repeat these moves — suggest different ones):\n"
            + "\n".join(failure_lines)
        )
    else:
        failure_section = ""

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points

Active hitter slots: {", ".join(HITTER_SLOTS)}

{anchor_section}

{undrop_section}

{failure_section}

Current roster (with PPG and total points):
{json.dumps(roster, indent=2)}

Top 25 available free agents (with PPG and total points):
{json.dumps(free_agents, indent=2)}

Task: Recommend up to 3 waiver/free-agent moves that would improve our team.
Only suggest players from the free agent list above — they must have the FREE AGENT
designation, not be on waivers or another team.

EVALUATION CRITERIA (in order of importance):
1. UNDROPPABLE: Never suggest dropping an undroppable player. ESPN will reject the move.
2. POINTS: Compare the PPG and total_points fields above. Only suggest a move if
   the added player clearly outperforms the dropped player. A hot-streak player with
   slightly lower season PPG can justify a swap, but weigh the risk: a dropped high-PPG
   player will likely be claimed by an opponent and used against you.
3. SP STREAMING: Prefer SP streamers with starts this week — pitching points are king.
4. POSITIONAL ANCHORS: Never drop a positional anchor unless the player being added can
   also fill that slot.
5. Positional versatility is NOT a reason to add or keep a player. Only points matter.
6. Never use subjective labels like "boom/bust" to justify dropping a high-PPG player.

If no moves are worthwhile, say: NO MOVES

Otherwise respond with exactly this structure and nothing else:
ADD: Player Name to Add | DROP: Player Name to Drop | REASON: Why this improves the team
ADD: Player Name to Add | DROP: Player Name to Drop | REASON: Why this improves the team

Only use player names exactly as they appear above. Do not add any other text.
"""


def build_il_cleanup_prompt(team, player_off_il, undroppable: set[str] = None) -> str:
    """Build prompt asking Gemini who to drop to make room for a player returning from IL."""
    roster = [serialize_player(p) for p in team.roster if p.name != player_off_il.name]
    returning = serialize_player(player_off_il)

    pitcher_slots = {"SP", "RP", "P"}
    hitters = [p for p in team.roster
               if not set(p.eligibleSlots) & pitcher_slots and p.name != player_off_il.name]

    anchor_lines = []
    for slot in set(HITTER_SLOTS):
        eligible = [p for p in hitters if slot in p.eligibleSlots]
        if len(eligible) == 1:
            anchor_lines.append(f"  {slot}: {eligible[0].name} (only eligible player — do NOT drop)")
    anchor_section = (
        "Positional anchors — dropping these creates an unfillable hole:\n" + "\n".join(anchor_lines)
        if anchor_lines else "No positional anchors — all slots have multiple eligible players."
    )

    if undroppable:
        undrop_list = ", ".join(sorted(undroppable))
        undrop_section = (
            f"UNDROPPABLE players (ESPN restriction — these CANNOT be dropped under any "
            f"circumstances):\n  {undrop_list}"
        )
    else:
        undrop_section = ""

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points

{player_off_il.name} is returning from the Injured List and must be moved to the active
roster. The roster is currently full, so one player must be dropped to make room.

Returning player stats:
{json.dumps(returning, indent=2)}

{anchor_section}

{undrop_section}

Current roster (excluding {player_off_il.name}):
{json.dumps(roster, indent=2)}

Task: Choose exactly ONE player to drop from the roster to make room for {player_off_il.name}.

Rules:
- NEVER drop an undroppable player. ESPN will reject the move.
- Choose the player with the lowest expected points contribution going forward.
- Do NOT drop a positional anchor unless {player_off_il.name} can also fill that slot.
- Prefer dropping bench players over active starters.
- Never factor in positional versatility — only points production matters.

Respond with exactly this structure and nothing else:
DROP: Player Name
REASONING: One sentence explaining why this player is the best drop.

Only use player names exactly as they appear above. Do not add any other text.
"""
