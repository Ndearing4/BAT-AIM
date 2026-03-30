"""
AI Fantasy Baseball Manager
Manages an ESPN fantasy baseball team using Claude AI for decision-making.
Optimized for Head-to-Head Points format.
"""

import os
import time
import json
import requests
from dotenv import load_dotenv
from google import genai
from espn_api.baseball import League
from datetime import datetime, date

# Load .env when running locally — no-op in GitHub Actions where secrets are
# injected as real environment variables
load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
LEAGUE_ID     = int(os.environ["ESPN_LEAGUE_ID"])
TEAM_ID       = int(os.environ["ESPN_TEAM_ID"])        # The AI team's ID in the league
ESPN_S2       = os.environ["ESPN_S2"]                  # ESPN auth cookie
SWID          = os.environ["ESPN_SWID"]                # ESPN auth cookie
GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
YEAR          = datetime.now().year

# ─── MOCK CONFIG ───────────────────────────────────────────────────────────────
# Set USE_LLM = False to skip Gemini entirely and use the mock responses below.
# Useful for testing ESPN API calls without burning quota.
# Swap in real player names from your roster/waiver wire before running.
USE_LLM = False

MOCK_LINEUP_RESPONSE = """
ACTIVE: Adley Rutschman, Vinnie Pasquantino, Jazz Chisholm Jr., Jose Ramirez, Masyn Winn, Riley Greene, Bryan Reynolds, Wyatt Langford, Kyle Schwarber
BENCH: Bryce Harper, Tyler Stephenson, Kyle Tucker
REASONING: Mock lineup for ESPN API testing — swapping Kyle Tucker to bench in favour of Riley Greene.
"""

MOCK_WAIVER_RESPONSE = """
NO MOVES
"""
# ───────────────────────────────────────────────────────────────────────────────


GM_PERSONA = """
You are ARIA (Automated Roster Intelligence Agent), an aggressive fantasy baseball GM 
optimized for Head-to-Head Points leagues. Your philosophy:
- Maximize total points each week, not categories
- Stream starting pitchers aggressively — SP points are king
- Prioritize high-floor hitters over boom/bust options
- Never leave empty roster slots
- Make waiver moves boldly; points on the bench are wasted
"""
# ───────────────────────────────────────────────────────────────────────────────


def get_league() -> League:
    return League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)


def serialize_player(p, include_stats=True) -> dict:
    """Convert ESPN player object to a clean dict for the AI."""
    d = {
        "name":     p.name,
        "position": p.eligibleSlots,
        "team":     p.proTeam,
        "injured":  p.injured,
        "status":   getattr(p, "injuryStatus", "ACTIVE"),
    }
    if include_stats and hasattr(p, "stats"):
        # Last 15 days stats if available
        d["recent_stats"] = p.stats.get(f"{YEAR}_last_15", {})
        d["season_stats"]  = p.stats.get(str(YEAR), {})
    return d


def build_lineup_prompt(team, league) -> str:
    roster = [serialize_player(p) for p in team.roster]
    
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

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points
{opponent_info}

Current roster:
{json.dumps(roster, indent=2)}

Task: Decide which players should be active vs on the bench today.

Respond with exactly this structure and nothing else:
ACTIVE: Player Name, Player Name, Player Name, ...
BENCH: Player Name, Player Name, ...
REASONING: One or two sentences explaining the key decisions.

Only use player names exactly as they appear above. Do not add any other text.
"""


def build_waiver_prompt(team, league) -> str:
    roster      = [serialize_player(p) for p in team.roster]
    free_agents = [serialize_player(p) for p in league.free_agents(size=25)]

    return f"""
{GM_PERSONA}

Today is {date.today().strftime('%A, %B %d, %Y')}.
League format: Head-to-Head Points

Current roster:
{json.dumps(roster, indent=2)}

Top 25 available free agents:
{json.dumps(free_agents, indent=2)}

Task: Recommend up to 3 waiver/free-agent moves that would improve our team.
Only suggest moves that are clear upgrades. Prefer SP streamers with starts this week.
If no moves are worthwhile, say: NO MOVES

Otherwise respond with exactly this structure and nothing else:
ADD: Player Name to Add | DROP: Player Name to Drop | REASON: Why this improves the team
ADD: Player Name to Add | DROP: Player Name to Drop | REASON: Why this improves the team

Only use player names exactly as they appear above. Do not add any other text.
"""


def _extract_retry_delay(e: Exception) -> float | None:
    """Pull retryDelay seconds out of a Gemini API exception, if present.

    Actual structure: e.details['error']['details'] is a list of typed objects.
    The RetryInfo entry looks like:
        {'@type': '...google.rpc.RetryInfo', 'retryDelay': '20s'}
    """
    try:
        detail_list = e.details["error"]["details"]
        for item in detail_list:
            if "RetryInfo" in item.get("@type", "") and "retryDelay" in item:
                return float(item["retryDelay"].rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None


def ask_gemini(prompt: str, mode: str = "lineup", retries: int = 3) -> str:
    """Send prompt to Gemini and return raw text response with retry logic.

    If USE_LLM is False, returns the appropriate mock response instead.
    On 429 quota errors, honours the retryDelay the API returns.
    On other errors, falls back to a simple 10/20/30s backoff.
    """
    if not USE_LLM:
        mock = MOCK_LINEUP_RESPONSE if mode == "lineup" else MOCK_WAIVER_RESPONSE
        print(f"🧪 USE_LLM=False — using mock {mode} response")
        return mock.strip()

    client = genai.Client(api_key=GEMINI_KEY)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            is_429  = getattr(e, "code", None) == 429
            is_last = attempt >= retries - 1

            if is_last:
                raise

            if is_429:
                delay = _extract_retry_delay(e)
                if delay is not None:
                    wait = delay + 2  # 2s buffer on top of server suggestion
                    print(f"⚠️  Gemini quota error (attempt {attempt + 1}): rate limit hit. "
                          f"Retrying in {wait:.0f}s (server-suggested {delay:.0f}s + 2s buffer)...")
                else:
                    wait = 60  # conservative fallback if no delay hint
                    print(f"⚠️  Gemini quota error (attempt {attempt + 1}): rate limit hit, "
                          f"no retryDelay in response. Retrying in {wait}s...")
            else:
                wait = 10 * (attempt + 1)
                print(f"⚠️  Gemini error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")

            time.sleep(wait)


def parse_lineup_response(raw: str) -> dict:
    """Parse Gemini's plain-text lineup response into a structured dict."""
    result = {"active": [], "bench": [], "reasoning": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("ACTIVE:"):
            result["active"] = [n.strip() for n in line[7:].split(",") if n.strip()]
        elif line.upper().startswith("BENCH:"):
            result["bench"] = [n.strip() for n in line[6:].split(",") if n.strip()]
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line[10:].strip()
    return result


# Slot priority order for auto-assignment
SLOT_PRIORITY = ["C", "1B", "2B", "3B", "SS", "OF", "UTIL", "SP", "RP", "P"]

# ESPN lineup slot ID map (confirmed from network traffic)
# IL (17) is assumed based on standard ESPN slot numbering — unconfirmed since
# no IL-eligible players are available to test a move with DevTools.
SLOT_ID = {
    "C":    0,
    "1B":   1,
    "2B":   2,
    "3B":   3,
    "SS":   4,
    "OF":   5,
    "DH":   10,
    "UTIL": 12,
    "SP":   13,
    "RP":   13,
    "P":    13,
    "BE":   16,
    "IL":   17,  # unconfirmed — update once a real IL move can be captured
}


def espn_set_lineup(items: list[dict], scoring_period: int):
    """POST one ESPN lineup transaction per item to the write API.

    Sends each move as a separate request with a single-item `items` list,
    matching the exact payload format captured from DevTools. Swaps (e.g.
    Tucker OF→BE and Greene BE→OF) are sent sequentially in the same call
    so ESPN sees both sides of the move.

    Each item in `items` should be:
        {"playerId": int, "fromLineupSlotId": int, "toLineupSlotId": int}
    """
    url = (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/flb"
        f"/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}/transactions/"
    )
    cookies = {"espn_s2": ESPN_S2, "SWID": SWID}
    headers = {"Content-Type": "application/json"}

    # Send players leaving active slots first (toLineupSlotId == BE),
    # then players entering active slots. This prevents ESPN from rejecting
    # a move into a slot that is still occupied.
    bench_slot = SLOT_ID["BE"]
    outgoing = [i for i in items if i["toLineupSlotId"] == bench_slot]
    incoming = [i for i in items if i["toLineupSlotId"] != bench_slot]
    ordered  = outgoing + incoming

    responses = []
    for item in ordered:
        payload = {
            "isLeagueManager": False,
            "teamId":          TEAM_ID,
            "type":            "ROSTER",
            "memberId":        SWID,
            "scoringPeriodId": scoring_period,
            "executionType":   "EXECUTE",
            "items":           [{"type": "LINEUP", **item}],
        }
        resp = requests.post(url, json=payload, cookies=cookies, headers=headers)
        print(f"   ESPN response {resp.status_code} for playerId {item['playerId']}")
        resp.raise_for_status()
        responses.append(resp.json())
    return responses


def _make_move(player, from_slot: str, to_slot: str) -> dict:
    """Build a single ESPN transaction item for a slot change."""
    return {
        "playerId":         player.playerId,
        "fromLineupSlotId": SLOT_ID.get(from_slot, 16),
        "toLineupSlotId":   SLOT_ID.get(to_slot,   16),
    }


def _submit(items: list[dict], scoring_period: int, label: str = ""):
    """Submit a batch of moves and log results. Skips if empty."""
    if not items:
        return
    if label:
        print(f"  [{label}]")
    espn_set_lineup(items, scoring_period)


def apply_lineup(team, league, raw_response: str):
    """Apply lineup decisions in three ordered phases to avoid slot conflicts.

    Phase 1 — Bench outgoing players first, freeing active slots.
    Phase 2 — Reposition active players who need a different slot
              (e.g. Jazz 2B→3B). Uses BE as a waypoint for triangular swaps.
    Phase 3 — Promote bench players into the newly freed active slots.
    Pitcher pass — Ensure only today's starters occupy P slots; swap in
              non-starting pitchers from the bench to replace any starters
              who are not pitching today.
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

    def find_open_slot(player, occupied: set[str]) -> str | None:
        for slot in SLOT_PRIORITY:
            if slot in player.eligibleSlots and slot not in occupied:
                return slot
        return None

    # Track which active slots are occupied as we go
    occupied: set[str] = {
        slot for name, slot in current_slot_by_name.items()
        if slot in active_slots
    }

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
        phase1.append(_make_move(player, from_slot, "BE"))
        occupied.discard(from_slot)
        current_slot_by_name[name] = "BE"
        print(f"   {name}: {from_slot} → BE")
    _submit(phase1, scoring_period, "bench outgoing")

    # ── Phase 2: reposition active hitters who need a different slot ────────
    print("Phase 2: repositioning active players...")
    phase2 = []
    for name in active_names:
        player = roster_by_name.get(name)
        if not player or is_pitcher(player):
            continue
        current = current_slot_by_name.get(name, "BE")
        # Determine best target slot for this player
        target = find_open_slot(player, occupied - ({current} if current != "BE" else set()))
        if target is None:
            print(f"⚠️  No open slot for {name}, leaving on bench.")
            continue
        if current == target:
            continue  # already in the right place
        if current != "BE" and target in occupied:
            # Triangular conflict — move via BE as a waypoint
            phase2.append(_make_move(player, current, "BE"))
            phase2.append(_make_move(player, "BE", target))
            print(f"   {name}: {current} → BE → {target} (via waypoint)")
        else:
            phase2.append(_make_move(player, current, target))
            print(f"   {name}: {current} → {target}")
        occupied.discard(current)
        occupied.add(target)
        current_slot_by_name[name] = target
    _submit(phase2, scoring_period, "reposition active")

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
        phase3.append(_make_move(player, "BE", target))
        occupied.add(target)
        current_slot_by_name[name] = target
        print(f"   {name}: BE → {target}")
    _submit(phase3, scoring_period, "promote from bench")

    # ── Pitcher pass: bench non-starters, promote starters ─────────────────
    # print("Pitcher pass: optimising P slots...")
    # pitchers = [p for p in team.roster if is_pitcher(p)]

    # # A pitcher is "starting today" if their proTeamId appears as the home/away
    # # starter — use the injuryStatus field as a proxy: ESPN marks non-starters
    # # as "DTDAY" or similar, but the most reliable signal available without a
    # # separate schedule API call is whether the player has a game today
    # # (p.injured == False and status == "ACTIVE"). Flag starters vs non-starters:
    # starters     = [p for p in pitchers if not p.injured and
    #                 getattr(p, "injuryStatus", "ACTIVE") == "ACTIVE"]
    # non_starters = [p for p in pitchers if p not in starters]

    # # Bench any non-starters currently in active P slots
    # pitcher_phase1 = []
    # for p in non_starters:
    #     slot = current_slot_by_name.get(p.name, "BE")
    #     if slot != "BE":
    #         pitcher_phase1.append(_make_move(p, slot, "BE"))
    #         occupied.discard(slot)
    #         current_slot_by_name[p.name] = "BE"
    #         print(f"   {p.name}: {slot} → BE (not starting)")
    # _submit(pitcher_phase1, scoring_period, "bench non-starting pitchers")

    # # Promote starters sitting on the bench into open P slots
    # pitcher_phase2 = []
    # for p in starters:
    #     if current_slot_by_name.get(p.name) == "BE":
    #         target = find_open_slot(p, occupied)
    #         if target:
    #             pitcher_phase2.append(_make_move(p, "BE", target))
    #             occupied.add(target)
    #             current_slot_by_name[p.name] = target
    #             print(f"   {p.name}: BE → {target} (starting today)")
    #         else:
    #             print(f"⚠️  No open P slot for starter {p.name}")
    # _submit(pitcher_phase2, scoring_period, "promote starting pitchers")

    # print(f"✅ Lineup complete. Reasoning: {decision['reasoning']}")


def parse_waiver_response(raw: str) -> list[dict]:
    """Parse Gemini's plain-text waiver response into a list of move dicts."""
    if raw.strip().upper() == "NO MOVES":
        return []
    moves = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("ADD:"):
            continue
        try:
            parts = {k.strip(): v.strip()
                     for part in line.split("|")
                     for k, v in [part.split(":", 1)]}
            moves.append({
                "add":    parts.get("ADD", ""),
                "drop":   parts.get("DROP", ""),
                "reason": parts.get("REASON", ""),
            })
        except ValueError:
            print(f"⚠️  Could not parse waiver line: {line}")
    return moves


def apply_waivers(team, league, raw_response: str):
    """Parse Gemini's waiver recommendations and apply moves to ESPN."""
    moves = parse_waiver_response(raw_response)

    if not moves:
        print("✅ No waiver moves needed this cycle.")
        return

    # Cache free agents once rather than re-fetching per move
    available = league.free_agents(size=200)

    for move in moves:
        add_name  = move["add"]
        drop_name = move["drop"]
        reason    = move["reason"]

        add_player  = next((p for p in available if p.name == add_name), None)
        drop_player = next((p for p in team.roster if p.name == drop_name), None)

        if add_player and drop_player:
            team.add_player(add_player, drop_player)
            print(f"✅ Added {add_name}, dropped {drop_name}. Reason: {reason}")
        else:
            print(f"⚠️  Could not execute move: Add {add_name} / Drop {drop_name} — player not found.")


def run(mode: str = "all"):
    """
    mode: 'lineup'  — only set lineup
          'waivers' — only process waivers  
          'all'     — both (default)
    """
    print(f"🤖 ARIA starting up — mode: {mode} — {datetime.now()}")
    league = get_league()
    team   = next(t for t in league.teams if t.team_id == TEAM_ID)
    print(f"📋 Managing: {team.team_name}")

    if mode in ("lineup", "all"):
        print("🧠 Asking Gemini for lineup decisions...")
        lineup_prompt    = build_lineup_prompt(team, league)
        lineup_response  = ask_gemini(lineup_prompt, mode="lineup")
        print(lineup_response)
        apply_lineup(team, league, lineup_response)

    if mode == "all":
        print("⏳ Waiting 15s before waiver call...")
        time.sleep(15)

    if mode in ("waivers", "all"):
        print("🧠 Asking Gemini for waiver recommendations...")
        waiver_prompt    = build_waiver_prompt(team, league)
        waiver_response  = ask_gemini(waiver_prompt, mode="waivers")
        print(waiver_response)
        apply_waivers(team, league, waiver_response)

    print("✅ ARIA run complete.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    run(mode)