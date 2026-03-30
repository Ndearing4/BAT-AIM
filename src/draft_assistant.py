"""
ARIA Live Draft Assistant
Connects directly to your ESPN draft and recommends picks in real time.

Usage:
    # Set env vars first:
    # $env:ANTHROPIC_API_KEY = "your-key"
    # $env:ESPN_LEAGUE_ID    = "123456"
    # $env:ESPN_S2           = "your-espn_s2-cookie"
    # $env:ESPN_SWID         = "{your-swid-cookie}"

    python src/draft_assistant.py

Run this script whenever it's the AI team's turn to pick.
It will fetch the live draft board and recommend the best available player.
"""

import os
import json
import anthropic
from espn_api.baseball import League
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
LEAGUE_ID     = int(os.environ["ESPN_LEAGUE_ID"])
ESPN_S2       = os.environ["ESPN_S2"]
SWID          = os.environ["ESPN_SWID"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
YEAR          = datetime.now().year
# ─────────────────────────────────────────────────────────────────────────────

GM_STRATEGY = """
You are ARIA, an expert fantasy baseball drafter for a Head-to-Head Points league.

H2H Points drafting philosophy:
- Rounds 1-3: Elite SP or elite OF/1B/3B with high points ceilings
- Rounds 4-6: High-volume SP — aces rack up Ks and wins = big points
- Prioritize SP depth over closers (saves score minimal points vs SP)
- Target multi-position eligible hitters (2B/SS, 1B/3B) for roster flexibility
- High K/BB ratio pitchers > groundball pitchers in points leagues
- Streaming SPs is viable late — don't reach for mid-tier SP early
- Never draft a closer before round 10 unless it's a clear elite closer
"""


def get_league() -> League:
    return League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)


def get_live_draft_state(league: League, ai_team_id: int, pick_num: int) -> dict:
    """
    Pull the current draft state from ESPN.
    Returns already-drafted players and available players.
    """
    draft = league.draft  # List of Pick objects already made

    my_roster = []
    for pick in draft:
        if pick.team.team_id == ai_team_id:
            my_roster.append({
                "name":     pick.playerName,
                "round":    pick.roundNum,
                "position": getattr(pick, "position", "?"),
            })

    # During a live ESPN draft, free_agents() returns ONLY truly undrafted
    # players — trust this list directly rather than filtering manually,
    # since league.draft can lag and cause already-drafted players to slip through.
    available_raw = league.free_agents(size=100)

    # ── FIX: Number each player with their ESPN rank so Claude understands
    # where they fall in overall value. This prevents Claude from treating
    # the #1 available as an elite early-round pick — their absolute rank
    # reflects how many players are already gone above them.
    num_teams   = len(league.teams)
    total_picks = len(draft)  # how many players have already been taken

    available = []
    for i, p in enumerate(available_raw):
        # Approximate their overall ESPN rank based on how many are gone + their
        # position in the remaining list
        approx_overall_rank = total_picks + i + 1
        available.append({
            "espn_overall_rank": approx_overall_rank,  # KEY FIX: makes round context explicit
            "name":      p.name,
            "positions": p.eligibleSlots,
            "pro_team":  p.proTeam,
            "injured":   p.injured,
        })

    current_round = ((pick_num - 1) // num_teams) + 1
    current_pick  = pick_num

    # ── FIX: Summarize roster by position so Claude can identify gaps
    roster_positions = {}
    for player in my_roster:
        pos = player.get("position", "?")
        roster_positions[pos] = roster_positions.get(pos, 0) + 1

    return {
        "my_roster":          my_roster,
        "roster_by_position": roster_positions,
        "available":          available[:60],  # Top 60 available
        "current_round":      current_round,
        "current_pick":       current_pick,
        "total_picked":       total_picks,
        "num_teams":          num_teams,
    }


def ask_claude_for_pick(state: dict) -> str:
    # ── FIX: Explicitly tell Claude how many players are already off the board
    # so it doesn't anchor on available[0] as if it were an elite pick.
    already_gone = state['total_picked']
    num_teams    = state['num_teams']

    prompt = f"""
{GM_STRATEGY}

=== DRAFT CONTEXT ===
- Round {state['current_round']} of the draft, overall pick #{state['current_pick']}
- {already_gone} players have ALREADY BEEN DRAFTED across all teams
- This is a {num_teams}-team league
- The "espn_overall_rank" field on each player below shows their approximate rank
  out of ALL players — a rank of {already_gone + 1} means they are the very next
  player off the board, not an elite top-10 talent.

=== MY CURRENT ROSTER ({len(state['my_roster'])} players drafted so far) ===
Position breakdown: {json.dumps(state['roster_by_position'])}
Full roster:
{json.dumps(state['my_roster'], indent=2) if state['my_roster'] else "No players yet."}

=== PLAYERS STILL AVAILABLE (ESPN rank order, {already_gone} already gone) ===
{json.dumps(state['available'], indent=2)}

=== YOUR TASK ===
Recommend a pick appropriate for Round {state['current_round']} given:
- Who is realistically available at this stage of the draft (NOT top-overall talent)
- My current roster composition and gaps (see position breakdown above)
- H2H Points league strategy

Provide:
1. ✅ TOP PICK — Name, Position, ESPN rank #{{}}, and why this player fits at Round {state['current_round']}
2. 🔁 BACKUP — Second choice with brief reason
3. 💡 One sentence on the biggest roster need for my next pick

Only recommend players explicitly listed above. Be direct and decisive.
"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


def main():
    print("\n🤖 ARIA Live Draft Assistant")
    print("=" * 45)

    # Get team ID
    try:
        ai_team_id = int(os.environ["ESPN_TEAM_ID"])
    except (KeyError, ValueError):
        ai_team_id = int(input("Enter the AI team's ESPN Team ID: ").strip())

    print("Connecting to ESPN draft...\n")
    try:
        league = get_league()
    except Exception as e:
        print(f"❌ Could not connect to ESPN: {e}")
        print("Check your ESPN_S2, ESPN_SWID, and ESPN_LEAGUE_ID values.")
        return

    print(f"✅ Connected to: {league.settings.name}")
    print(f"   {len(league.teams)} teams | Season {YEAR}\n")

    # Ask for starting overall pick number (in case script started mid-draft)
    start = input("What is the current overall pick number? (press Enter to start at 1): ").strip()
    overall_pick = int(start) if start.isdigit() else 1

    while True:
        input("⏳ Press ENTER when it's time for ARIA to pick (or Ctrl+C to quit)...\n")

        print("📡 Fetching live draft board from ESPN...")
        try:
            state = get_live_draft_state(league, ai_team_id, overall_pick)
        except Exception as e:
            print(f"❌ Error fetching draft state: {e}")
            print("Try again or check your connection.\n")
            continue

        print(f"   Round {state['current_round']} | Pick #{state['current_pick']} | {len(state['available'])} players available\n")
        print("🧠 ARIA is thinking...\n")

        recommendation = ask_claude_for_pick(state)
        print("─" * 45)
        print(recommendation)
        print("─" * 45)

        picked = input("\nWho did ESPN auto-draft / you selected? (press Enter to skip logging): ").strip()
        if picked:
            print(f"✅ Logged: {picked}\n")
        else:
            print()
        overall_pick += len(league.teams)  # Advance by one full round (next AI turn)


if __name__ == "__main__":
    main()