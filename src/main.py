"""
AI Fantasy Baseball Manager
Manages an ESPN fantasy baseball team using Claude AI for decision-making.
Optimized for Head-to-Head Points format.
"""

import time
from datetime import datetime

from config import TEAM_ID
from espn_client import get_league, fetch_undroppable_names
from llm import ask_gemini
from prompts import build_lineup_prompt, build_pitcher_prompt, build_waiver_prompt
from update_lineup import apply_lineup
from update_waivers import apply_waivers, manage_il


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

    if mode in ("waivers", "all"):
        print("🔒 Fetching undroppable list from ESPN...")
        undroppable = fetch_undroppable_names()
        if undroppable:
            print(f"   Undroppable: {', '.join(sorted(undroppable))}")

        print("🏥 Managing IL slots...")
        manage_il(team, league, undroppable)

        all_failures = []
        for attempt in range(3):
            label = f" (retry {attempt}/2)" if attempt > 0 else ""
            print(f"🧠 Asking Gemini for waiver recommendations{label}...")
            waiver_prompt   = build_waiver_prompt(team, league, undroppable, all_failures or None)
            waiver_response = ask_gemini(waiver_prompt, mode="waivers")
            print(waiver_response)

            if waiver_response.strip().upper() == "NO MOVES":
                break

            failures = apply_waivers(team, league, waiver_response, undroppable)
            if not failures:
                break
            all_failures.extend(failures)
            print(f"⚠️  {len(failures)} move(s) failed (attempt {attempt + 1}/3), "
                  f"retrying with failure context...")

    if mode == "all":
        # Re-fetch roster to reflect IL and waiver changes before setting lineup
        print("🔄 Refreshing roster after roster moves...")
        time.sleep(15)
        league = get_league()
        team   = next(t for t in league.teams if t.team_id == TEAM_ID)

    if mode in ("lineup", "all"):
        print("🧠 Asking Gemini for pitcher rotation...")
        pitcher_prompt    = build_pitcher_prompt(team)
        pitcher_response  = ask_gemini(pitcher_prompt, mode="pitchers")
        print(pitcher_response)

        print("🧠 Asking Gemini for lineup decisions...")
        lineup_prompt    = build_lineup_prompt(team, league)
        lineup_response  = ask_gemini(lineup_prompt, mode="lineup")
        print(lineup_response)
        apply_lineup(team, league, lineup_response, pitcher_response)

    print("✅ ARIA run complete.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    run(mode)
