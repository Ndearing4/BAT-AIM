"""
Configuration constants for the AI Fantasy Baseball Manager.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

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

# ─── ROSTER CONFIG ─────────────────────────────────────────────────────────────
# Hard-coded active hitter slots for this league. Update if league settings change.
# List each slot once per slot that exists (e.g. three OF slots → three "OF" entries).
# Pitchers are handled separately — do NOT include SP/RP/P here.
HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
IL_SLOTS     = 3     # Number of IL slots in the league
BENCH_SLOTS  = 3     # Number of bench slots in the league
ROSTER_SIZE  = 22    # Max roster size (16 active + 3 bench + 3 IL)

# ─── MOCK CONFIG ───────────────────────────────────────────────────────────────
# Set USE_LLM = False to skip Gemini entirely and use the mock responses below.
# Useful for testing ESPN API calls without burning quota.
# Swap in real player names from your roster/waiver wire before running.
USE_LLM = True

MOCK_LINEUP_RESPONSE = """
ACTIVE: Adley Rutschman, Bryce Harper, Jazz Chisholm Jr., Jose Ramirez, Masyn Winn, Riley Greene, Wyatt Langford, Kyle Schwarber
BENCH: Vinnie Pasquantino, Tyler Stephenson, Bryan Reynolds, Kyle Tucker
REASONING: Mock lineup for ESPN API testing — swapping Kyle Tucker to bench in favour of Riley Greene.
"""

MOCK_PITCHER_RESPONSE = """
ROTATION: Logan Webb, Jesus Luzardo, Jacob deGrom, Dylan Cease, Edwin Diaz, Michael King, Jack Flaherty
PBENCH: Shota Imanaga
REASONING: Mock pitcher rotation — starting all seven healthiest pitchers, benching Imanaga.
"""

MOCK_WAIVER_RESPONSE = """
ADD: Brandon Lowe | DROP: Jazz Chisholm Jr. | REASON: Mock waiver claim
"""

MOCK_IL_CLEANUP_RESPONSE = """
DROP: Bryan Reynolds
REASONING: Mock IL cleanup — dropping lowest-PPG bench player to make room.
"""

# ─── GM PERSONA ────────────────────────────────────────────────────────────────
GM_PERSONA = """
You are ARIA (Automated Roster Intelligence Agent), an aggressive fantasy baseball GM
optimized for Head-to-Head Points leagues. Your philosophy:
- Maximize total points each week, not categories
- Start your highest points-per-game (PPG) players — benching a high scorer is the
  single most costly mistake you can make
- Stream starting pitchers aggressively — SP points are king
- Trust the numbers: use season PPG and recent performance (last 15 days) to judge
  player value. Never override stats with subjective labels like "boom/bust."
  A hot-streak player with slightly lower season PPG can be worth a pickup, but
  consider the risk: a dropped high-PPG player may be claimed by an opponent.
- Never leave empty roster slots
- Make waiver moves boldly; points on the bench are wasted
- Positional versatility is NOT a factor when choosing who starts — only total
  expected points matters. Versatility only matters when assigning players to slots
  after the lineup is already chosen.
"""

# ─── SLOT MAPPINGS ─────────────────────────────────────────────────────────────
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
    "IL":   17,
}

IL_ELIGIBLE_STATUSES = {"TEN_DAY_DL", "FIFTEEN_DAY_DL", "SIXTY_DAY_DL", "INJURY_RESERVE", "SUSPENSION", "OUT"}
