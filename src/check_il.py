"""List all players on the roster with their IL-related status."""
import os
from dotenv import load_dotenv
from espn_api.baseball import League
from datetime import datetime

load_dotenv()

LEAGUE_ID = int(os.environ["ESPN_LEAGUE_ID"])
TEAM_ID   = int(os.environ["ESPN_TEAM_ID"])
ESPN_S2   = os.environ["ESPN_S2"]
SWID      = os.environ["ESPN_SWID"]
YEAR      = datetime.now().year

league = League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)
team = next(t for t in league.teams if t.team_id == TEAM_ID)

from config import IL_ELIGIBLE_STATUSES

print(f"{'Name':<25} {'Slot':<6} {'Status':<20} {'Injured':<8} {'IL in eligible':<15} {'Stashable?'}")
print("-" * 100)

for p in team.roster:
    status = getattr(p, "injuryStatus", "ACTIVE")
    has_il = "IL" in p.eligibleSlots
    stashable = has_il and status in IL_ELIGIBLE_STATUSES and p.lineupSlot != "IL"
    on_il = p.lineupSlot == "IL"

    flag = ""
    if on_il:
        flag = "ALREADY ON IL"
    elif stashable:
        flag = "YES"
    elif has_il and p.lineupSlot != "IL":
        flag = f"NO (status={status})"

    if has_il or p.injured or status != "ACTIVE" or on_il:
        print(f"{p.name:<25} {p.lineupSlot:<6} {status:<20} {str(p.injured):<8} {str(has_il):<15} {flag}")

print()
print("Full roster slots for reference:")
for p in team.roster:
    print(f"  {p.name:<25} slot: {p.lineupSlot:<6} eligibleSlots: {p.eligibleSlots}")
