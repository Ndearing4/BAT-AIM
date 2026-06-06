"""
Quick test: stash Dylan Cease to IL slot.
Finds him on the roster, prints his current state, and submits the move.
Run: python src/test_il_stash.py
"""
import os
import requests
from dotenv import load_dotenv
from espn_api.baseball import League
from datetime import datetime

load_dotenv()

LEAGUE_ID = int(os.environ["ESPN_LEAGUE_ID"])
TEAM_ID   = int(os.environ["ESPN_TEAM_ID"])
ESPN_S2   = os.environ["ESPN_S2"]
SWID      = os.environ["ESPN_SWID"]
YEAR      = datetime.now().year

SLOT_ID = {
    "C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5,
    "DH": 10, "UTIL": 12, "SP": 13, "RP": 13, "P": 13,
    "BE": 16, "IL": 17,
}

league = League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)
team = next(t for t in league.teams if t.team_id == TEAM_ID)

# Find Dylan Cease
cease = next((p for p in team.roster if p.name == "Dylan Cease"), None)
if not cease:
    print("Dylan Cease not found on roster. Players on roster:")
    for p in team.roster:
        print(f"  {p.name:<25} slot: {p.lineupSlot:<6} eligible: {p.eligibleSlots}")
    raise SystemExit(1)

print(f"Found: {cease.name}")
print(f"  Player ID:     {cease.playerId}")
print(f"  Current slot:  {cease.lineupSlot}")
print(f"  Eligible slots: {cease.eligibleSlots}")
print(f"  Injured:       {cease.injured}")
print(f"  Injury status: {getattr(cease, 'injuryStatus', 'ACTIVE')}")
print(f"  IL eligible:   {'IL' in cease.eligibleSlots}")
print()

if "IL" not in cease.eligibleSlots:
    print("Dylan Cease is NOT IL-eligible — cannot stash. Aborting.")
    raise SystemExit(1)

if cease.lineupSlot == "IL":
    print("Dylan Cease is already on IL. Nothing to do.")
    raise SystemExit(0)

from_slot_id = SLOT_ID.get(cease.lineupSlot, 16)
to_slot_id   = SLOT_ID["IL"]

print(f"Moving: {cease.lineupSlot} (slot {from_slot_id}) → IL (slot {to_slot_id})")
print()

url = (
    f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/flb"
    f"/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}/transactions/"
)
cookies = {"espn_s2": ESPN_S2, "SWID": SWID}
headers = {"Content-Type": "application/json"}

payload = {
    "isLeagueManager": False,
    "teamId":          TEAM_ID,
    "type":            "ROSTER",
    "memberId":        SWID,
    "scoringPeriodId": league.current_week,
    "executionType":   "EXECUTE",
    "items": [{
        "playerId":         cease.playerId,
        "type":             "LINEUP",
        "fromLineupSlotId": from_slot_id,
        "toLineupSlotId":   to_slot_id,
        "fromTeamId":       0,
        "toTeamId":         0,
    }],
}

print("Payload:")
import json
print(json.dumps(payload, indent=2))
print()

resp = requests.post(url, json=payload, cookies=cookies, headers=headers)
print(f"Response: {resp.status_code}")
print(resp.text[:500] if resp.text else "(empty)")

if resp.ok:
    print("\n✅ Dylan Cease stashed to IL successfully.")
else:
    print(f"\n❌ Failed with status {resp.status_code}")
