"""
ESPN Fantasy Baseball API interactions.
"""

import requests
from espn_api.baseball import League

from config import (
    LEAGUE_ID, TEAM_ID, ESPN_S2, SWID, YEAR, SLOT_ID,
)


def get_league() -> League:
    return League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)


def fetch_undroppable_names(team_id: int = TEAM_ID) -> set[str]:
    """Fetch the set of undroppable player names on our roster from ESPN's raw API.

    The espn_api library doesn't expose the 'droppable' flag, so we hit the
    mRoster view directly.
    """
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb"
        f"/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}?view=mRoster"
    )
    cookies = {"espn_s2": ESPN_S2, "SWID": SWID}
    resp = requests.get(url, cookies=cookies)
    resp.raise_for_status()
    data = resp.json()

    team = next(t for t in data["teams"] if t["id"] == team_id)
    return {
        entry["playerPoolEntry"]["player"]["fullName"]
        for entry in team["roster"]["entries"]
        if not entry["playerPoolEntry"]["player"].get("droppable", True)
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


def make_move(player, from_slot: str, to_slot: str) -> dict:
    """Build a single ESPN transaction item for a slot change."""
    return {
        "playerId":         player.playerId,
        "fromLineupSlotId": SLOT_ID.get(from_slot, 16),
        "toLineupSlotId":   SLOT_ID.get(to_slot,   16),
    }


def submit(items: list[dict], scoring_period: int, label: str = ""):
    """Submit a batch of moves and log results. Skips if empty."""
    if not items:
        return
    if label:
        print(f"  [{label}]")
    espn_set_lineup(items, scoring_period)


def espn_add_drop(add_player, drop_player, scoring_period: int):
    """POST a free-agent add + drop transaction directly to the ESPN write API.

    Payload format confirmed from DevTools. If the player is still within the
    waiver window ESPN will return 400 — change both the top-level `type` and
    the add-item `type` to "WAIVER" in that case.
    """
    url = (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/flb"
        f"/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}/transactions/"
    )
    cookies = {"espn_s2": ESPN_S2, "SWID": SWID}
    headers = {"Content-Type": "application/json"}

    payload = {
        "isLeagueManager": False,
        "teamId":          TEAM_ID,
        "type":            "FREEAGENT",
        "memberId":        SWID,
        "scoringPeriodId": scoring_period,
        "executionType":   "EXECUTE",
        "items": [
            {"playerId": add_player.playerId,  "type": "ADD",  "toTeamId":   TEAM_ID},
            {"playerId": drop_player.playerId, "type": "DROP", "fromTeamId": TEAM_ID},
        ],
    }

    resp = requests.post(url, json=payload, cookies=cookies, headers=headers)
    print(f"   ESPN response {resp.status_code} — add {add_player.name} / drop {drop_player.name}")
    print(resp.text)
    resp.raise_for_status()
    return resp.json()


def espn_drop_player(player, from_slot: str, scoring_period: int):
    """Drop a player from the roster without adding anyone.

    TODO: Confirm standalone drop payload via ESPN DevTools. The item structure
    matches the confirmed drop-within-ROSTER format, but a solo drop may use
    a different transaction type (e.g. "FREEAGENT").
    """
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
        "scoringPeriodId": scoring_period,
        "executionType":   "EXECUTE",
        "items": [{
            "playerId":         player.playerId,
            "type":             "DROP",
            "fromLineupSlotId": SLOT_ID.get(from_slot, 16),
            "fromTeamId":       TEAM_ID,
            "toLineupSlotId":   -1,
            "toTeamId":         0,
        }],
    }

    resp = requests.post(url, json=payload, cookies=cookies, headers=headers)
    print(f"   ESPN response {resp.status_code} — drop {player.name}")
    resp.raise_for_status()
    return resp.json()


def espn_activate_from_il(il_player, drop_player, drop_from_slot: str,
                          activate_to_slot: str, scoring_period: int):
    """Atomically drop a player and move another off IL in one transaction.

    Payload format confirmed via ESPN DevTools. Both items are sent in a single
    ROSTER transaction so ESPN processes the drop and IL activation together.
    """
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
        "scoringPeriodId": scoring_period,
        "executionType":   "EXECUTE",
        "items": [
            {
                "playerId":         drop_player.playerId,
                "type":             "DROP",
                "fromLineupSlotId": SLOT_ID.get(drop_from_slot, 16),
                "fromTeamId":       TEAM_ID,
                "toLineupSlotId":   -1,
                "toTeamId":         0,
            },
            {
                "playerId":         il_player.playerId,
                "type":             "LINEUP",
                "fromLineupSlotId": SLOT_ID["IL"],
                "toLineupSlotId":   SLOT_ID.get(activate_to_slot, 16),
                "fromTeamId":       0,
                "toTeamId":         0,
            },
        ],
    }

    resp = requests.post(url, json=payload, cookies=cookies, headers=headers)
    print(f"   ESPN response {resp.status_code} — drop {drop_player.name}, "
          f"activate {il_player.name}: IL → {activate_to_slot}")
    resp.raise_for_status()
    return resp.json()
