# 🤖 ARIA — Automated Roster Intelligence Agent

An AI-powered ESPN fantasy baseball manager using Google Gemini. Built for **Head-to-Head Points** leagues.

---

## 🚀 Quick Start

### 1. Get Your ESPN Credentials

You need two ESPN auth cookies (`espn_s2` and `SWID`):

1. Log into ESPN Fantasy in your browser
2. Open DevTools → Application → Cookies → `https://www.espn.com`
3. Copy the values for `espn_s2` and `SWID`

> **Note:** Your `SWID` value includes curly braces — keep them. It should look like `{E8C520B4-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`.

### 2. Get Your League & Team IDs

- **League ID**: Found in your ESPN league URL — `fantasy.espn.com/baseball/league?leagueId=XXXXXXX`
- **Team ID**: Click on your team in ESPN → the URL shows `teamId=X`

### 3. Get a Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Create an API key — no credit card required for the free tier

### 4. Create Your `.env` File

Create a `.env` file in the project root and fill in your values:

```
ESPN_LEAGUE_ID=
ESPN_TEAM_ID=
ESPN_S2=
ESPN_SWID=
GEMINI_API_KEY=
```

> **Important:** Add `.env` to your `.gitignore` so credentials are never committed.
> ```bash
> echo ".env" >> .gitignore
> ```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run ARIA

```bash
python src/ai_manager.py          # lineup + waivers (default)
python src/ai_manager.py lineup   # lineup only
python src/ai_manager.py waivers  # waivers only
```

---

## ☁️ GitHub Actions (Automated Runs)

To have ARIA run automatically every day:

### 1. Set Up a GitHub Repository

```bash
git init
git add .
git commit -m "ARIA — initial setup"
git remote add origin https://github.com/YOUR_USERNAME/fantasy-baseball-ai.git
git push -u origin main
```

### 2. Add GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these 5 secrets:

| Secret Name | Value |
|---|---|
| `ESPN_LEAGUE_ID` | Your league ID number |
| `ESPN_TEAM_ID` | The AI team's ID number |
| `ESPN_S2` | Your `espn_s2` cookie value |
| `ESPN_SWID` | Your `SWID` cookie value (include curly braces) |
| `GEMINI_API_KEY` | Your Gemini API key |

### 3. Enable GitHub Actions

Go to your repo → **Actions** tab → click **"I understand my workflows, go ahead and enable them"**

ARIA will automatically:
- Set the optimal lineup every morning at 9:30 AM ET
- Check the waiver wire every morning at 6:00 AM ET

---

## 🎯 Draft Day Assistant

Run this on your laptop **during** your ESPN draft for real-time pick recommendations:

```bash
python src/draft_assistant.py
```

ARIA will fetch the live draft board from ESPN each time it's your turn and recommend the best available player based on your roster needs and H2H Points strategy.

---

## 🧠 Customizing ARIA's Strategy

Edit the `GM_PERSONA` string near the top of `src/ai_manager.py` to change how ARIA manages the team. Examples:

- **Punting steals**: `"Ignore SB, maximize HR/RBI/R/K"`
- **Pitching heavy**: `"Draft 6+ SP, stream aggressively, accept weak offense"`
- **Balanced**: `"Target balanced scorers at every position"`

---

## 🧪 Testing Without Calling the AI

Set `USE_LLM = False` in `src/ai_manager.py` to skip Gemini entirely and use the hardcoded mock responses instead. This lets you test ESPN API calls without burning quota.

Edit `MOCK_LINEUP_RESPONSE` and `MOCK_WAIVER_RESPONSE` with real player names from your roster before running:

```python
USE_LLM = False

MOCK_LINEUP_RESPONSE = """
ACTIVE: Player A, Player B, ...
BENCH: Player C, Player D, ...
REASONING: Testing ESPN API calls.
"""
```

Set `USE_LLM = True` when you're ready to use the real Gemini API.

---

## ⚙️ How Lineup Setting Works

ARIA applies lineup changes in four ordered phases to avoid ESPN slot conflicts:

1. **Bench outgoing players** — frees active slots before anything tries to fill them
2. **Reposition active players** — handles slot changes (e.g. moving a player from 2B to 3B to make room). Uses the bench as a waypoint for triangular swaps
3. **Promote bench players** — moves players from bench into the newly freed active slots
4. **Pitcher pass** — independently manages all P slots. Non-starters are benched; starting pitchers sitting on the bench are promoted. Uses ESPN's injury status as a proxy for whether a pitcher is starting today

Lineup moves are submitted one at a time to match the ESPN API's expected payload format, with outgoing moves (to bench) always sent before incoming moves (to active slots).

---

## ⚠️ Notes

- **ESPN cookies expire periodically.** If ARIA stops working, refresh your `espn_s2` and `SWID` values in your `.env` or GitHub secrets
- **Gemini free tier limits.** The free tier allows a limited number of requests per day. If ARIA hits the daily quota, it will fail with a clear error message and will not retry (retrying won't help until the quota resets at midnight Pacific time). Per-minute rate limits are retried automatically using the delay suggested by the API
- **The `espn-api` library is unofficial.** ESPN may change their internal API without notice, which could break ARIA
- **ESPN's write API is also unofficial.** Lineup moves are submitted directly via ESPN's internal transactions endpoint, reverse-engineered from browser network traffic. This works reliably but is not a supported integration
- **The pitcher starting detection is approximate.** ARIA uses ESPN's injury status field as a proxy for whether a pitcher is starting today. For more accurate detection, a separate schedule API would be needed
- **IL slot ID is unconfirmed.** The IL slot is assumed to be ID `17` based on standard ESPN numbering. If you need to move a player to/from IL, confirm the correct ID via DevTools first and update `SLOT_ID` in `ai_manager.py`
- **ARIA makes real roster moves.** Double-check early in the season that it's behaving as expected before fully trusting it