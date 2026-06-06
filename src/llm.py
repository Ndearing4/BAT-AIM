"""
LLM interaction layer — Gemini API calls and response parsing.
"""

import time
from google import genai

from config import (
    USE_LLM, GEMINI_KEY,
    MOCK_LINEUP_RESPONSE, MOCK_PITCHER_RESPONSE,
    MOCK_WAIVER_RESPONSE, MOCK_IL_CLEANUP_RESPONSE,
)


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
        if mode == "lineup":
            mock = MOCK_LINEUP_RESPONSE
        elif mode == "pitchers":
            mock = MOCK_PITCHER_RESPONSE
        elif mode == "il_cleanup":
            mock = MOCK_IL_CLEANUP_RESPONSE
        else:
            mock = MOCK_WAIVER_RESPONSE
        print(f"🧪 USE_LLM=False — using mock {mode} response")
        return mock.strip()

    client = genai.Client(api_key=GEMINI_KEY)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
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
    result = {"active": [], "bench": [], "rotation": [], "pbench": [], "reasoning": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("ACTIVE:"):
            result["active"] = [n.strip() for n in line[7:].split(",") if n.strip()]
        elif line.upper().startswith("BENCH:"):
            result["bench"] = [n.strip() for n in line[6:].split(",") if n.strip()]
        elif line.upper().startswith("ROTATION:"):
            result["rotation"] = [n.strip() for n in line[9:].split(",") if n.strip()]
        elif line.upper().startswith("PBENCH:"):
            result["pbench"] = [n.strip() for n in line[7:].split(",") if n.strip()]
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line[10:].strip()
    return result


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
