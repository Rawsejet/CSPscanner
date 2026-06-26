"""Open iron-condor position store + live management-trigger evaluation.

Positions persist in a flat JSON file (``ic_positions.json``, gitignored) so the
dashboard can track condors across sessions. Each refresh re-pulls the current
chain for a position's expiry and lights up the management triggers the user
actually uses: take profit at ~50% of credit, defend the tested side when a
short delta crosses ~0.30, and close into the last day(s) to avoid gamma/pin.

The pure ``position_status`` function does the P&L + trigger math from
already-fetched option dicts, so it is unit-testable without the network;
``evaluate_position`` wraps it with the Tradier fetch.
"""
import json
import os
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from tradier_client import TradierClient, TradierAPIError
from ic_metrics import _mid, _bid, _ask, _strike, _signed_delta

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "ic_positions.json")

LEG_FIELDS = (
    "short_put_strike", "long_put_strike", "short_call_strike", "long_call_strike",
)


# --- persistence ---

def load_positions(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or POSITIONS_FILE
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_positions(positions: List[Dict[str, Any]], path: Optional[str] = None) -> None:
    path = path or POSITIONS_FILE
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)


def add_position(position: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """Append a position (assigning an ``id`` and ``entry_date`` if missing)."""
    position = dict(position)
    position.setdefault("id", secrets.token_hex(4))
    position.setdefault("entry_date", datetime.now().date().isoformat())
    position.setdefault("quantity", 1)
    positions = load_positions(path)
    positions.append(position)
    save_positions(positions, path)
    return position


def remove_position(position_id: str, path: Optional[str] = None) -> None:
    positions = [p for p in load_positions(path) if p.get("id") != position_id]
    save_positions(positions, path)


# --- live evaluation ---

def position_status(position: Dict[str, Any], short_put, long_put, short_call,
                    long_call, spot: Optional[float], dte: Optional[int],
                    profit_target: float = 0.50, tested_delta: float = 0.30,
                    time_dte: int = 1, stop_multiple: float = 2.0) -> Dict[str, Any]:
    """P&L and active management triggers from current leg quotes (pure)."""
    entry = position.get("entry_credit")
    qty = position.get("quantity", 1) or 1

    # Cost to buy the condor back now: shorts at ask, longs at bid (natural);
    # mid for the headline. Profit when this is below the entry credit.
    value_mid = (_mid(short_put) + _mid(short_call)) - (_mid(long_put) + _mid(long_call))
    value_nat = (_ask(short_put) + _ask(short_call)) - (_bid(long_put) + _bid(long_call))

    pl_per_share = (entry - value_mid) if entry is not None else None
    pl_pct = (pl_per_share / entry * 100) if (pl_per_share is not None and entry) else None
    pl_dollars = (pl_per_share * 100 * qty) if pl_per_share is not None else None

    dput = _signed_delta(short_put, "put")
    dcall = _signed_delta(short_call, "call")
    tested = max(abs(dput) if dput is not None else 0.0,
                 abs(dcall) if dcall is not None else 0.0)

    triggers: List[str] = []
    if pl_pct is not None and pl_pct >= profit_target * 100:
        triggers.append(f"PROFIT — captured {pl_pct:.0f}% of credit (target {profit_target*100:.0f}%); close")
    if tested >= tested_delta:
        side = "put" if (abs(dput or 0) >= abs(dcall or 0)) else "call"
        triggers.append(f"TESTED — {side} short Δ {tested:.2f}; roll the untested side in or close")
    if dte is not None and dte <= time_dte and pl_pct is not None and pl_pct > 0:
        triggers.append(f"TIME — {dte}DTE with profit; close to avoid gamma/pin")
    if pl_pct is not None and pl_pct <= -stop_multiple * 100:
        triggers.append(f"STOP — down {abs(pl_pct):.0f}% of credit (≥{stop_multiple:.0f}×); consider closing")

    return {
        "id": position.get("id"),
        "ticker": position.get("ticker"),
        "expiry": position.get("expiry"),
        "dte": dte,
        "quantity": qty,
        "entry_credit": entry,
        "spot": spot,
        "value_mid": value_mid,
        "value_natural": value_nat,
        "pl_per_share": pl_per_share,
        "pl_pct": pl_pct,
        "pl_dollars": pl_dollars,
        "short_put_delta": dput,
        "short_call_delta": dcall,
        "tested_delta": tested,
        "triggers": triggers,
        "error": None,
    }


def _find_leg(chain: List[Dict[str, Any]], option_type: str, strike: float):
    for o in chain:
        if o.get("option_type") == option_type and _strike(o) == strike:
            return o
    return None


def evaluate_position(client: TradierClient, position: Dict[str, Any],
                      **trigger_kwargs) -> Dict[str, Any]:
    """Fetch the live chain for a position and return its status + triggers.

    Any data/transport problem degrades to a status dict carrying ``error``
    rather than raising, so one stale position can't break the whole table.
    """
    base = {"id": position.get("id"), "ticker": position.get("ticker"),
            "expiry": position.get("expiry"), "triggers": []}
    try:
        expiry = position["expiry"]
        today = datetime.now().date()
        expiry_d = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (expiry_d - today).days

        spot = client.get_spot_price(position["ticker"]) or None
        chain = client.get_option_chain(position["ticker"], expiry)
        sp = _find_leg(chain, "put", position["short_put_strike"])
        lp = _find_leg(chain, "put", position["long_put_strike"])
        sc = _find_leg(chain, "call", position["short_call_strike"])
        lc = _find_leg(chain, "call", position["long_call_strike"])
        if not all((sp, lp, sc, lc)):
            return {**base, "dte": dte, "spot": spot,
                    "error": "one or more legs not found in the current chain"}

        return position_status(position, sp, lp, sc, lc, spot, dte, **trigger_kwargs)
    except TradierAPIError as e:
        return {**base, "error": str(e)}
    except (KeyError, ValueError, TypeError) as e:
        return {**base, "error": f"bad position record: {e}"}
