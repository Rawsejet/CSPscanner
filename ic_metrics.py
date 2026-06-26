"""Pure (API-free) math for weekly iron condors.

These helpers turn a Tradier option chain — the same raw dict shape the CSP
scanner already consumes (``strike``/``bid``/``ask`` plus a nested ``greeks``
block) — into the honest risk/edge metrics the iron-condor scanner ranks on.
Nothing here touches the network, so it is all unit-testable against fixed
inputs.

Conventions
-----------
* Option dicts look like
  ``{"strike": float, "bid": float, "ask": float,
     "greeks": {"delta": float, "mid_iv": float}}``.
* Put deltas are treated as negative and call deltas as positive regardless of
  how the feed reports the sign (some report magnitudes) — mirroring the CSP
  scanner's ``-abs(delta)`` normalisation.
* Credit/debit values are per share; ``* 100`` gives dollars per contract.
* **Max loss is governed by the wider wing, never the body** (the gap between
  the short strikes): at expiry the underlying can only sit in one losing wing,
  so widening the body makes the trade *safer*, not riskier.
"""
import math
from typing import Any, Dict, List, Optional

OptionDict = Dict[str, Any]


# --- tolerant field accessors (raw Tradier option dicts may carry None/strings) ---

def _num(opt: OptionDict, key: str) -> Optional[float]:
    try:
        v = opt.get(key)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _strike(opt: OptionDict) -> Optional[float]:
    return _num(opt, "strike")


def _bid(opt: OptionDict) -> float:
    v = _num(opt, "bid")
    return v if v is not None else 0.0


def _ask(opt: OptionDict) -> float:
    v = _num(opt, "ask")
    return v if v is not None else 0.0


def _mid(opt: OptionDict) -> float:
    return (_bid(opt) + _ask(opt)) / 2


def _raw_delta(opt: OptionDict) -> Optional[float]:
    greeks = opt.get("greeks") or {}
    try:
        d = greeks.get("delta")
        return float(d) if d is not None else None
    except (TypeError, ValueError):
        return None


def _signed_delta(opt: OptionDict, kind: str) -> Optional[float]:
    """Delta normalised to the option's natural sign: puts negative, calls positive.

    Robust to feeds that report the magnitude only (cf. the CSP scanner's
    ``-abs(raw_delta)``).
    """
    d = _raw_delta(opt)
    if d is None:
        return None
    return -abs(d) if kind == "put" else abs(d)


def implied_vol(opt: OptionDict) -> Optional[float]:
    """Per-contract implied vol from the greeks block (ORATS ``mid_iv``/``smv_vol``).

    Tradier returns these on ``markets/options/chains`` when ``greeks=true`` —
    the CSP scanner already requests greeks but only reads delta.
    """
    greeks = opt.get("greeks") or {}
    for key in ("mid_iv", "smv_vol", "ask_iv", "bid_iv"):
        try:
            v = greeks.get(key)
            if v not in (None, 0, "0"):
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


# --- volatility helpers ---

def realized_vol(closes: List[float], window: int = 20,
                 trading_days: int = 252) -> Optional[float]:
    """Annualised realized volatility from the last ``window`` daily closes.

    Returns None when there aren't enough usable returns. Mirrors the
    realized-vol computation inside ``utils.calculate_iv_rank`` so the iron
    condor's VRP (IV vs RV) uses the same definition.
    """
    if not closes or len(closes) < window + 1:
        return None
    returns = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        if prev and cur and prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
    if len(returns) < window:
        return None
    win = returns[-window:]
    mean = sum(win) / window
    var = sum((r - mean) ** 2 for r in win) / (window - 1)
    return math.sqrt(var) * math.sqrt(trading_days)


def expected_move(spot: Optional[float], atm_iv: Optional[float],
                  dte: Optional[int], days_per_year: int = 365) -> Optional[float]:
    """One-sigma expected move over ``dte`` days: ``spot * atm_iv * sqrt(dte/year)``."""
    if not spot or not atm_iv or dte is None or dte <= 0:
        return None
    return spot * atm_iv * math.sqrt(dte / days_per_year)


# --- contract selection ---

def _nearest_by_delta(options: List[OptionDict], kind: str,
                      target_mag: float) -> Optional[OptionDict]:
    """Option whose (sign-normalised) delta is closest to the target magnitude."""
    target = -target_mag if kind == "put" else target_mag
    best, best_diff = None, None
    for o in options:
        sd = _signed_delta(o, kind)
        if sd is None:
            continue
        diff = abs(sd - target)
        if best_diff is None or diff < best_diff:
            best, best_diff = o, diff
    return best


def _nearest_strike_option(options: List[OptionDict], target: float,
                           below: Optional[float] = None,
                           above: Optional[float] = None) -> Optional[OptionDict]:
    """Option whose strike is nearest ``target``, optionally constrained to be
    strictly ``below`` and/or ``above`` the given strikes (so a long wing always
    lands beyond its short)."""
    best, best_diff = None, None
    for o in options:
        k = _strike(o)
        if k is None:
            continue
        if below is not None and not k < below:
            continue
        if above is not None and not k > above:
            continue
        diff = abs(k - target)
        if best_diff is None or diff < best_diff:
            best, best_diff = o, diff
    return best


def atm_implied_vol(calls: List[OptionDict],
                    spot: Optional[float] = None) -> Optional[float]:
    """ATM implied vol: the call nearest 0.5 delta, falling back to nearest spot."""
    atm = _nearest_by_delta(calls, "call", 0.5)
    if atm is not None:
        iv = implied_vol(atm)
        if iv:
            return iv
    if spot is not None:
        nearest = _nearest_strike_option(calls, spot)
        if nearest is not None:
            return implied_vol(nearest)
    return None


def build_condor(spot: float, puts: List[OptionDict], calls: List[OptionDict],
                 short_delta: float = 0.10,
                 wing_width: float = 20.0) -> Optional[Dict[str, Any]]:
    """Construct a symmetric-by-delta iron condor from a chain.

    Picks the OTM short put and short call whose deltas are closest to
    ``short_delta`` in magnitude, then the long strikes nearest ``wing_width``
    beyond each short (snapped to listed strikes). Returns the four legs and
    their strikes, or ``None`` if a valid condor can't be built (no OTM strikes,
    missing greeks, or no listed strike beyond a short for the wing).
    """
    if not spot or spot <= 0:
        return None
    otm_puts = [p for p in puts if (_strike(p) or 0) < spot]
    otm_calls = [c for c in calls if (_strike(c) or 0) > spot]
    short_put = _nearest_by_delta(otm_puts, "put", short_delta)
    short_call = _nearest_by_delta(otm_calls, "call", short_delta)
    if short_put is None or short_call is None:
        return None

    spk, sck = _strike(short_put), _strike(short_call)
    long_put = _nearest_strike_option(puts, spk - wing_width, below=spk)
    long_call = _nearest_strike_option(calls, sck + wing_width, above=sck)
    if long_put is None or long_call is None:
        return None

    lpk, lck = _strike(long_put), _strike(long_call)
    if not (lpk < spk < sck < lck):
        return None

    return {
        "short_put": short_put, "long_put": long_put,
        "short_call": short_call, "long_call": long_call,
        "short_put_strike": spk, "long_put_strike": lpk,
        "short_call_strike": sck, "long_call_strike": lck,
        "put_wing": spk - lpk, "call_wing": lck - sck, "body": sck - spk,
    }


def condor_metrics(condor: Dict[str, Any], spot: float, dte: int,
                   atm_iv: Optional[float] = None,
                   realized_volatility: Optional[float] = None) -> Dict[str, Any]:
    """Honest risk/edge metrics for a built condor.

    Headline credit/max-loss use the **mid** price; the **natural** credit
    (sell the bid, buy the ask) is reported alongside so the 4-leg slippage is
    visible — on far-OTM weeklies it can exceed the entire mid credit.

    Key metrics:
      * ``edge_gap`` = delta-implied PoP − breakeven win rate (percentage points).
        Negative means the structure relies entirely on the volatility risk
        premium to be profitable.
      * ``vrp_ratio``/``vrp_points`` = ATM implied vol vs realized vol. > 1
        (positive points) means you're paid to sell; < 1 means IV is cheap
        relative to what the stock is realizing (no edge).
    """
    sp, lp = condor["short_put"], condor["long_put"]
    sc, lc = condor["short_call"], condor["long_call"]
    spk, sck = condor["short_put_strike"], condor["short_call_strike"]
    wing = max(condor["put_wing"], condor["call_wing"])

    credit_mid = (_mid(sp) + _mid(sc)) - (_mid(lp) + _mid(lc))
    credit_natural = (_bid(sp) + _bid(sc)) - (_ask(lp) + _ask(lc))

    credit_dollars = credit_mid * 100
    max_loss = (wing - credit_mid) * 100
    return_on_risk = (credit_dollars / max_loss * 100) if max_loss > 0 else None

    breakeven_low = spk - credit_mid
    breakeven_high = sck + credit_mid

    dput = _signed_delta(sp, "put")
    dcall = _signed_delta(sc, "call")
    pop = (1 - abs(dput) - abs(dcall)) * 100 if (dput is not None and dcall is not None) else None
    denom = max_loss + credit_dollars
    breakeven_win_rate = (max_loss / denom * 100) if denom > 0 else None
    edge_gap = (pop - breakeven_win_rate) if (pop is not None and breakeven_win_rate is not None) else None

    em = expected_move(spot, atm_iv, dte) if atm_iv else None
    em_mult_put = ((spot - spk) / em) if em else None
    em_mult_call = ((sck - spot) / em) if em else None

    vrp_ratio = (atm_iv / realized_volatility) if (atm_iv and realized_volatility) else None
    vrp_points = ((atm_iv - realized_volatility) * 100) if (atm_iv is not None and realized_volatility is not None) else None

    return {
        "credit_mid": credit_mid,
        "credit_natural": credit_natural,
        "wing": wing,
        "body": condor["body"],
        "max_loss": max_loss,
        "return_on_risk": return_on_risk,
        "breakeven_low": breakeven_low,
        "breakeven_high": breakeven_high,
        "short_put_delta": dput,
        "short_call_delta": dcall,
        "pop": pop,
        "breakeven_win_rate": breakeven_win_rate,
        "edge_gap": edge_gap,
        "expected_move": em,
        "em_mult_put": em_mult_put,
        "em_mult_call": em_mult_call,
        "vrp_ratio": vrp_ratio,
        "vrp_points": vrp_points,
    }


def edge_grade(metrics: Dict[str, Any], vrp_min: float = 1.1) -> str:
    """Stoplight grade for a condor's *real* edge.

    The far-OTM edge gap is structurally negative (it's the premium you sell), so
    grading keys off the two things that actually decide profitability: a positive
    variance risk premium and a fillable credit.

    * **Red**  — you'd pay to enter (natural credit <= 0) *or* IV is below realized
      vol (VRP < 1.0): no edge.
    * **Green** — VRP rich (>= ``vrp_min``) *and* fillable for a credit.
    * **Amber** — everything in between (marginal/unknown VRP, still fillable).
    """
    vrp = metrics.get("vrp_ratio")
    nat = metrics.get("credit_natural")
    if nat is not None and nat <= 0:
        return "Red"
    if vrp is not None and vrp < 1.0:
        return "Red"
    if vrp is not None and vrp >= vrp_min:
        return "Green"
    return "Amber"


def resolve_wing(spot: float, wing_mode: str, wing_value: float) -> float:
    """Dollar wing width from either a percent-of-spot or a fixed dollar amount.

    A flat dollar wing doesn't travel across the universe — $20 is ~2% of a
    $1,200 name but ~25% of an $80 one — so ``"pct"`` (wing = ``wing_value``% of
    spot) is the sensible default; ``"fixed"`` keeps a set dollar wing for the
    deliberately-narrow high-priced case. The result is later snapped to listed
    strikes by ``build_condor``.
    """
    if wing_mode == "pct":
        return spot * (wing_value / 100.0)
    return float(wing_value)
