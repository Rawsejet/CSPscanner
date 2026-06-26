"""Unit tests for the pure iron-condor math in ``ic_metrics``.

All expectations are hand-computed so the risk/edge formulas are pinned down
independently of any live chain.
"""
import math
import pytest

from ic_metrics import (
    realized_vol, expected_move, build_condor, condor_metrics,
    atm_implied_vol, implied_vol, edge_grade, resolve_wing,
)


def opt(strike, bid, ask, delta=None, iv=None, otype=None):
    o = {"strike": strike, "bid": bid, "ask": ask}
    greeks = {}
    if delta is not None:
        greeks["delta"] = delta
    if iv is not None:
        greeks["mid_iv"] = iv
    if greeks:
        o["greeks"] = greeks
    if otype:
        o["option_type"] = otype
    return o


# A symmetric chain around spot=100. Short strikes at the 0.16-delta (90P/110C),
# 10-wide wings -> longs at 80/120.
PUTS = [
    opt(80, 0.40, 0.50, delta=-0.06),
    opt(85, 0.60, 0.70, delta=-0.10),
    opt(90, 1.00, 1.10, delta=-0.16),
    opt(95, 1.50, 1.60, delta=-0.25),
]
CALLS = [
    opt(105, 1.50, 1.60, delta=0.25),
    opt(110, 0.90, 1.00, delta=0.16),
    opt(115, 0.60, 0.70, delta=0.10),
    opt(120, 0.30, 0.40, delta=0.06),
]


# --- build_condor ---

def test_build_condor_selects_shorts_by_delta_and_snaps_wings():
    c = build_condor(100, PUTS, CALLS, short_delta=0.16, wing_width=10)
    assert c["short_put_strike"] == 90
    assert c["short_call_strike"] == 110
    assert c["long_put_strike"] == 80
    assert c["long_call_strike"] == 120
    assert c["put_wing"] == 10 and c["call_wing"] == 10
    assert c["body"] == 20


def test_build_condor_handles_magnitude_only_deltas():
    """A feed reporting put deltas as positive magnitudes must still work."""
    puts = [opt(90, 1.0, 1.1, delta=0.16), opt(80, 0.4, 0.5, delta=0.06)]
    calls = [opt(110, 0.9, 1.0, delta=0.16), opt(120, 0.3, 0.4, delta=0.06)]
    c = build_condor(100, puts, calls, short_delta=0.16, wing_width=10)
    assert c["short_put_strike"] == 90 and c["short_call_strike"] == 110


def test_build_condor_none_when_no_otm_strikes():
    # spot below every put strike -> no OTM short put can be chosen
    assert build_condor(50, PUTS, CALLS, short_delta=0.16, wing_width=10) is None


def test_build_condor_none_when_wing_runs_off_the_board():
    # 0.06-delta short put is the lowest strike (80); no strike below it for the wing
    assert build_condor(100, PUTS, CALLS, short_delta=0.06, wing_width=10) is None


# --- condor_metrics (hand-computed) ---

def test_condor_metrics_match_hand_computation():
    c = build_condor(100, PUTS, CALLS, short_delta=0.16, wing_width=10)
    m = condor_metrics(c, spot=100, dte=7, atm_iv=0.30, realized_volatility=0.25)

    # credit_mid = (1.05 + 0.95) - (0.45 + 0.35) = 1.20
    assert m["credit_mid"] == pytest.approx(1.20)
    # credit_natural = (1.00 + 0.90) - (0.50 + 0.40) = 1.00
    assert m["credit_natural"] == pytest.approx(1.00)
    # max loss = (10 - 1.20) * 100 = 880
    assert m["max_loss"] == pytest.approx(880.0)
    assert m["return_on_risk"] == pytest.approx(120 / 880 * 100)
    assert m["breakeven_low"] == pytest.approx(88.80)
    assert m["breakeven_high"] == pytest.approx(111.20)
    # PoP = 1 - 0.16 - 0.16 = 68%
    assert m["pop"] == pytest.approx(68.0)
    # breakeven WR = 880 / (880 + 120) = 88%
    assert m["breakeven_win_rate"] == pytest.approx(88.0)
    # edge gap = 68 - 88 = -20pp (negative -> relies on VRP)
    assert m["edge_gap"] == pytest.approx(-20.0)
    # VRP: IV 30 vs RV 25 -> ratio 1.2, +5 vol pts
    assert m["vrp_ratio"] == pytest.approx(0.30 / 0.25)
    assert m["vrp_points"] == pytest.approx(5.0)


def test_condor_metrics_expected_move_multiples():
    c = build_condor(100, PUTS, CALLS, short_delta=0.16, wing_width=10)
    m = condor_metrics(c, spot=100, dte=7, atm_iv=0.30)
    em = expected_move(100, 0.30, 7)
    assert m["em_mult_put"] == pytest.approx((100 - 90) / em)
    assert m["em_mult_call"] == pytest.approx((110 - 100) / em)


def test_condor_metrics_no_vol_inputs_degrade_to_none():
    c = build_condor(100, PUTS, CALLS, short_delta=0.16, wing_width=10)
    m = condor_metrics(c, spot=100, dte=7)
    assert m["expected_move"] is None
    assert m["vrp_ratio"] is None and m["vrp_points"] is None
    # edge gap still computable without vol
    assert m["edge_gap"] == pytest.approx(-20.0)


# --- expected_move ---

def test_expected_move_formula_and_guards():
    assert expected_move(100, 0.30, 7) == pytest.approx(100 * 0.30 * math.sqrt(7 / 365))
    assert expected_move(100, 0.30, 0) is None
    assert expected_move(0, 0.30, 7) is None
    assert expected_move(100, None, 7) is None


# --- realized_vol ---

def test_realized_vol_insufficient_data_is_none():
    assert realized_vol([100, 101, 102]) is None


def test_realized_vol_constant_prices_is_zero():
    assert realized_vol([100.0] * 30, window=20) == pytest.approx(0.0)


def test_realized_vol_positive_when_prices_move():
    closes = [100.0]
    for i in range(40):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
    rv = realized_vol(closes, window=20)
    assert rv is not None and rv > 0


# --- implied vol helpers ---

def test_atm_iv_picks_half_delta_call():
    calls = [opt(100, 2.0, 2.1, delta=0.50, iv=0.42),
             opt(110, 0.9, 1.0, delta=0.16, iv=0.55)]
    assert atm_implied_vol(calls) == pytest.approx(0.42)


def test_implied_vol_falls_back_across_keys():
    assert implied_vol({"greeks": {"smv_vol": 0.33}}) == pytest.approx(0.33)
    assert implied_vol({"greeks": {}}) is None
    assert implied_vol({}) is None


# --- edge_grade ---

def test_edge_grade_red_when_unfillable_or_negative_vrp():
    # natural credit <= 0 -> Red even with rich VRP
    assert edge_grade({"vrp_ratio": 1.4, "credit_natural": -0.5}) == "Red"
    # IV below realized (VRP < 1) -> Red
    assert edge_grade({"vrp_ratio": 0.8, "credit_natural": 1.0}) == "Red"


def test_edge_grade_green_when_rich_vrp_and_fillable():
    assert edge_grade({"vrp_ratio": 1.25, "credit_natural": 0.5}) == "Green"


def test_edge_grade_amber_in_between():
    assert edge_grade({"vrp_ratio": 1.05, "credit_natural": 0.5}) == "Amber"
    # unknown VRP but fillable -> Amber, not Red
    assert edge_grade({"vrp_ratio": None, "credit_natural": 0.5}) == "Amber"


# --- resolve_wing ---

def test_resolve_wing_pct_scales_with_price():
    # 2% of spot: tiny on a cheap name, the same ~$20 on a $1,000 name
    assert resolve_wing(1000, "pct", 2.0) == pytest.approx(20.0)
    assert resolve_wing(80, "pct", 2.0) == pytest.approx(1.6)


def test_resolve_wing_fixed_ignores_price():
    assert resolve_wing(80, "fixed", 20.0) == pytest.approx(20.0)
    assert resolve_wing(1200, "fixed", 20.0) == pytest.approx(20.0)
