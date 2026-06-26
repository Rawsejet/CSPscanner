"""Tests for the iron-condor position store and live trigger logic."""
import pytest

from positions import (
    load_positions, save_positions, add_position, remove_position,
    position_status,
)


def opt(strike, bid, ask, delta=None):
    o = {"strike": strike, "bid": bid, "ask": ask}
    if delta is not None:
        o["greeks"] = {"delta": delta}
    return o


# --- JSON store (uses a temp path, never the real file) ---

def test_add_load_remove_roundtrip(tmp_path):
    path = str(tmp_path / "pos.json")
    assert load_positions(path) == []
    saved = add_position({"ticker": "AAPL", "expiry": "2026-07-02",
                          "short_put_strike": 260, "long_put_strike": 255,
                          "short_call_strike": 290, "long_call_strike": 295,
                          "entry_credit": 1.20, "quantity": 2}, path=path)
    assert saved["id"]  # id assigned
    assert saved["entry_date"]  # entry_date defaulted
    loaded = load_positions(path)
    assert len(loaded) == 1 and loaded[0]["ticker"] == "AAPL"
    remove_position(saved["id"], path=path)
    assert load_positions(path) == []


def test_load_missing_or_corrupt_file_is_empty(tmp_path):
    assert load_positions(str(tmp_path / "nope.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_positions(str(bad)) == []


# --- pure trigger logic ---

POS = {"id": "x", "ticker": "T", "expiry": "2026-07-02", "entry_credit": 1.00,
       "quantity": 1, "short_put_strike": 90, "long_put_strike": 80,
       "short_call_strike": 110, "long_call_strike": 120}


def _legs(sp_mid, sc_mid, lp_mid, lc_mid, sp_delta=-0.10, sc_delta=0.10):
    # build legs whose mids equal the requested values (bid=ask=mid for simplicity)
    return (opt(90, sp_mid, sp_mid, sp_delta), opt(80, lp_mid, lp_mid),
            opt(110, sc_mid, sc_mid, sc_delta), opt(120, lc_mid, lc_mid))


def test_profit_trigger_fires_at_target():
    # entry 1.00; current condor value 0.40 -> 60% captured -> profit trigger
    sp, lp, sc, lc = _legs(0.30, 0.30, 0.10, 0.10)  # value = 0.60-0.20 = 0.40
    s = position_status(POS, sp, lp, sc, lc, spot=100, dte=4)
    assert s["value_mid"] == pytest.approx(0.40)
    assert s["pl_pct"] == pytest.approx(60.0)
    assert any(t.startswith("PROFIT") for t in s["triggers"])


def test_tested_side_trigger_identifies_side():
    # short call delta 0.34 -> tested on the call side
    sp, lp, sc, lc = _legs(0.20, 0.60, 0.05, 0.10, sp_delta=-0.08, sc_delta=0.34)
    s = position_status(POS, sp, lp, sc, lc, spot=108, dte=4)
    tested = [t for t in s["triggers"] if t.startswith("TESTED")]
    assert tested and "call" in tested[0]


def test_time_trigger_only_with_profit_near_expiry():
    sp, lp, sc, lc = _legs(0.30, 0.30, 0.10, 0.10)  # value 0.40, 60% profit
    near = position_status(POS, sp, lp, sc, lc, spot=100, dte=1)
    assert any(t.startswith("TIME") for t in near["triggers"])
    # same profit but plenty of time left -> no TIME trigger
    far = position_status(POS, sp, lp, sc, lc, spot=100, dte=6)
    assert not any(t.startswith("TIME") for t in far["triggers"])


def test_stop_trigger_on_large_loss():
    # current value 3.20 vs entry 1.00 -> -220% -> stop (>= 2x)
    sp, lp, sc, lc = _legs(2.00, 1.50, 0.15, 0.15)  # value = 3.50 - 0.30 = 3.20
    s = position_status(POS, sp, lp, sc, lc, spot=100, dte=3)
    assert s["pl_pct"] == pytest.approx(-220.0)
    assert any(t.startswith("STOP") for t in s["triggers"])


def test_no_triggers_when_calm_and_midflight():
    # small profit, deltas tame, lots of time -> no triggers
    sp, lp, sc, lc = _legs(0.45, 0.45, 0.10, 0.10, sp_delta=-0.10, sc_delta=0.10)
    s = position_status(POS, sp, lp, sc, lc, spot=100, dte=5)
    assert s["triggers"] == []
