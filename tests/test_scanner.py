import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from utils import calculate_annualized_roc, calculate_iv_rank
from scanner import CSPScanner
from tradier_client import TradierClient, TradierAPIError


# ── Utility: date strings in the Income window (7-14 days) ──────────────────

def _income_expiry(offset_days: int = 10) -> str:
    return (datetime.now().date() + timedelta(days=offset_days)).isoformat()


# ── Utility: synthetic price histories with known volatility profiles ───────

def _vol_then_calm_history():
    """Volatile first, calm recently → current realized vol near the LOW → low IV Rank."""
    closes, price = [], 100.0
    for i in range(200):
        price *= 1.05 if i % 2 == 0 else 1 / 1.05
        closes.append({"close": round(price, 4)})
    for i in range(40):
        price *= 1.0005 if i % 2 == 0 else 1 / 1.0005
        closes.append({"close": round(price, 4)})
    return closes


def _calm_then_vol_history():
    """Calm first, volatile recently → current realized vol near the HIGH → high IV Rank."""
    closes, price = [], 100.0
    for i in range(200):
        price *= 1.0005 if i % 2 == 0 else 1 / 1.0005
        closes.append({"close": round(price, 4)})
    for i in range(40):
        price *= 1.05 if i % 2 == 0 else 1 / 1.05
        closes.append({"close": round(price, 4)})
    return closes


# ── utils.py tests ─────────────────────────────────────────────────────────

def test_roc_calculation():
    # Premium: 2.0, Strike: 100, DTE: 30
    # ROC = (2/100) * (365/30) = 24.33%
    roc = calculate_annualized_roc(2.0, 100.0, 30)
    assert roc == pytest.approx(24.333, 0.01)


def test_roc_zero_dte():
    roc = calculate_annualized_roc(2.0, 100.0, 0)
    assert roc == 0.0


def test_iv_rank_constant_prices_is_none():
    """No price movement → no volatility dispersion → cannot rank."""
    assert calculate_iv_rank([100.0] * 100) is None


def test_iv_rank_insufficient_data_is_none():
    assert calculate_iv_rank([100.0, 101.0, 99.0]) is None


def test_iv_rank_low_when_currently_calm():
    closes = [d["close"] for d in _vol_then_calm_history()]
    rank = calculate_iv_rank(closes)
    assert rank is not None and rank < 20


def test_iv_rank_high_when_currently_volatile():
    closes = [d["close"] for d in _calm_then_vol_history()]
    rank = calculate_iv_rank(closes)
    assert rank is not None and rank > 80


# ── tradier_client.py tests ────────────────────────────────────────────────

class TestExpirationParsing:
    def test_dict_with_list(self):
        """Multiple expirations come back as a dict with a list value."""
        client = TradierClient("fake")
        with patch.object(client, "_request", return_value={
            "expirations": {"date": ["2026-07-01", "2026-07-15", "2026-08-01"]}
        }):
            result = client.get_available_expirations("AAPL")
            assert result == ["2026-07-01", "2026-07-15", "2026-08-01"]

    def test_dict_with_single_string(self):
        """Single expiration comes back as a plain string, not a list."""
        client = TradierClient("fake")
        with patch.object(client, "_request", return_value={
            "expirations": {"date": "2026-07-17"}
        }):
            result = client.get_available_expirations("AAPL")
            assert result == ["2026-07-17"]

    def test_empty_response(self):
        client = TradierClient("fake")
        with patch.object(client, "_request", return_value={}):
            result = client.get_available_expirations("AAPL")
            assert result == []

    def test_list_response(self):
        """Edge case: expirations already a list."""
        client = TradierClient("fake")
        with patch.object(client, "_request", return_value={
            "expirations": ["2026-07-01", "2026-07-15"]
        }):
            result = client.get_available_expirations("AAPL")
            assert result == ["2026-07-01", "2026-07-15"]


class TestTradierClientErrors:
    def test_sandbox_url(self):
        client = TradierClient("fake", use_sandbox=True)
        assert "sandbox" in client.base_url

    def test_production_url(self):
        client = TradierClient("fake", use_sandbox=False)
        assert "sandbox" not in client.base_url


class TestFundamentalsParsing:
    """The beta fundamentals calendar replaces the old (wrong) markets/calendar."""

    def test_earnings_dates_parsed_and_filtered(self):
        client = TradierClient("fake")
        payload = [
            {"request": "AAPL", "type": "Symbol", "results": [
                {"type": "Stock", "tables": {"corporate_calendars": [
                    {"event": "Q3 2026 Earnings Release", "event_type": 14,
                     "begin_date_time": "2026-06-25T00:00:00"},
                    {"event": "Annual Shareholders Meeting", "event_type": 7,
                     "begin_date_time": "2026-09-01T00:00:00"},
                ]}},
            ]},
        ]
        with patch.object(client, "_request", return_value=payload):
            # Datetime is normalized to a date; the non-earnings event is dropped.
            assert client.get_earnings_dates("AAPL") == ["2026-06-25"]

    def test_dividend_dates_parsed_from_nested_row(self):
        client = TradierClient("fake")
        payload = [
            {"request": "AAPL", "type": "Symbol", "results": [
                {"type": "Stock", "tables": {"cash_dividends": [
                    {"cash_dividend": {"ex_date": "2026-07-10T00:00:00"}},
                ]}},
            ]},
        ]
        with patch.object(client, "_request", return_value=payload):
            assert client.get_dividend_dates("AAPL") == ["2026-07-10"]

    def test_fundamentals_error_propagates(self):
        """A failed request must raise so the scanner can mark the data N/A."""
        client = TradierClient("fake")
        with patch.object(client, "_request", side_effect=TradierAPIError("404", 404)):
            with pytest.raises(TradierAPIError):
                client.get_earnings_dates("AAPL")

    def test_empty_fundamentals_returns_empty(self):
        client = TradierClient("fake")
        with patch.object(client, "_request", return_value=[]):
            assert client.get_earnings_dates("AAPL") == []


# ── scanner.py tests ───────────────────────────────────────────────────────

class TestScannerFiltering:
    def _make_mock_client(self, extra_chains=None):
        mock = MagicMock()
        mock.get_spot_price.return_value = 150.0
        mock.get_available_expirations.return_value = [_income_expiry()]
        mock.get_earnings_dates.return_value = []
        mock.get_dividend_dates.return_value = []
        mock.get_historical_prices.return_value = [
            {"close": 148.0} for _ in range(50)
        ]

        if extra_chains is not None:
            # Return a different chain per call
            mock.get_option_chain.side_effect = extra_chains
        else:
            mock.get_option_chain.return_value = [
                {
                    'option_type': 'put',
                    'strike': 140.0,
                    'bid': 1.50,
                    'ask': 1.60,
                    'greeks': {'delta': -0.15},
                },
            ]
        return mock

    def test_valid_put_passes(self):
        """A put with good delta, spread, ROC, and OTM should survive."""
        mock_client = self._make_mock_client()
        scanner = CSPScanner(mock_client)

        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert len(df) == 1
            assert df.iloc[0]["Ticker"] == "TEST"
            assert df.iloc[0]["Strike"] == 140.0
        finally:
            scanner_module.SP100_TICKERS = original

    def test_delta_too_negative_rejected(self):
        """Delta -0.40 is outside -0.25..-0.10 range."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 1.50,
                'ask': 1.60,
                'greeks': {'delta': -0.40},
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_delta_too_small_rejected(self):
        """Delta -0.05 is outside -0.25..-0.10 range."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 1.50,
                'ask': 1.60,
                'greeks': {'delta': -0.05},
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_missing_greeks_rejected(self):
        """A contract with no greeks dict must be skipped, not fall back to -0.20."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 1.50,
                'ask': 1.60,
                # No greeks key at all
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_zero_bid_rejected(self):
        """A contract with bid=0 has no real market; skip it."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 0.00,
                'ask': 0.50,
                'greeks': {'delta': -0.15},
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_wide_spread_rejected(self):
        """Spread wider than max_spread_pct of mid-price is rejected (default 15%)."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 0.10,
                'ask': 0.50,
                'greeks': {'delta': -0.15},
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_itm_strike_rejected(self):
        """Strike >= spot is ATM/ITM, rejected for OTM filter."""
        mock_client = self._make_mock_client()
        mock_client.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 160.0,  # Above spot of 150
                'bid': 5.00,
                'ask': 5.20,
                'greeks': {'delta': -0.30},
            },
        ]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_low_iv_rank_rejected(self):
        """A valid put on a ticker with low IV Rank should be filtered out."""
        mock_client = self._make_mock_client()
        mock_client.get_historical_prices.return_value = _vol_then_calm_history()
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income", min_iv_rank=20)
            assert df.empty
        finally:
            scanner_module.SP100_TICKERS = original

    def test_high_iv_rank_passes(self):
        """A valid put on a ticker with high IV Rank survives and reports the rank."""
        mock_client = self._make_mock_client()
        mock_client.get_historical_prices.return_value = _calm_then_vol_history()
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income", min_iv_rank=20)
            assert len(df) == 1
            assert df.iloc[0]["IV Rank"] > 20
        finally:
            scanner_module.SP100_TICKERS = original

    def test_uncomputable_iv_rank_not_filtered(self):
        """When IV Rank can't be computed (flat history), don't filter the contract."""
        mock_client = self._make_mock_client()  # default history is constant
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income", min_iv_rank=20)
            assert len(df) == 1
            assert pd.isna(df.iloc[0]["IV Rank"])
        finally:
            scanner_module.SP100_TICKERS = original

    def test_signal_strength_levels(self):
        """Signal Strength should be Low/Medium/High per SPEC.md."""
        # High ROC + sweet spot delta → High
        assert CSPScanner._signal_strength(30, -0.15, False, False) == "High"
        # Medium ROC + sweet spot → Medium
        assert CSPScanner._signal_strength(18, -0.15, False, False) == "Medium"
        # Low ROC + non-sweet-spot delta → Low
        assert CSPScanner._signal_strength(13, -0.10, False, False) == "Low"
        # High ROC, sweet-spot delta, earnings penalty → score 3+1-2=2 → Medium
        assert CSPScanner._signal_strength(30, -0.15, True, False) == "Medium"
        # High ROC + dividend penalty → score 3+1-1=3 → Medium
        assert CSPScanner._signal_strength(30, -0.15, False, True) == "Medium"
        # Earnings + dividend double penalty → Low
        assert CSPScanner._signal_strength(30, -0.15, True, True) == "Low"

    def test_earnings_warning_flagged(self):
        """If earnings falls within the expiry window, flag YES."""
        earnings_in_window = [_income_expiry()]  # Same window as expiry
        mock_client = self._make_mock_client()
        mock_client.get_earnings_dates.return_value = earnings_in_window

        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert len(df) == 1
            assert df.iloc[0]["Earnings Warning"] == "YES"
        finally:
            scanner_module.SP100_TICKERS = original

    def test_results_sorted_by_roc_desc(self):
        """Results should be sorted by ROC descending."""
        mock = self._make_mock_client()
        # Create chain with multiple puts at different strikes (different ROCs)
        # Put A: strike 135, mid 0.80, DTE 10 → ROC = (0.8/135)*(365/10) = 21.6%
        # Put B: strike 140, mid 1.55, DTE 10 → ROC = (1.55/140)*(365/10) = 40.4%
        # Put B should rank first (higher ROC)
        mock.get_option_chain.return_value = [
            {
                'option_type': 'put',
                'strike': 135.0,
                'bid': 0.78,
                'ask': 0.82,
                'greeks': {'delta': -0.12},
            },
            {
                'option_type': 'put',
                'strike': 140.0,
                'bid': 1.50,
                'ask': 1.60,
                'greeks': {'delta': -0.20},
            },
        ]
        scanner = CSPScanner(mock)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            assert len(df) == 2, f"Expected 2 signals, got {len(df)}"
            assert df.iloc[0]["ROC"] >= df.iloc[1]["ROC"], \
                f"Expected ROC desc: {df.iloc[0]['ROC']} >= {df.iloc[1]['ROC']}"
            assert df.iloc[0]["Strike"] == 140.0  # Higher ROC put first
        finally:
            scanner_module.SP100_TICKERS = original

    def test_diagnostics_attached(self):
        """Scan should attach diagnostics to the DataFrame."""
        mock_client = self._make_mock_client()
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            df = scanner.scan(horizon="Income")
            diag = df.attrs.get('diagnostics')
            assert diag is not None
            summary = diag.summary()
            assert summary["tickers_scanned"] >= 1
        finally:
            scanner_module.SP100_TICKERS = original

    def test_break_even_cushion_capital_columns(self):
        """Each signal exposes break-even, cushion %, capital, and max loss."""
        # Default mock: strike 140, bid/ask 1.50/1.60 (mid 1.55), spot 150.
        mock_client = self._make_mock_client()
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            row = scanner.scan(horizon="Income").iloc[0]
            assert row["Break-even"] == pytest.approx(138.45, abs=0.01)
            assert row["Cushion %"] == pytest.approx(7.7, abs=0.1)
            assert row["Capital"] == pytest.approx(14000.0)
            assert row["Max Loss"] == pytest.approx(13845.0, abs=0.01)
        finally:
            scanner_module.SP100_TICKERS = original

    def test_exclude_earnings_drops_known_event(self):
        """exclude_earnings drops a contract whose window holds a known earnings date."""
        mock_client = self._make_mock_client()
        mock_client.get_earnings_dates.return_value = [_income_expiry()]
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            assert scanner.scan(horizon="Income", exclude_earnings=True).empty
            kept = scanner.scan(horizon="Income", exclude_earnings=False)
            assert len(kept) == 1
            assert kept.iloc[0]["Earnings Warning"] == "YES"
        finally:
            scanner_module.SP100_TICKERS = original

    def test_unavailable_fundamentals_flag_na_not_no(self):
        """If earnings data can't be fetched, flag N/A (never a false 'NO')."""
        mock_client = self._make_mock_client()
        mock_client.get_earnings_dates.side_effect = TradierAPIError("not entitled", 404)
        scanner = CSPScanner(mock_client)
        import scanner as scanner_module
        original = scanner_module.SP100_TICKERS
        scanner_module.SP100_TICKERS = ["TEST"]
        try:
            # Even with exclude on, an N/A contract is kept (unknown != has-earnings).
            df = scanner.scan(horizon="Income", exclude_earnings=True)
            assert len(df) == 1
            assert df.iloc[0]["Earnings Warning"] == "N/A"
            assert df.iloc[0]["Dividend Warning"] == "N/A"
            assert df.attrs["diagnostics"].summary()["fundamentals_unavailable"] == 1
        finally:
            scanner_module.SP100_TICKERS = original
