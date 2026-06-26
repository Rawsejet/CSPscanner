"""Weekly iron-condor entry scanner.

Mirrors ``scanner.CSPScanner``: the same bounded ``ThreadPoolExecutor`` fan-out,
the same per-ticker diagnostics that merge across worker threads, and the same
auth-error-aborts / transient-error-records contract. Where the CSP scanner
filters individual puts, this builds one iron condor per ticker on the nearest
weekly expiry and ranks them by *real edge* — the variance risk premium (implied
vs realized vol) and whether the 4-leg credit is actually fillable — rather than
by headline premium.

Per the design decision, negative-edge / negative-VRP / thin-credit condors are
**flagged, not hidden** (only structural failures — no condor could be built —
drop a ticker). Earnings inside the weekly window are excluded by default
(binary-event risk that dwarfs a week's premium).
"""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tradier_client import TradierClient, TradierAPIError
from utils import is_date_between
from ic_metrics import (
    build_condor, condor_metrics, atm_implied_vol, realized_vol, edge_grade,
    resolve_wing,
)
from scanner import SCAN_UNIVERSE


class ICDiagnostics:
    """Tracks what happened during an iron-condor scan, for UI transparency."""

    def __init__(self):
        self.tickers_attempted = 0
        self.tickers_scanned = 0
        self.expiries_found = 0
        self.chains_fetched = 0
        self.condors_built = 0
        self.rejected_no_condor = 0
        self.rejected_no_chain = 0
        self.rejected_earnings = 0
        self.rejected_thin_credit = 0
        self.flagged_negative_vrp = 0
        self.flagged_negative_edge = 0
        self.flagged_thin_credit = 0
        self.flagged_earnings = 0
        self.fundamentals_unavailable = 0
        self.api_errors: List[str] = []

    _COUNTERS = (
        "tickers_attempted", "tickers_scanned", "expiries_found", "chains_fetched",
        "condors_built", "rejected_no_condor", "rejected_no_chain",
        "rejected_earnings", "rejected_thin_credit", "flagged_negative_vrp",
        "flagged_negative_edge", "flagged_thin_credit", "flagged_earnings",
        "fundamentals_unavailable",
    )

    def merge(self, other: "ICDiagnostics") -> None:
        for name in self._COUNTERS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.api_errors.extend(other.api_errors)

    def summary(self) -> Dict[str, Any]:
        out = {name: getattr(self, name) for name in self._COUNTERS}
        out["api_errors"] = len(self.api_errors)
        return out


class ICScanner:
    def __init__(self, client: TradierClient):
        self.client = client

    def scan(self, tickers: Optional[List[str]] = None, short_delta: float = 0.10,
             wing_mode: str = "pct", wing_value: float = 2.0,
             dte_min: int = 2, dte_max: int = 12,
             exclude_earnings: bool = True, min_natural_credit: Optional[float] = None,
             max_workers: int = 8) -> pd.DataFrame:
        """Scan the universe for one weekly iron condor per ticker.

        short_delta: target |delta| for both short strikes (far-OTM = smaller).
        wing_mode/wing_value: how the dollar wing is sized per ticker — ``"pct"``
            (``wing_value``% of that ticker's spot, so wings scale with price) or
            ``"fixed"`` (a flat ``wing_value`` dollars). The wing defines max loss.
        dte_min/dte_max: weekly expiry window; the nearest expiry inside it is used.
        exclude_earnings: drop tickers with a known earnings date in the window.
        min_natural_credit: if set, drop condors whose natural (sell-bid/buy-ask)
            credit is below it — a hard liquidity gate. None keeps them (flagged).
        """
        universe = tickers or SCAN_UNIVERSE
        signals: List[Dict[str, Any]] = []
        diag = ICDiagnostics()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._scan_ticker, ticker, short_delta, wing_mode, wing_value,
                    dte_min, dte_max, exclude_earnings, min_natural_credit
                ): ticker
                for ticker in universe
            }
            for future in as_completed(futures):
                ticker_signals, ticker_diag = future.result()
                signals.extend(ticker_signals)
                diag.merge(ticker_diag)

        df = pd.DataFrame(signals)
        if not df.empty:
            # Default order: richest VRP first, then least-negative edge gap.
            df = df.sort_values(["VRP", "Edge (pp)"], ascending=False).reset_index(drop=True)

        df.attrs["diagnostics"] = diag
        return df

    def _scan_ticker(self, ticker: str, short_delta: float, wing_mode: str,
                     wing_value: float, dte_min: int, dte_max: int,
                     exclude_earnings: bool, min_natural_credit: Optional[float]
                     ) -> Tuple[List[Dict[str, Any]], ICDiagnostics]:
        diag = ICDiagnostics()
        signals: List[Dict[str, Any]] = []
        diag.tickers_attempted += 1

        try:
            spot = self.client.get_spot_price(ticker)
            if not spot:
                return signals, diag
            diag.tickers_scanned += 1

            # Size the wing per ticker so it scales with price (snapped to listed
            # strikes inside build_condor).
            wing_width = resolve_wing(spot, wing_mode, wing_value)

            weekly = self._nearest_weekly(
                self.client.get_available_expirations(ticker), dte_min, dte_max)
            if weekly is None:
                return signals, diag
            expiry, dte = weekly
            diag.expiries_found += 1

            # Earnings once per ticker; unavailable data must degrade to "N/A".
            try:
                earnings_dates = self.client.get_earnings_dates(ticker)
                fundamentals_available = True
            except TradierAPIError:
                earnings_dates = []
                fundamentals_available = False
                diag.fundamentals_unavailable += 1

            rvol = self._realized_vol(ticker)

            chain = self.client.get_option_chain(ticker, expiry)
            if not chain:
                diag.rejected_no_chain += 1
                return signals, diag
            diag.chains_fetched += 1

            puts = [o for o in chain if o.get("option_type") == "put"]
            calls = [o for o in chain if o.get("option_type") == "call"]

            condor = build_condor(spot, puts, calls, short_delta, wing_width)
            if condor is None:
                diag.rejected_no_condor += 1
                return signals, diag

            atm_iv = atm_implied_vol(calls, spot)
            m = condor_metrics(condor, spot, dte, atm_iv=atm_iv, realized_volatility=rvol)
            diag.condors_built += 1

            # Earnings window flag/filter.
            today = datetime.now().date()
            expiry_d = datetime.strptime(expiry, "%Y-%m-%d").date()
            if fundamentals_available:
                has_earnings = self._date_in_window(earnings_dates, today, expiry_d)
                earnings_flag = "YES" if has_earnings else "NO"
            else:
                has_earnings = False
                earnings_flag = "N/A"
            if exclude_earnings and has_earnings:
                diag.rejected_earnings += 1
                return signals, diag

            nat = m["credit_natural"]
            if min_natural_credit is not None and (nat is None or nat < min_natural_credit):
                diag.rejected_thin_credit += 1
                return signals, diag

            # Flag counters (shown, not hidden).
            if m["vrp_ratio"] is not None and m["vrp_ratio"] < 1.0:
                diag.flagged_negative_vrp += 1
            if m["edge_gap"] is not None and m["edge_gap"] < 0:
                diag.flagged_negative_edge += 1
            if nat is not None and nat <= 0:
                diag.flagged_thin_credit += 1
            if has_earnings:
                diag.flagged_earnings += 1

            signals.append(self._row(ticker, spot, expiry, dte, condor, m,
                                     atm_iv, rvol, earnings_flag))
        except TradierAPIError as e:
            if e.status_code in (401, 403):
                raise
            diag.api_errors.append(f"{ticker}: {e}")

        return signals, diag

    @staticmethod
    def _row(ticker, spot, expiry, dte, condor, m, atm_iv, rvol, earnings_flag):
        def r(v, n=2):
            return round(v, n) if v is not None else None
        return {
            "Ticker": ticker,
            "Grade": edge_grade(m),
            "Spot": r(spot),
            "Expiry": expiry,
            "DTE": dte,
            "Short Put": condor["short_put_strike"],
            "Long Put": condor["long_put_strike"],
            "Short Call": condor["short_call_strike"],
            "Long Call": condor["long_call_strike"],
            "Wing": condor["put_wing"],
            "Body": condor["body"],
            "Credit": r(m["credit_mid"]),
            "Credit (nat)": r(m["credit_natural"]),
            "Max Loss": r(m["max_loss"], 0),
            "RoR %": r(m["return_on_risk"], 1),
            "PoP %": r(m["pop"], 1),
            "BE WR %": r(m["breakeven_win_rate"], 1),
            "Edge (pp)": r(m["edge_gap"], 1),
            "VRP": r(m["vrp_ratio"]),
            "ATM IV %": r(atm_iv * 100, 0) if atm_iv else None,
            "RV %": r(rvol * 100, 0) if rvol else None,
            "Put EM x": r(m["em_mult_put"], 1),
            "Call EM x": r(m["em_mult_call"], 1),
            "BE Low": r(m["breakeven_low"]),
            "BE High": r(m["breakeven_high"]),
            "Earnings": earnings_flag,
        }

    @staticmethod
    def _nearest_weekly(expirations: List[str], dte_min: int, dte_max: int
                        ) -> Optional[Tuple[str, int]]:
        """The expiry inside [dte_min, dte_max] closest to a 7-day target.

        Falls back to the nearest future expiry if none lands in the window, so a
        ticker with sparse weeklies still produces a candidate.
        """
        today = datetime.now().date()
        cand: List[Tuple[int, str]] = []
        for e in expirations:
            if not isinstance(e, str):
                continue
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - today).days
            if dte > 0:
                cand.append((dte, e))
        if not cand:
            return None
        in_window = [c for c in cand if dte_min <= c[0] <= dte_max]
        pool = in_window or cand
        pool.sort(key=lambda x: (abs(x[0] - 7), x[0]))
        return pool[0][1], pool[0][0]

    @staticmethod
    def _date_in_window(date_strings: List[str], start, end) -> bool:
        for d in date_strings:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if is_date_between(dt, start, end):
                return True
        return False

    def _realized_vol(self, ticker: str, window: int = 20) -> Optional[float]:
        """Annualized realized vol from one year of daily closes (for the VRP)."""
        try:
            history = self.client.get_historical_prices(ticker, interval="daily", span="year")
        except (TradierAPIError, KeyError, TypeError):
            return None
        if not history:
            return None
        closes = [float(d["close"]) for d in history if isinstance(d, dict) and "close" in d]
        return realized_vol(closes, window=window)
