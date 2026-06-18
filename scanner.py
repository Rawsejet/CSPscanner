import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from tradier_client import TradierClient, TradierAPIError
from utils import calculate_annualized_roc, calculate_iv_rank, get_date_range, is_date_between

# S&P 100 Tickers
SP100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "TSLA", "V",
    "JPM", "UNH", "MA", "JNJ", "AVGO", "HD", "PG", "COST", "ORCL", "ADBE",
    "MRK", "XOM", "CVX", "ABBV", "LLY", "PEP", "MCD", "TMO", "ACN", "ABT",
    "LIN", "CSCO", "WMT", "CRM", "DHR", "NEE", "BMY", "RTX", "TXN", "QCOM",
    "HON", "UNP", "PM", "UPS", "LOW", "AMT", "SPGI", "INTU", "BKNG", "CAT",
    "GS", "BLK", "ISRG", "MDLZ", "GILD", "MMM", "AXP", "SYK", "DE", "CI",
    "MO", "ZTS", "TJX", "CB", "BK", "SCHW", "SO", "PLD", "USB", "DUK",
    "BDX", "TGT", "COF", "ITW", "NSC", "APD", "EMR", "AMD", "SHW", "EQIX",
    "AON", "CL", "GD", "WM", "ICE", "GM", "F", "NKE", "ADI", "LRCX", "SPCX"
]


class ScanDiagnostics:
    """Tracks what happened during a scan for UI transparency."""
    def __init__(self):
        self.tickers_attempted = 0
        self.tickers_scanned = 0
        self.quotes_fetched = 0
        self.expiries_found = 0
        self.chains_fetched = 0
        self.contracts_evaluated = 0
        self.rejected_delta = 0
        self.rejected_spread = 0
        self.rejected_zero_bid = 0
        self.rejected_roc = 0
        self.rejected_otm = 0
        self.rejected_no_greeks = 0
        self.rejected_missing_fields = 0
        self.rejected_iv_rank = 0
        self.rejected_earnings = 0
        self.fundamentals_unavailable = 0
        self.api_errors: List[str] = []

    # Integer counters aggregated across per-ticker scans.
    _COUNTERS = (
        "tickers_attempted", "tickers_scanned", "quotes_fetched", "expiries_found",
        "chains_fetched", "contracts_evaluated", "rejected_no_greeks",
        "rejected_missing_fields", "rejected_delta", "rejected_zero_bid",
        "rejected_spread", "rejected_roc", "rejected_otm", "rejected_iv_rank",
        "rejected_earnings", "fundamentals_unavailable",
    )

    def merge(self, other: "ScanDiagnostics") -> None:
        """Fold another diagnostics object (from a worker thread) into this one."""
        for name in self._COUNTERS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.api_errors.extend(other.api_errors)

    def summary(self) -> Dict[str, Any]:
        return {
            "tickers_attempted": self.tickers_attempted,
            "tickers_scanned": self.tickers_scanned,
            "quotes_fetched": self.quotes_fetched,
            "expiries_found": self.expiries_found,
            "chains_fetched": self.chains_fetched,
            "contracts_evaluated": self.contracts_evaluated,
            "rejected_no_greeks": self.rejected_no_greeks,
            "rejected_missing_fields": self.rejected_missing_fields,
            "rejected_delta": self.rejected_delta,
            "rejected_zero_bid": self.rejected_zero_bid,
            "rejected_spread": self.rejected_spread,
            "rejected_roc": self.rejected_roc,
            "rejected_otm": self.rejected_otm,
            "rejected_iv_rank": self.rejected_iv_rank,
            "rejected_earnings": self.rejected_earnings,
            "fundamentals_unavailable": self.fundamentals_unavailable,
            "api_errors": len(self.api_errors),
        }


class CSPScanner:
    def __init__(self, client: TradierClient):
        self.client = client

    def scan(self, horizon: str = "Income", tickers: Optional[List[str]] = None,
             max_spread_pct: float = 0.15, min_iv_rank: float = 20.0,
             exclude_earnings: bool = False, max_workers: int = 8) -> pd.DataFrame:
        """
        Scans the universe for CSP signals.
        horizon: "Income" (7-14 days) or "Classic" (30-45 days)
        tickers: optional override list; defaults to SP100_TICKERS
        max_spread_pct: max bid-ask spread as fraction of mid-price (default 15%)
        min_iv_rank: minimum (realized-vol proxy) IV Rank, 0-100. Contracts whose
            underlying ranks below this are rejected so we don't sell "cheap"
            premium. Tickers whose IV Rank can't be computed are not filtered out.
        exclude_earnings: when True, drop contracts whose expiry window contains a
            *known* earnings date (binary event risk). Contracts on tickers whose
            earnings data is unavailable are kept and flagged "N/A" — an empty
            calendar must never read as "no earnings".
        max_workers: bounded concurrency for per-ticker fetches. Keeps the scan
            fast without overwhelming Tradier's rate limit (429s are retried in
            the client with backoff).
        """
        if horizon == "Income":
            start_days, end_days = 7, 14
        else:
            start_days, end_days = 30, 45

        target_start, target_end = get_date_range(start_days, end_days)
        universe = tickers or SP100_TICKERS

        signals: List[Dict[str, Any]] = []
        diag = ScanDiagnostics()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._scan_ticker, ticker, horizon,
                    target_start, target_end, max_spread_pct, min_iv_rank,
                    exclude_earnings
                ): ticker
                for ticker in universe
            }
            for future in as_completed(futures):
                # Auth errors (401/403) re-raise out of _scan_ticker so the whole
                # scan aborts with a clear message instead of silently emptying.
                ticker_signals, ticker_diag = future.result()
                signals.extend(ticker_signals)
                diag.merge(ticker_diag)

        df = pd.DataFrame(signals)

        # Sort by ROC descending so best signals appear first
        if not df.empty:
            df = df.sort_values("ROC", ascending=False).reset_index(drop=True)

        # Attach diagnostics as an attribute for the UI to consume
        df.attrs['diagnostics'] = diag
        return df

    def _scan_ticker(self, ticker: str, horizon: str, target_start, target_end,
                     max_spread_pct: float, min_iv_rank: float = 20.0,
                     exclude_earnings: bool = False
                     ) -> Tuple[List[Dict[str, Any]], ScanDiagnostics]:
        """Scan a single ticker. Returns (signals, per-ticker diagnostics).

        Transient API failures are recorded in the diagnostics and the ticker is
        skipped; authentication failures (401/403) are re-raised so the caller can
        abort the entire scan.
        """
        diag = ScanDiagnostics()
        signals: List[Dict[str, Any]] = []
        diag.tickers_attempted += 1

        try:
            # Fetch spot price
            spot = self.client.get_spot_price(ticker)
            diag.quotes_fetched += 1
            if spot == 0:
                return signals, diag
            diag.tickers_scanned += 1

            # Fetch & filter expiries (ignore unparseable dates)
            available_expiries = self.client.get_available_expirations(ticker)
            diag.expiries_found += len(available_expiries)

            valid_expiries = []
            for exp in available_expiries:
                if not isinstance(exp, str):
                    continue
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if is_date_between(exp_date, target_start, target_end):
                    valid_expiries.append(exp)
            if not valid_expiries:
                return signals, diag

            # Fetch fundamentals ONCE per ticker (reused across expiries).
            # If they're unavailable, we must NOT report "no earnings" — track
            # availability so the event flags can degrade to "N/A" instead.
            try:
                earnings_dates = self.client.get_earnings_dates(ticker)
                dividend_dates = self.client.get_dividend_dates(ticker)
                fundamentals_available = True
            except TradierAPIError:
                earnings_dates = []
                dividend_dates = []
                fundamentals_available = False
                diag.fundamentals_unavailable += 1

            # One history fetch powers both the 50-day MA (informational) and the
            # realized-vol IV Rank proxy (a filter).
            ma50, iv_rank = self._technicals(ticker)

            today = datetime.now().date()

            for expiry_date in valid_expiries:
                chain = self.client.get_option_chain(ticker, expiry_date)
                if not chain:
                    continue
                diag.chains_fetched += 1
                expiry_d = datetime.strptime(expiry_date, "%Y-%m-%d").date()

                # Event risk depends only on the expiry window, so resolve it once
                # per expiry. Unknown (data unavailable) must read "N/A", not "NO".
                if fundamentals_available:
                    has_earnings = self._date_in_window(earnings_dates, today, expiry_d)
                    has_dividend = self._date_in_window(dividend_dates, today, expiry_d)
                    earnings_flag = "YES" if has_earnings else "NO"
                    dividend_flag = "YES" if has_dividend else "NO"
                else:
                    has_earnings = has_dividend = False
                    earnings_flag = dividend_flag = "N/A"

                puts = [c for c in chain if c.get('option_type') == 'put']

                for put in puts:
                    diag.contracts_evaluated += 1
                    strike = put.get('strike')
                    bid = put.get('bid')
                    ask = put.get('ask')

                    # Skip if any essential pricing field is missing
                    if strike is None or bid is None or ask is None:
                        diag.rejected_missing_fields += 1
                        continue

                    # Zero-bid: no real market, skip
                    if bid <= 0:
                        diag.rejected_zero_bid += 1
                        continue

                    mid = (bid + ask) / 2

                    # Read delta from nested greeks — skip if missing
                    raw_delta = (put.get('greeks') or {}).get('delta')
                    if raw_delta is None:
                        diag.rejected_no_greeks += 1
                        continue

                    # Puts always have delta in [-1, 0]; normalise the sign so the
                    # filter is robust to feeds that report the magnitude.
                    delta = -abs(raw_delta)

                    # 1. Delta filter: -0.25 to -0.10
                    if not (-0.25 <= delta <= -0.10):
                        diag.rejected_delta += 1
                        continue

                    # 2. Liquidity: bid-ask spread <= max_spread_pct of mid
                    spread = ask - bid
                    if (spread / mid) > max_spread_pct:
                        diag.rejected_spread += 1
                        continue

                    # 3. OTM check: strike < spot
                    if strike >= spot:
                        diag.rejected_otm += 1
                        continue

                    # 4. ROC: annualized >= 12%
                    dte = (expiry_d - today).days
                    if dte <= 0:
                        continue
                    roc = calculate_annualized_roc(mid, strike, dte)
                    if roc < 12:
                        diag.rejected_roc += 1
                        continue

                    # 5. Volatility: skip "cheap" premium (low IV Rank). When IV
                    # Rank can't be computed (thin history) we don't filter.
                    if iv_rank is not None and iv_rank < min_iv_rank:
                        diag.rejected_iv_rank += 1
                        continue

                    # 6. Event risk: optionally hard-exclude a *known* earnings
                    # date inside the window. Unknown ("N/A") contracts are kept
                    # so an unavailable calendar can't silently empty the scan.
                    if exclude_earnings and has_earnings:
                        diag.rejected_earnings += 1
                        continue

                    # Downside framing: break-even price, % cushion from spot to
                    # break-even, cash-secured capital tied up, and the max loss
                    # if the stock goes to zero.
                    breakeven = strike - mid
                    cushion_pct = (spot - breakeven) / spot * 100 if spot else None
                    capital = strike * 100
                    max_loss = breakeven * 100

                    # Technical alignment bonus
                    below_ma50 = strike < ma50 if ma50 else None

                    signals.append({
                        "Ticker": ticker,
                        "Cycle": horizon,
                        "Strike": strike,
                        "Spot": spot,
                        "Break-even": round(breakeven, 2),
                        "Cushion %": round(cushion_pct, 2) if cushion_pct is not None else None,
                        "DTE": dte,
                        "Premium": round(mid, 2),
                        "ROC": round(roc, 2),
                        "IV Rank": round(iv_rank, 1) if iv_rank is not None else None,
                        "Delta": round(delta, 3),
                        "Capital": round(capital, 2),
                        "Max Loss": round(max_loss, 2),
                        "Earnings Warning": earnings_flag,
                        "Dividend Warning": dividend_flag,
                        "Below 50d MA": "YES" if below_ma50 else ("NO" if below_ma50 is False else "N/A"),
                        "Signal Strength": self._signal_strength(roc, delta, has_earnings, has_dividend),
                    })
        except TradierAPIError as e:
            if e.status_code in (401, 403):
                raise
            diag.api_errors.append(f"{ticker}: {e}")

        return signals, diag

    @staticmethod
    def _date_in_window(date_strings: List[str], start, end) -> bool:
        """True if any parseable date string falls within [start, end]."""
        for d in date_strings:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if is_date_between(dt, start, end):
                return True
        return False

    def scan_income(self, tickers: Optional[List[str]] = None,
                    max_spread_pct: float = 0.15) -> pd.DataFrame:
        return self.scan("Income", tickers, max_spread_pct)

    def scan_classic(self, tickers: Optional[List[str]] = None,
                     max_spread_pct: float = 0.15) -> pd.DataFrame:
        return self.scan("Classic", tickers, max_spread_pct)

    @staticmethod
    def _signal_strength(roc: float, delta: float, has_earnings: bool, has_dividend: bool) -> str:
        """
        Signal Strength tiers (aligned with SPEC.md Low/Med/High).

        High: ROC > 25%, no earnings/dividend risk, delta in sweet spot (-0.15 to -0.20)
        Med:  ROC 15-25%, minor risk factors
        Low:  ROC 12-15%, or multiple risk flags
        """
        score = 0

        # ROC contribution
        if roc > 25:
            score += 3
        elif roc > 15:
            score += 2
        else:
            score += 1

        # Delta sweet spot bonus
        if -0.20 <= delta <= -0.15:
            score += 1

        # Risk penalties
        if has_earnings:
            score -= 2
        if has_dividend:
            score -= 1

        if score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"

    def _technicals(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch one year of daily history once and derive both technicals.

        Returns (ma50, iv_rank) where either may be None if the data is too thin.
        """
        try:
            history = self.client.get_historical_prices(symbol, interval="daily", span="year")
        except (TradierAPIError, KeyError, TypeError):
            return None, None
        if not history:
            return None, None

        closes = [float(d['close']) for d in history if isinstance(d, dict) and 'close' in d]

        # 50-day simple moving average (need >= 20 days to be meaningful)
        last50 = closes[-50:]
        ma50 = sum(last50) / len(last50) if len(last50) >= 20 else None

        iv_rank = calculate_iv_rank(closes)
        return ma50, iv_rank
