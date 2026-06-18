import requests
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradierAPIError(Exception):
    """Raised for non-recoverable Tradier API errors (auth, server)."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TradierClient:
    PRODUCTION_BASE = "https://api.tradier.com/v1"
    SANDBOX_BASE = "https://sandbox.tradier.com/v1"
    # Beta fundamentals (earnings/dividends) live under /beta, not /v1, and are
    # production-only — sandbox tokens 404 here, which the scanner treats as
    # "data unavailable" (flags N/A) rather than "no events".
    FUNDAMENTALS_BASE = "https://api.tradier.com/beta"

    def __init__(self, access_token: str, use_sandbox: bool = False):
        self.access_token = access_token
        self.base_url = self.SANDBOX_BASE if use_sandbox else self.PRODUCTION_BASE
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                 _retry: int = 0, base: Optional[str] = None) -> Dict[str, Any]:
        url = f"{base or self.base_url}/{endpoint}"
        try:
            # Concurrency is bounded by the scanner's thread pool; bursts that hit
            # the rate limit are handled by the 429 retry/backoff below.
            response = requests.get(url, headers=self.headers, params=params, timeout=15)

            # Retry on 429 with exponential backoff
            if response.status_code == 429 and _retry < 3:
                wait = float(response.headers.get("Retry-After", 2 ** _retry))
                logger.info(f"Rate limited on {endpoint}, sleeping {wait:.1f}s (retry {_retry + 1}/3)")
                time.sleep(wait)
                return self._request(endpoint, params, _retry + 1, base=base)

            response.raise_for_status()

            data = response.json()
            logger.debug(f"Response for {endpoint}: keys={list(data.keys()) if isinstance(data, dict) else type(data).__name__}")

            if isinstance(data, dict) and "error" in data:
                message = data.get("message", data["error"])
                raise TradierAPIError(message, response.status_code)

            return data

        except TradierAPIError:
            raise
        except requests.exceptions.HTTPError:
            if response.status_code == 401:
                raise TradierAPIError(
                    f"Authentication failed (401). Check your API token and "
                    f"that {'sandbox' if 'sandbox' in self.base_url else 'production'} mode matches your token type.",
                    401
                )
            if response.status_code == 403:
                raise TradierAPIError(
                    f"Forbidden (403). This may indicate a sandbox token sent to production or vice-versa.",
                    403
                )
            if response.status_code == 429:
                raise TradierAPIError("Rate limit exceeded. Too many requests.", 429)
            raise TradierAPIError(f"HTTP {response.status_code} for {endpoint}", response.status_code)
        except requests.exceptions.RequestException as e:
            raise TradierAPIError(f"Request failed: {e}")

    def get_available_expirations(self, symbol: str) -> List[str]:
        """
        Fetch all available expiration dates for a symbol.

        Tradier returns:
          Single expiration: {"expirations": {"date": "2026-07-17"}}
          Multiple:          {"expirations": {"date": ["2026-06-19", "2026-06-26", ...]}}
        """
        endpoint = "markets/options/expirations"
        params = {"symbol": symbol}
        data = self._request(endpoint, params)

        try:
            expirations = data.get('expirations', {})
            if isinstance(expirations, dict):
                raw = expirations.get('date', [])
                # Normalise: single string -> list
                if isinstance(raw, str):
                    return [raw]
                return list(raw) if raw else []
            # Already a list (edge case)
            if isinstance(expirations, list):
                return expirations
            return []
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not parse expirations for {symbol}: {e}")
            return []

    def get_spot_price(self, symbol: str) -> float:
        """Fetch the current market price for a ticker.

        Prefers `last`, but `last` can be null outside regular hours, so it falls
        back to the bid/ask mid and then the prior close.
        """
        endpoint = "markets/quotes"
        params = {"symbols": symbol}
        data = self._request(endpoint, params)

        try:
            quote = data['quotes']['quote']
            for key in ('last', 'close', 'prevclose'):
                val = quote.get(key)
                if val:
                    return float(val)
            bid, ask = quote.get('bid'), quote.get('ask')
            if bid and ask:
                return float((bid + ask) / 2)
            return 0.0
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not fetch spot price for {symbol}: {e}")
            return 0.0

    def get_option_chain(self, symbol: str, expiration_date: str) -> List[Dict[str, Any]]:
        """
        Fetch the full option chain for a symbol and specific expiration.
        Includes Greeks (delta, gamma, theta, vega, iv).
        expiration_date format: YYYY-MM-DD
        """
        endpoint = "markets/options/chains"
        params = {
            "symbol": symbol,
            "expiration": expiration_date,
            "greeks": "true"
        }
        data = self._request(endpoint, params)

        try:
            # Tradier returns: {"options": {"option": [{"contract": "...", ...}]}}
            # Single option: {"options": {"option": {"contract": "...", ...}}}
            raw = data.get('options', {}).get('option', [])
            if isinstance(raw, dict):
                return [raw]
            return list(raw) if raw else []
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not fetch option chain for {symbol} on {expiration_date}: {e}")
            return []

    @staticmethod
    def _normalize_date(raw: Any) -> Optional[str]:
        """Normalize 'YYYY-MM-DD' or 'YYYY-MM-DDThh:mm:ss' to 'YYYY-MM-DD'.

        Returns None for anything that isn't a parseable date. The fundamentals
        endpoints report datetimes (e.g. '2026-06-25T00:00:00'); the scanner's
        windowing parses strict '%Y-%m-%d', so we truncate and validate here.
        """
        if not isinstance(raw, str) or len(raw) < 10:
            return None
        candidate = raw[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            return None
        return candidate

    def _fundamentals_dates(self, symbol: str, endpoint: str, table: str,
                            date_keys, event_filter=None) -> List[str]:
        """Pull normalized YYYY-MM-DD dates out of a beta fundamentals response.

        The beta fundamentals endpoints return a list of
        ``{request, type, results: [{tables: {<table>: [ {...}, ... ]}}]}``,
        where ``results`` can hold several entries (some with a ``null`` table).
        For each row we take the first present ``date_keys`` value, optionally
        gated by ``event_filter``.

        Raises TradierAPIError if the request itself fails (e.g. the beta
        fundamentals data isn't entitled on this plan) so the caller can mark the
        data *unavailable* rather than reporting a false "no events".
        """
        data = self._request(endpoint, {"symbols": symbol}, base=self.FUNDAMENTALS_BASE)

        dates: List[str] = []
        try:
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for result in entry.get("results", []) or []:
                    if not isinstance(result, dict):
                        continue
                    rows = (result.get("tables") or {}).get(table) or []
                    if isinstance(rows, dict):
                        rows = [rows]
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        if event_filter and not event_filter(row):
                            continue
                        found = self._row_date(row, date_keys)
                        if found:
                            dates.append(found)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Could not parse {table} for {symbol}: {e}")
        return dates

    def _row_date(self, row: Dict[str, Any], date_keys) -> Optional[str]:
        """First normalizable date found in ``row`` across ``date_keys``."""
        for key in date_keys:
            norm = self._normalize_date(row.get(key))
            if norm:
                return norm
        return None

    def get_earnings_dates(self, symbol: str) -> List[str]:
        """Upcoming earnings dates (YYYY-MM-DD) for a symbol.

        Uses the beta ``markets/fundamentals/calendars`` endpoint and keeps only
        earnings-type events from each result's ``corporate_calendars`` table.
        (The old ``markets/calendar`` endpoint is Tradier's market *holiday*
        calendar and never contained earnings, so the warning was always "NO".)

        Raises TradierAPIError on transport/auth failure so the scanner can
        distinguish "data unavailable" from "no earnings in window".
        """
        # Tradier's corporate_calendars event_type codes for quarterly earnings:
        # 7-10 are the results/releases, 12-15 the earnings conference calls.
        # (Other codes are AGMs=1, conferences=20, annual report=30, etc.) Match
        # on the code first; some terse older rows ("Q3 2010", "reports first
        # quarter results") carry an earnings code but no "earnings" in the text.
        EARNINGS_EVENT_TYPES = {7, 8, 9, 10, 12, 13, 14, 15}

        def _is_earnings(row: Dict[str, Any]) -> bool:
            if row.get("event_type") in EARNINGS_EVENT_TYPES:
                return True
            return "earning" in str(row.get("event", "")).lower()

        return self._fundamentals_dates(
            symbol,
            "markets/fundamentals/calendars",
            "corporate_calendars",
            date_keys=("begin_date_time", "begin_date", "date",
                       "estimated_date_for_next_event"),
            event_filter=_is_earnings,
        )

    def get_dividend_dates(self, symbol: str) -> List[str]:
        """Upcoming ex-dividend dates (YYYY-MM-DD) for a symbol.

        Uses the beta ``markets/fundamentals/dividends`` endpoint and pulls
        ex-dividend dates from the ``cash_dividends`` table. Raises
        TradierAPIError on transport/auth failure (see get_earnings_dates).
        """
        return self._fundamentals_dates(
            symbol,
            "markets/fundamentals/dividends",
            "cash_dividends",
            date_keys=("ex_date", "ex_dividend_date", "date"),
        )

    def get_historical_prices(self, symbol: str, interval: str = "daily", span: str = "month") -> List[Dict[str, Any]]:
        """Fetch historical price data for technical indicators (e.g., 50-day MA)."""
        endpoint = "markets/history"
        params = {
            "symbol": symbol,
            "interval": interval,
            "span": span
        }
        data = self._request(endpoint, params)

        try:
            return data.get('history', {}).get('day', [])
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not fetch history for {symbol}: {e}")
            return []
