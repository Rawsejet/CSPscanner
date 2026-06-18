import requests
import time
import logging
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

    def __init__(self, access_token: str, use_sandbox: bool = False):
        self.access_token = access_token
        self.base_url = self.SANDBOX_BASE if use_sandbox else self.PRODUCTION_BASE
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, _retry: int = 0) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        try:
            # Concurrency is bounded by the scanner's thread pool; bursts that hit
            # the rate limit are handled by the 429 retry/backoff below.
            response = requests.get(url, headers=self.headers, params=params, timeout=15)

            # Retry on 429 with exponential backoff
            if response.status_code == 429 and _retry < 3:
                wait = float(response.headers.get("Retry-After", 2 ** _retry))
                logger.info(f"Rate limited on {endpoint}, sleeping {wait:.1f}s (retry {_retry + 1}/3)")
                time.sleep(wait)
                return self._request(endpoint, params, _retry + 1)

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

    def get_earnings_dates(self, symbol: str) -> List[str]:
        """
        Fetch upcoming earnings dates for a symbol.

        Uses Tradier fundamentals corporate-calendar endpoint.
        Returns a list of date strings (YYYY-MM-DD).
        """
        endpoint = "markets/calendar"
        params = {
            "type": "earnings",
            "symbol": symbol
        }
        data = self._request(endpoint, params)

        try:
            items = data.get('earnings', {}).get('earning', [])
            if isinstance(items, dict):
                items = [items]
            return [item.get('date', '') for item in items if item.get('date')]
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not fetch earnings dates for {symbol}: {e}")
            return []

    def get_dividend_dates(self, symbol: str) -> List[str]:
        """
        Fetch upcoming ex-dividend dates for a symbol.

        Uses Tradier fundamentals corporate-calendar endpoint.
        Returns a list of date strings (YYYY-MM-DD).
        """
        endpoint = "markets/calendar"
        params = {
            "type": "dividends",
            "symbol": symbol
        }
        data = self._request(endpoint, params)

        try:
            items = data.get('dividends', {}).get('dividend', [])
            if isinstance(items, dict):
                items = [items]
            return [item.get('date', '') for item in items if item.get('date')]
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not fetch dividend dates for {symbol}: {e}")
            return []

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
