import math
from datetime import datetime, timedelta

def calculate_iv_rank(closes, window=20, trading_days=252):
    """
    Approximate IV Rank using realized (historical) volatility.

    Tradier exposes no implied-volatility history, so this ranks the most recent
    `window`-day annualized realized volatility within its range over the supplied
    price history (typically one year of daily closes). Returns a 0-100 value, or
    None when there is insufficient data or no volatility dispersion to rank.

    NOTE: this is a realized-vol proxy, not true IV Rank.
    """
    if not closes or len(closes) < window + 2:
        return None

    # Daily log returns (skip non-positive prices defensively)
    returns = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        if prev and cur and prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
    if len(returns) < window:
        return None

    # Rolling annualized realized-volatility series
    rv_series = []
    annualize = math.sqrt(trading_days)
    for i in range(window, len(returns) + 1):
        win = returns[i - window:i]
        mean = sum(win) / window
        var = sum((r - mean) ** 2 for r in win) / (window - 1)
        rv_series.append(math.sqrt(var) * annualize)

    rv_now = rv_series[-1]
    rv_min, rv_max = min(rv_series), max(rv_series)
    if rv_max == rv_min:
        return None

    return (rv_now - rv_min) / (rv_max - rv_min) * 100

def calculate_annualized_roc(premium, strike, days_to_expiry):
    """
    Calculates the Annualized Return on Capital (ROC) for a CSP.
    ROC = (Premium / Strike) * (365 / DTE)
    """
    if days_to_expiry <= 0:
        return 0.0

    roc = (premium / strike) * (365 / days_to_expiry)
    return roc * 100  # Return as percentage

def get_date_range(start_days, end_days):
    """
    Returns a tuple of (start_date, end_date) relative to today.
    """
    today = datetime.now().date()
    start_date = today + timedelta(days=start_days)
    end_date = today + timedelta(days=end_days)
    return start_date, end_date

def is_date_between(target_date, start_date, end_date):
    """
    Checks if a given date falls between two dates (inclusive).
    """
    return start_date <= target_date <= end_date
