# S&P 100 CSP Signal Dashboard

A Streamlit dashboard that scans the S&P 100 via the [Tradier](https://tradier.com/) API for
**Cash Secured Put (CSP)** opportunities in the low-to-medium risk band, across two time horizons.

## Features

- **Dual-scan horizons** — *Income* (7–14 DTE) and *Classic* (30–45 DTE).
- **Signal filters** — every contract must clear all of:
  - **Delta** between −0.25 and −0.10 (≈ 75–90% probability of profit).
  - **Liquidity** — bid-ask spread within a configurable % of the mid (default 15%).
  - **OTM** — strike below spot.
  - **Yield** — annualized Return on Capital ≥ 12%.
  - **IV Rank** — above a configurable floor (default 20) so you aren't selling "cheap" premium.
- **Heat Map table** — signals ranked by ROC, color-coded by ROC and Signal Strength.
- **Earnings & dividend warnings** — flags events falling inside the expiry window.
- **Portfolio Simulation** — sizes contracts against your available cash for the top signals.
- **Risk Visualizer** — estimated probability of assignment for the top signal.
- **Scan Diagnostics** — transparency panel showing how many contracts were rejected and why.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, choose **Production** or **Sandbox**, paste your Tradier API token, and click
**Run Scanner**. You can also set the token via a `.env` file (loaded automatically):

```
TRADIER_API_KEY=your_token_here
```

> The **Environment** selector must match your token type — a sandbox token sent to production
> (or vice-versa) returns a 401/403. The app surfaces this with a specific hint.

## Configuration (sidebar)

| Control | Default | Notes |
|---|---|---|
| Environment | Production | Sandbox uses paper-account endpoints. |
| Tradier API Token | — | Or set `TRADIER_API_KEY` in `.env`. |
| Available Cash ($) | 10,000 | Drives the Portfolio Simulation. |
| Max Bid-Ask Spread (%) | 15 | Liquidity filter; 5–30%. |
| Min IV Rank | 20 | Volatility filter; 0–50 (see note below). |

## How it works

- **`app.py`** — Streamlit UI. Scans are cached for 5 minutes (`st.cache_data`) keyed on
  token/environment/horizon/spread/IV-rank, so re-running with the same settings is instant and
  costs no API credits.
- **`tradier_client.py`** — thin Tradier REST wrapper with 429 retry/backoff and defensive
  parsing of Tradier's "single object vs list" response quirks.
- **`scanner.py`** — the filtering logic. Tickers are scanned concurrently with a bounded thread
  pool (`max_workers=8`); the per-ticker option-chain fetches and filters produce the signals plus
  a merged diagnostics summary.
- **`utils.py`** — ROC, IV Rank, and date helpers.

### A note on IV Rank

Tradier exposes no implied-volatility history, so true IV Rank isn't computable from the API.
This app approximates it from **1-year realized volatility**: it ranks the most recent 20-day
annualized realized vol within its trailing range (0–100). Tickers without enough history are
**not** filtered on IV Rank (they show `N/A`).

## Testing

```bash
pytest                       # all tests
pytest tests/test_scanner.py # a single file
```

Tests mock the Tradier client, so they validate the filter logic without consuming API credits.

## Tech stack

Python 3.12+ · Streamlit · pandas · requests · python-dotenv
