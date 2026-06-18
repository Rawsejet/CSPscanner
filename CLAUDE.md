# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- **Run Dashboard**: `streamlit run app.py`
- **Run All Tests**: `pytest`
- **Run Single Test**: `pytest tests/test_file.py`
- **Install Dependencies**: `pip install -r requirements.txt`

## Architecture and Structure
The project is a Tradier-integrated CSP (Cash Secured Put) signal scanner.

- **`app.py`**: The Streamlit frontend. Handles user input (Cash, Max Spread, Min IV Rank, Environment), result caching (`st.cache_data`, 5-min TTL keyed on token/env/horizon/spread/IV rank), the Heat Map UI, the Scan Diagnostics panel, and the Portfolio Simulation. The cached `run_scan()` strips the diagnostics object off `df.attrs` and returns `(df, summary_dict)` so results pickle cleanly.
- **`tradier_client.py`**: Low-level API wrapper for Tradier. Manages authentication, 429 retry/backoff, and raw data retrieval (Spot prices, Option chains, Greeks, Earnings/Dividends, History). `get_spot_price` falls back `last → close → prevclose → bid/ask mid`.
- **`scanner.py`**: The core business logic. Implements the dual-scan cycles (Income: 7-14 days, Classic: 30-45 days) and filters contracts on Delta, Liquidity (spread), OTM, ROC, and IV Rank. The scan fans tickers out across a bounded `ThreadPoolExecutor`; each `_scan_ticker` returns its own `ScanDiagnostics` which are merged. Auth errors (401/403) re-raise to abort the whole scan; other API errors are recorded per-ticker. `_technicals()` fetches one year of history once and derives both the 50-day MA and the IV Rank.
- **`utils.py`**: Shared calculations — Annualized Return on Capital, `calculate_iv_rank` (realized-volatility proxy, see below), and date utilities.
- **`tests/`**: Unit tests for signal logic and mocked API tests to ensure filtering accuracy without consuming API credits.

## Key Constraints
- **API Rate Limiting**: The scan runs concurrently but throughput is capped by `ThreadPoolExecutor(max_workers=8)`; bursts that hit the limit are retried with exponential backoff on HTTP 429. Caching (5-min TTL) avoids re-fetching on repeated runs. Do **not** reintroduce per-call `sleep` stagger — bounded concurrency is the chosen mechanism.
- **Risk Profile**: Focus is strictly on Low-to-Medium risk (Delta -0.10 to -0.25).
- **IV Rank is a proxy**: Tradier exposes no implied-volatility history, so IV Rank is approximated from 1-year **realized** volatility. Tickers with insufficient history are not filtered on IV Rank.
