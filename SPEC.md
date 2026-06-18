# Spec: S&P 100 CSP Signal Dashboard

## 1. System Overview
A Streamlit-based dashboard that interfaces with the Tradier API to scan the top 100 S&P tickers. It identifies Cash Secured Put (CSP) opportunities across two distinct time horizons, filtering for low-to-medium risk.

## 2. Scanning Parameters
### A. The Universe
*   **Tickers**: Top 100 S&P 500 stocks by market cap.
*   **Risk Profile**: Low to Medium.
    *   **Target Delta**: $-0.10$ (Low Risk) to $-0.25$ (Medium Risk).
    *   **Probability of Profit (PoP)**: $\approx 75\% \text{ to } 90\%$.

### B. Time-Frame Cycles (Dual-Scan)
The scanner will run two parallel logic paths for every ticker:
1.  **The Income Cycle (Short-Term)**:
    *   **Expiry**: 7 to 14 days.
    *   **Goal**: Rapid theta decay, high turnover.
2.  **The Classic Cycle (Medium-Term)**:
    *   **Expiry**: 30 to 45 days.
    *   **Goal**: Higher absolute premium, more time for the trade to "work" if the stock dips.

## 3. Signal Logic (The "Buy" Criteria)
A "Good Buy Signal" is triggered if a contract meets **all** the following:
*   **Liquidity**: Bid-Ask spread is within a configurable threshold of the mid-price (default $\le 15\%$, adjustable 5–30% in the sidebar).
*   **Yield**: Annualized Return on Capital (ROC) $\ge 12\%$.
*   **Technical Alignment**: Strike price is below the 50-day Moving Average (preferable) or at least below current spot.
*   **Volatility**: IV Rank $> 20\%$ (ensures you aren't selling "cheap" puts). Configurable in the sidebar (0–50). Tradier exposes no IV history, so IV Rank is approximated from 1-year realized volatility; tickers without enough history are not filtered.

## 4. Dashboard Features (UI/UX)
*   **The "Heat Map" Table**: A sortable grid showing:
    *   `Ticker` | `Cycle (Short/Classic)` | `Strike` | `Days to Expiry` | `Premium` | `Annualized %` | `Delta` | `Signal Strength (Low/Med/High)`.
*   **Risk Visualizer**: A simple gauge for each signal showing the "Probability of Assignment."
*   **Portfolio Simulator**: A sidebar where you enter your "Available Cash," and the dashboard calculates exactly how many contracts you can sell for a given signal without over-leveraging.
*   **Quick-Links**: Buttons to open the Tradier trading page for that specific ticker.

## 5. Technical Stack
*   **Language**: Python 3.12+
*   **Frontend**: Streamlit (for rapid deployment and interactive tables).
*   **API**: Tradier SDK / REST API.
*   **Data Processing**: Pandas (for calculating ROC and filtering deltas).
*   **Caching**: `st.cache_data` to prevent hitting Tradier API rate limits during session refreshes.

## 6. Edge Cases & Safety Checks
*   **Earnings Warning**: The dashboard must flag any ticker with an earnings date falling *inside* the chosen expiry window (earnings = high risk of assignment).
*   **Dividend Warning**: Flag upcoming ex-dividend dates that could pull the stock price down.
*   **API Rate Limiting**: Implement a staggered fetch (sleep) to avoid being blocked by Tradier.
