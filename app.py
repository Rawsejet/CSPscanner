import streamlit as st
import pandas as pd
from tradier_client import TradierClient, TradierAPIError
from scanner import CSPScanner
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="S&P 100 CSP Signal Dashboard", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def run_scan(api_key: str, use_sandbox: bool, horizon_key: str,
             max_spread_pct: float, min_iv_rank: float):
    """Run a scan, caching results for 5 minutes to avoid hammering the API.

    Cached on (token, env, horizon, spread, IV rank): re-clicking Run with the
    same settings is instant and costs no API credits. Returns (DataFrame,
    diagnostics summary dict); the custom diagnostics object is stripped so the
    result caches cleanly.
    """
    client = TradierClient(api_key, use_sandbox=use_sandbox)
    scanner = CSPScanner(client)
    df = scanner.scan(
        horizon=horizon_key,
        max_spread_pct=max_spread_pct,
        min_iv_rank=min_iv_rank,
    )
    diag = df.attrs.pop("diagnostics", None)
    return df, (diag.summary() if diag else None)

# Sidebar Configuration
st.sidebar.header("Settings")

env = st.sidebar.radio(
    "Environment",
    ["Production", "Sandbox"],
    help="Sandbox for testing with paper accounts; Production for live data.",
)
use_sandbox = env == "Sandbox"

api_key = st.sidebar.text_input(
    "Tradier API Token",
    value=os.environ.get("TRADIER_API_KEY", ""),
    type="password",
)
available_cash = st.sidebar.number_input(
    "Available Cash ($)", min_value=1000, value=10000, step=1000
)
max_spread = st.sidebar.slider(
    "Max Bid-Ask Spread (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Max spread as % of mid-price. Wider = more contracts pass, but potentially less liquid.",
)
min_iv_rank = st.sidebar.slider(
    "Min IV Rank",
    min_value=0,
    max_value=50,
    value=20,
    step=1,
    help=(
        "Minimum IV Rank (0-100). Higher = only richer premium passes. "
        "Approximated from 1-year realized volatility (Tradier has no IV history); "
        "tickers without enough history are not filtered."
    ),
)

st.title("S&P 100 CSP Signal Dashboard")
st.markdown("Scanning for Low-to-Medium risk Cash Secured Put opportunities.")

if not api_key:
    st.warning("Please enter your Tradier API Token in the sidebar to start scanning.")
    st.stop()

# Horizon Selection
horizon = st.radio(
    "Select Scan Horizon",
    ["Income (7-14 Days)", "Classic (30-45 Days)"],
    horizontal=True,
)
horizon_key = "Income" if "Income" in horizon else "Classic"

if st.button("Run Scanner"):
    with st.spinner(f"Scanning S&P 100 for {horizon_key} signals..."):
        try:
            df, diag = run_scan(api_key, use_sandbox, horizon_key, max_spread / 100, min_iv_rank)
        except TradierAPIError as e:
            st.error(f"API Error: {e}")
            if e.status_code in (401, 403):
                st.error(
                    "Authentication failed. Most common causes:\n"
                    "1. You selected **Sandbox** but your token is **Production** (or vice-versa).\n"
                    "2. The token has expired or been revoked.\n\n"
                    "Switch the **Environment** selector in the sidebar and try again."
                )
            st.stop()

        # Diagnostics panel (always shown)
        if diag:
            with st.expander("Scan Diagnostics"):
                s = diag
                cols = st.columns(4)
                cols[0].metric("Tickers Scanned", s["tickers_scanned"])
                cols[1].metric("Chains Fetched", s["chains_fetched"])
                cols[2].metric("Contracts Evaluated", s["contracts_evaluated"])
                cols[3].metric("API Errors", s["api_errors"])

                rejections = {
                    "No Greeks": s["rejected_no_greeks"],
                    "Missing Fields": s["rejected_missing_fields"],
                    "Delta Out of Range": s["rejected_delta"],
                    "Zero Bid": s["rejected_zero_bid"],
                    "Wide Spread": s["rejected_spread"],
                    "Low ROC": s["rejected_roc"],
                    "Low IV Rank": s["rejected_iv_rank"],
                    "Not OTM": s["rejected_otm"],
                }
                st.write("**Rejection breakdown:**")
                rej_df = pd.DataFrame(list(rejections.items()), columns=["Reason", "Count"])
                st.dataframe(rej_df, use_container_width=True, hide_index=True)

        if df.empty:
            st.info("No signals found matching the risk criteria.")
            if diag:
                s = diag
                if s["tickers_scanned"] == 0:
                    st.warning(
                        "No tickers were successfully scanned. "
                        "Check your API token and environment setting (Sandbox vs Production)."
                    )
                elif s["chains_fetched"] == 0:
                    st.info(
                        "Tickers were scanned but no expirations matched the selected horizon window. "
                        "Try the other horizon (Income vs Classic)."
                    )
        else:
            st.subheader(f"Identified Signals ({len(df)})")

            # Heat Map Table — sorted by ROC descending
            st.markdown("### Signal Heat Map")

            def color_strength(val):
                colors = {"High": "#d4edda", "Medium": "#fff3cd", "Low": "#f8d7da"}
                return f"background-color: {colors.get(val, '#ffffff')}"

            def color_roc(val):
                if val >= 25:
                    return "color: #155724; font-weight: bold"
                elif val >= 15:
                    return "color: #856404; font-weight: bold"
                return "color: #721c24"

            st.dataframe(
                df.style.map(color_strength, subset=["Signal Strength"])
                .map(color_roc, subset=["ROC"]),
                use_container_width=True,
            )

            # Portfolio Simulation — show the top signals, capped at 4 columns
            st.markdown("### Portfolio Simulation")
            MAX_SIMS = 8
            sim_df = df.head(MAX_SIMS)
            if len(df) > MAX_SIMS:
                st.caption(f"Showing top {MAX_SIMS} of {len(df)} signals by ROC.")
            max_cols = min(len(sim_df), 4)
            cols = st.columns(max_cols)
            for i, (_, row) in enumerate(sim_df.iterrows()):
                strike = row["Strike"]
                premium = row["Premium"]
                max_contracts = int(available_cash // (strike * 100))
                total_premium = max_contracts * premium * 100

                with cols[i % max_cols]:
                    st.metric(
                        f"{row['Ticker']} (${strike:.0f}P)",
                        f"{max_contracts} Contracts",
                        f"Est. Premium: ${total_premium:.2f}",
                    )

                    # Quick link to the Tradier dashboard (sandbox accounts use
                    # the same dashboard). No reliable per-symbol deep link exists.
                    st.link_button(
                        f"Open {row['Ticker']} in Tradier",
                        "https://dash.tradier.com/",
                    )

            # Risk Visualizer for the top signal
            st.markdown("---")
            top = df.iloc[0]
            st.subheader(f"Risk Analysis: {top['Ticker']} @ ${top['Strike']:.0f} Strike")

            prob_assignment = abs(top["Delta"]) * 100
            st.write(f"**Estimated Probability of Assignment:** {prob_assignment:.1f}%")
            st.progress(prob_assignment / 100)

            st.info(
                f"This trade has a {100 - prob_assignment:.1f}% probability of expiring worthless, "
                f"yielding ${top['Premium']:.2f} per share."
            )

else:
    st.info("Click 'Run Scanner' to start searching for signals.")
