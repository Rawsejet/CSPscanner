import streamlit as st
import pandas as pd
from tradier_client import TradierClient, TradierAPIError
from scanner import CSPScanner
from iron_condor import ICScanner
from positions import load_positions, add_position, remove_position, evaluate_position
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Options Income Dashboard", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def run_scan(api_key: str, use_sandbox: bool, horizon_key: str,
             max_spread_pct: float, min_iv_rank: float, exclude_earnings: bool,
             max_capital: float):
    """Run a CSP scan, caching results for 5 minutes to avoid hammering the API.

    Cached on (token, env, horizon, spread, IV rank, exclude-earnings, capital):
    re-clicking Run with the same settings is instant and costs no API credits.
    Returns (DataFrame, diagnostics summary dict); the custom diagnostics object
    is stripped so the result caches cleanly.
    """
    client = TradierClient(api_key, use_sandbox=use_sandbox)
    scanner = CSPScanner(client)
    df = scanner.scan(
        horizon=horizon_key,
        max_spread_pct=max_spread_pct,
        min_iv_rank=min_iv_rank,
        exclude_earnings=exclude_earnings,
        max_capital=max_capital,
    )
    diag = df.attrs.pop("diagnostics", None)
    return df, (diag.summary() if diag else None)


@st.cache_data(ttl=300, show_spinner=False)
def run_ic_scan(api_key: str, use_sandbox: bool, short_delta: float,
                wing_mode: str, wing_value: float, dte_min: int, dte_max: int,
                exclude_earnings: bool, min_natural_credit):
    """Run a weekly iron-condor scan, cached for 5 minutes like the CSP scan."""
    client = TradierClient(api_key, use_sandbox=use_sandbox)
    scanner = ICScanner(client)
    df = scanner.scan(
        short_delta=short_delta,
        wing_mode=wing_mode,
        wing_value=wing_value,
        dte_min=dte_min,
        dte_max=dte_max,
        exclude_earnings=exclude_earnings,
        min_natural_credit=min_natural_credit,
    )
    diag = df.attrs.pop("diagnostics", None)
    return df, (diag.summary() if diag else None)


@st.cache_data(ttl=60, show_spinner=False)
def evaluate_positions_cached(api_key: str, use_sandbox: bool, positions_key: str):
    """Live status for all tracked condors (cached 60s; the Refresh button clears it).

    ``positions_key`` is the positions list JSON-encoded so it is hashable for the
    cache — avoids re-pulling chains on every unrelated Streamlit rerun.
    """
    pos_list = json.loads(positions_key)
    client = TradierClient(api_key, use_sandbox=use_sandbox)
    return [evaluate_position(client, p) for p in pos_list]


# Sidebar Configuration (shared auth + CSP settings)
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
    "Available Cash ($)", min_value=1000, value=10000, step=1000,
    help=(
        "Your buying power. For CSPs, any put whose collateral (strike × 100) "
        "exceeds this is hidden. For iron condors it sizes how many contracts you "
        "could carry against each trade's max loss."
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption("**CSP Scanner settings** (Iron Condor settings are on its tab)")
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
exclude_earnings = st.sidebar.checkbox(
    "Exclude earnings inside expiry window",
    value=True,
    help=(
        "Drop contracts whose expiry window contains a known earnings date — "
        "binary event risk that can dwarf the premium, especially on short "
        "(Income) horizons. Tickers whose earnings data is unavailable are kept "
        "and shown as N/A (an empty calendar is never treated as 'no earnings')."
    ),
)

# Display-only ranking metric. Annualized ROC is mathematically inflated for
# short expiries (it assumes continuous redeployment), so offer fairer lenses.
RANK_COLUMNS = {
    "Annualized ROC": "ROC",
    "Return % (held to expiry)": "Return %",
    "$/Day per contract": "$/Day",
    "Downside cushion %": "Cushion %",
}
rank_by = st.sidebar.selectbox(
    "Rank signals by",
    list(RANK_COLUMNS),
    index=0,
    help=(
        "Orders the heat map and picks the 'top' signal. Annualized ROC favors "
        "the shortest expiries; Return % and $/Day compare horizons more "
        "honestly, and Cushion % ranks safety first. Click Run to apply."
    ),
)

st.title("Options Income Dashboard")

if not api_key:
    st.warning("Please enter your Tradier API Token in the sidebar to start scanning.")
    st.stop()

tab_csp, tab_ic = st.tabs(["CSP Scanner", "Iron Condors (weekly)"])

with tab_csp:
    st.markdown("Scanning the S&P 100 + Nasdaq-100 for Low-to-Medium risk Cash Secured Puts.")

    # Horizon Selection
    horizon = st.radio(
        "Select Scan Horizon",
        ["Income (7-14 Days)", "Classic (30-45 Days)"],
        horizontal=True,
    )
    horizon_key = "Income" if "Income" in horizon else "Classic"

    if st.button("Run Scanner"):
        with st.spinner(f"Scanning for {horizon_key} signals..."):
            try:
                df, diag = run_scan(api_key, use_sandbox, horizon_key, max_spread / 100, min_iv_rank, exclude_earnings, available_cash)
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
                        "Earnings in Window": s["rejected_earnings"],
                        "Exceeds Capital": s["rejected_capital"],
                    }
                    st.write("**Rejection breakdown:**")
                    rej_df = pd.DataFrame(list(rejections.items()), columns=["Reason", "Count"])
                    st.dataframe(rej_df, use_container_width=True, hide_index=True)

                    if s["fundamentals_unavailable"]:
                        st.caption(
                            f"⚠️ Earnings/dividend data was unavailable for "
                            f"{s['fundamentals_unavailable']} ticker(s) — their contracts "
                            "show **N/A** and can't be excluded by the earnings filter. "
                            "The beta fundamentals endpoints require an entitled "
                            "Tradier plan; verify access if you expect this data."
                        )

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
                # Re-rank by the chosen metric (sort_values returns a new frame, so
                # the cached scan result is never mutated). Drives the heat map, the
                # Portfolio Simulation top-N, and the "top signal" risk analysis.
                df = df.sort_values(RANK_COLUMNS[rank_by], ascending=False).reset_index(drop=True)

                st.subheader(f"Identified Signals ({len(df)})")

                # Heat Map Table — ranked by the selected metric
                st.markdown(f"### Signal Heat Map · ranked by {rank_by}")
                st.caption(
                    "**ROC** is *annualized* — it assumes you redeploy this capital at "
                    "the same rate all year, which a 7–14 day trade rarely does without "
                    "a losing cycle. **Return %** is what you actually keep if held to "
                    "expiry; **$/Day** is premium per contract per day (a fair "
                    "income-rate comparison across horizons)."
                )

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
                    capital_deployed = max_contracts * strike * 100
                    breakeven = row["Break-even"]
                    cushion = row["Cushion %"]

                    with cols[i % max_cols]:
                        st.metric(
                            f"{row['Ticker']} (${strike:.0f}P)",
                            f"{max_contracts} Contracts",
                            f"Est. Premium: ${total_premium:.2f}",
                        )
                        cushion_txt = f" ({cushion:.1f}% below spot)" if pd.notna(cushion) else ""
                        st.caption(
                            f"Capital tied up: ${capital_deployed:,.0f} · "
                            f"Break-even ${breakeven:.2f}{cushion_txt}"
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

                cushion = top["Cushion %"]
                rcols = st.columns(3)
                rcols[0].metric("Break-even", f"${top['Break-even']:.2f}")
                rcols[1].metric(
                    "Downside cushion",
                    f"{cushion:.1f}%" if pd.notna(cushion) else "N/A",
                    help="How far the stock can fall before this trade loses money at expiry.",
                )
                rcols[2].metric("Capital / contract", f"${top['Capital']:,.0f}")

                prob_assignment = abs(top["Delta"]) * 100
                st.write(f"**Estimated Probability of Assignment:** {prob_assignment:.1f}%")
                st.progress(prob_assignment / 100)

                st.info(
                    f"This trade has a {100 - prob_assignment:.1f}% probability of expiring worthless, "
                    f"yielding ${top['Premium']:.2f} per share."
                )

    else:
        st.info("Click 'Run Scanner' to start searching for signals.")


with tab_ic:
    ic_find, ic_track = st.tabs(["Find Entries", "My Condors"])

with ic_find:
    st.markdown("### Weekly Iron Condor — entry finder")
    st.caption(
        "Sells far-OTM weekly condors ranked by **real edge** — the variance risk "
        "premium (implied vs realized vol) and whether the 4-leg credit is actually "
        "fillable — not by headline premium. Negative-edge setups are shown in "
        "**red**, never hidden."
    )

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic_short_delta = ic1.slider(
        "Short delta (×100)", min_value=3, max_value=20, value=10,
        help="Target |delta| for both short strikes. Lower = further OTM (higher win rate, less credit).",
    )
    wing_mode_label = ic2.selectbox(
        "Wing sizing", ["% of spot", "Fixed $"],
        help="% of spot scales the wing with the stock's price (a flat $ wing is tiny on a $1,200 name, huge on an $80 one). Fixed $ keeps a set dollar wing.",
    )
    if wing_mode_label == "% of spot":
        ic_wing_mode = "pct"
        ic_wing_value = ic3.slider(
            "Wing (% of spot)", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help="Wing width as a percent of each ticker's price, snapped to listed strikes. Sets max loss = (wing − credit) × 100.",
        )
    else:
        ic_wing_mode = "fixed"
        ic_wing_value = ic3.number_input(
            "Wing width ($)", min_value=0.5, value=20.0, step=0.5,
            help="Flat dollar wing on every ticker (e.g. your MU/SNDK $20). Snapped to listed strikes.",
        )
    ic_dte = ic4.slider(
        "DTE window", min_value=1, max_value=14, value=(2, 12),
        help="Weekly expiry window; the nearest expiry inside it is used.",
    )

    ic5, ic6 = st.columns(2)
    ic_excl_earn = ic5.checkbox(
        "Exclude earnings week", value=True,
        help="Drop tickers with a known earnings date inside the weekly window (binary gap risk).",
    )
    ic_only_fillable = ic6.checkbox(
        "Only show fillable (natural credit > 0)", value=False,
        help="Hide condors you'd have to pay to enter at market. Off = show them, flagged red.",
    )

    if st.button("Scan Iron Condors"):
        with st.spinner("Scanning weekly iron condors..."):
            try:
                ic_df, ic_diag = run_ic_scan(
                    api_key, use_sandbox, ic_short_delta / 100,
                    ic_wing_mode, float(ic_wing_value),
                    ic_dte[0], ic_dte[1], ic_excl_earn,
                    0.0 if ic_only_fillable else None,
                )
            except TradierAPIError as e:
                st.error(f"API Error: {e}")
                if e.status_code in (401, 403):
                    st.error(
                        "Authentication failed — check your token and the **Environment** "
                        "selector (Sandbox vs Production)."
                    )
                st.stop()

        if ic_diag:
            with st.expander("Scan Diagnostics"):
                s = ic_diag
                cols = st.columns(4)
                cols[0].metric("Tickers Scanned", s["tickers_scanned"])
                cols[1].metric("Condors Built", s["condors_built"])
                cols[2].metric("Negative-VRP Flags", s["flagged_negative_vrp"])
                cols[3].metric("API Errors", s["api_errors"])

                breakdown = {
                    "No condor built (no strikes)": s["rejected_no_condor"],
                    "No option chain": s["rejected_no_chain"],
                    "Earnings in window (excluded)": s["rejected_earnings"],
                    "Thin credit (filtered out)": s["rejected_thin_credit"],
                    "Flagged: negative edge gap": s["flagged_negative_edge"],
                    "Flagged: unfillable credit": s["flagged_thin_credit"],
                }
                st.write("**Build / flag breakdown:**")
                st.dataframe(
                    pd.DataFrame(list(breakdown.items()), columns=["Reason", "Count"]),
                    use_container_width=True, hide_index=True,
                )
                if s["fundamentals_unavailable"]:
                    st.caption(
                        f"⚠️ Earnings data was unavailable for {s['fundamentals_unavailable']} "
                        "ticker(s) — their condors show **N/A** and can't be earnings-excluded."
                    )

        if ic_df.empty:
            st.info(
                "No condors could be built. Try a wider DTE window, a different short "
                "delta, or check that the chosen tickers have weekly options."
            )
        else:
            IC_RANK = {
                "Edge (VRP, then edge gap)": ["VRP", "Edge (pp)"],
                "VRP (IV vs realized)": ["VRP"],
                "Return on risk": ["RoR %"],
                "Probability of profit": ["PoP %"],
                "Credit (natural)": ["Credit (nat)"],
            }
            rank_choice = st.selectbox("Rank condors by", list(IC_RANK), index=0)
            ic_df = ic_df.sort_values(IC_RANK[rank_choice], ascending=False).reset_index(drop=True)

            greens = int((ic_df["Grade"] == "Green").sum())
            st.subheader(f"{len(ic_df)} condors · {greens} graded green")
            if greens == 0:
                st.warning(
                    "**No green-grade condors.** Implied vol is at or below realized vol "
                    "across the scanned names, so there's no variance-risk-premium edge to "
                    "sell right now. That's the scanner doing its job — wait for a regime "
                    "where options are over-priced rather than selling cheap premium."
                )

            # How many contracts your cash could carry against each max loss.
            ic_df = ic_df.copy()
            ic_df["Max Contracts"] = ic_df["Max Loss"].apply(
                lambda ml: int(available_cash // ml) if isinstance(ml, (int, float)) and ml > 0 else 0
            )

            st.caption(
                "**Grade**: green = positive VRP & fillable · amber = marginal · red = negative "
                "VRP or you'd pay to enter. **VRP** = implied ÷ realized vol (>1 = paid to sell). "
                "**Edge (pp)** = PoP − breakeven win rate (negative is normal far-OTM; VRP is what "
                "makes it pay). **Credit (nat)** is the at-market fill — 4-leg slippage can exceed "
                "the mid credit. **EM x** = expected-moves from spot to each short. **Max Loss** = "
                "(wing − credit) × 100; the wide body never adds to it."
            )

            def color_grade(val):
                colors = {"Green": "#d4edda", "Amber": "#fff3cd", "Red": "#f8d7da"}
                return f"background-color: {colors.get(val, '#ffffff')}"

            def color_vrp(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 1.1:
                    return "color: #155724; font-weight: bold"
                if val >= 1.0:
                    return "color: #856404"
                return "color: #721c24; font-weight: bold"

            st.dataframe(
                ic_df.style.map(color_grade, subset=["Grade"]).map(color_vrp, subset=["VRP"]),
                use_container_width=True,
            )
            st.caption(
                "Track any of these once you've opened them in the **My Condors** tab for "
                "live P&L and management triggers."
            )
    else:
        st.info("Set your parameters and click 'Scan Iron Condors'.")


with ic_track:
    st.markdown("### My Condors — live tracking")
    st.caption(
        "Enter the condors you've opened; each refresh re-pulls the live chain and "
        "flags the triggers you use: take profit at 50% of credit, defend the tested "
        "side at Δ≈0.30, and close into the last day(s) to dodge gamma. Stored locally "
        "in ic_positions.json (gitignored)."
    )

    with st.expander("➕ Add an open condor"):
        with st.form("add_condor", clear_on_submit=True):
            f1, f2, f3 = st.columns(3)
            add_ticker = f1.text_input("Ticker").strip().upper()
            add_expiry = f2.text_input("Expiry (YYYY-MM-DD)").strip()
            add_qty = f3.number_input("Contracts", min_value=1, value=1, step=1)
            g1, g2, g3, g4 = st.columns(4)
            add_lp = g1.number_input("Long put", min_value=0.0, value=0.0, step=0.5)
            add_sp = g2.number_input("Short put", min_value=0.0, value=0.0, step=0.5)
            add_sc = g3.number_input("Short call", min_value=0.0, value=0.0, step=0.5)
            add_lc = g4.number_input("Long call", min_value=0.0, value=0.0, step=0.5)
            add_credit = st.number_input(
                "Entry credit received (per share)", min_value=0.0, value=0.0, step=0.05)
            if st.form_submit_button("Add position"):
                if add_ticker and add_expiry and add_lp < add_sp < add_sc < add_lc and add_credit > 0:
                    add_position({
                        "ticker": add_ticker, "expiry": add_expiry, "quantity": int(add_qty),
                        "long_put_strike": add_lp, "short_put_strike": add_sp,
                        "short_call_strike": add_sc, "long_call_strike": add_lc,
                        "entry_credit": float(add_credit),
                    })
                    st.success(f"Added {add_ticker} {add_sp:g}/{add_sc:g} condor.")
                else:
                    st.error(
                        "Need a ticker, expiry, a valid ladder "
                        "(long put < short put < short call < long call), and a positive credit."
                    )

    pos_list = load_positions()
    if not pos_list:
        st.info("No open condors tracked yet — add one above.")
    else:
        if st.button("🔄 Refresh live status"):
            evaluate_positions_cached.clear()
        statuses = evaluate_positions_cached(
            api_key, use_sandbox, json.dumps(pos_list, sort_keys=True))

        for status, pos in zip(statuses, pos_list):
            ladder = (f"{pos['long_put_strike']:g}/{pos['short_put_strike']:g}.."
                      f"{pos['short_call_strike']:g}/{pos['long_call_strike']:g}")
            if status.get("error"):
                st.warning(f"**{pos['ticker']}** {pos['expiry']} ({ladder}): {status['error']}")
                if st.button("Remove", key=f"rm_{pos['id']}"):
                    remove_position(pos["id"])
                    st.rerun()
                st.divider()
                continue

            qty = status["quantity"]
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.markdown(f"**{status['ticker']}** · {ladder} · {status['dte']}DTE · {qty}x")
            pl, plp = status["pl_dollars"], status["pl_pct"]
            c2.metric(
                "Open P&L", f"${pl:,.0f}" if pl is not None else "—",
                f"{plp:+.0f}%" if plp is not None else None,
            )
            c3.metric(
                "Buy-back (mid)", f"${status['value_mid'] * 100 * qty:,.0f}",
                help=f"Debit to close now. At-market (natural): "
                     f"${status['value_natural'] * 100 * qty:,.0f}.",
            )
            dp, dc = status["short_put_delta"], status["short_call_delta"]
            c4.metric(
                "Short Δ p/c",
                f"{dp:.2f}/{dc:.2f}" if (dp is not None and dc is not None) else "—",
            )

            if status["triggers"]:
                for tmsg in status["triggers"]:
                    st.warning("⚠️ " + tmsg)
            else:
                st.caption("No active triggers — holding.")
            if st.button("Remove", key=f"rm_{pos['id']}"):
                remove_position(pos["id"])
                st.rerun()
            st.divider()
