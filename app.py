# pip install streamlit yfinance pandas numpy scipy plotly pytz

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import pytz
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


st.set_page_config(page_title="Commodity Dashboard", layout="wide")

# ------------------ Black-Scholes Greeks (with dividend yield) ------------------
def bs_greeks(S, K, T, r, sigma, opt_type='call', q=0.0):
    T = max(T, 1e-8)
    sigma = max(sigma, 1e-8)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Nd1 = norm.cdf(d1); Nd2 = norm.cdf(d2); nd1 = norm.pdf(d1)

    if opt_type == 'call':
        delta = np.exp(-q*T) * Nd1
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*Nd2 + q*S*np.exp(-q*T)*Nd1) / 365
        rho = (K*T*np.exp(-r*T)*Nd2) / 100
    else:
        delta = np.exp(-q*T) * (Nd1 - 1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho = -(K*T*np.exp(-r*T)*norm.cdf(-d2)) / 100

    gamma = (np.exp(-q*T)*nd1)/(S*sigma*np.sqrt(T))
    vega = (S*np.exp(-q*T)*nd1*np.sqrt(T))/100
    return delta, gamma, vega, theta, rho

# ------------------ Data fetch helpers ------------------
@st.cache_data(ttl=900)
def get_spot_price(ticker):
    """Return only the latest float price (safe for caching)."""
    t = yf.Ticker(ticker)
    try:
        price = float(t.fast_info.get("last_price", np.nan))
    except Exception:
        price = np.nan
    if np.isnan(price):
        hist = t.history(period="1d", auto_adjust=False)
        price = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
    return float(price)

@st.cache_data(ttl=900)
def get_history(ticker, period="1y", interval="1d"):
    return yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

@st.cache_data(ttl=900)
def get_options(ticker):
    t = yf.Ticker(ticker)
    return t.options or []

@st.cache_data(ttl=900)
def get_option_chain_df(ticker, expiry):
    """Return calls & puts as plain DataFrames (cache-safe)."""
    oc = yf.Ticker(ticker).option_chain(expiry)
    calls = oc.calls.reset_index(drop=True).copy()
    puts  = oc.puts.reset_index(drop=True).copy()
    return calls, puts

@st.cache_data(ttl=900)
def get_futures_curve(symbols):
    rows = []
    for sym in symbols:
        df = yf.download(sym, period="5d", interval="1d", progress=False)
        if df.empty:
            continue
        price = df["Close"].dropna().iloc[-1]
        rows.append({"contract": sym, "last": float(price)})
    return pd.DataFrame(rows)

# ------------------ UI ------------------
st.title("🛢️ Commodity Dashboard — Price, Curve, IV Smile & Surface, Greeks")

left, right = st.columns([1,1])

with left:
    st.subheader("1) Underlying & Proxies")
    commodity = st.selectbox(
        "Choose a commodity (spot proxy / front):",
        ["CL=F (WTI Crude)", "GC=F (Gold)", "SI=F (Silver)",
         "NG=F (Nat Gas)", "HG=F (Copper)", "ZC=F (Corn)", "ZW=F (Wheat)"],
        index=1
    )
    etf_default = {
        "CL=F (WTI Crude)": "USO",
        "GC=F (Gold)": "GLD",
        "SI=F (Silver)": "SLV",
        "NG=F (Nat Gas)": "UNG",
        "HG=F (Copper)": "CPER",
        "ZC=F (Corn)": "CORN",
        "ZW=F (Wheat)": "WEAT"
    }[commodity]
    etf_ticker = st.text_input("ETF for options (IV/smile/surface):", etf_default)
    st.caption("Yahoo free data provides option chains for ETFs/stocks. Futures options are typically not available.")

with right:
    st.subheader("2) Risk-free & Timezone")
    r = st.number_input("Risk-free rate (annual, decimal)", value=0.045, step=0.005, format="%.5f")
    tz = st.selectbox("Local market timezone (for T calc)",
                      ["America/New_York", "Europe/London", "Europe/Zurich",
                       "Europe/Berlin", "UTC"], index=1)

# ------------------ Spot / Last Price ------------------
st.markdown("### Spot / Last Price")
underlying_symbol = commodity.split()[0]
spot = get_spot_price(underlying_symbol)
st.metric(label=f"{underlying_symbol} — Last price", value=spot)

hist = get_history(underlying_symbol, period="6mo", interval="1d")

if hist is not None and len(hist) > 0:
    close_series = None
    if isinstance(hist.columns, pd.MultiIndex):
        key = ("Close", underlying_symbol)
        if key in hist.columns:
            close_series = hist[key]
        elif "Close" in hist.columns.get_level_values(0):
            close_slice = hist.xs("Close", axis=1, level=0, drop_level=False)
            close_series = close_slice.iloc[:, 0]
    else:
        if "Close" in hist.columns:
            close_series = hist["Close"]
        elif "Adj Close" in hist.columns:
            close_series = hist["Adj Close"]

    if close_series is not None:
        close_series = pd.to_numeric(pd.Series(close_series), errors="coerce").dropna()
        x = pd.to_datetime(close_series.index)
        y = close_series.values
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Close"))
        fig_price.update_layout(
            title=f"{underlying_symbol} — 6M Price",
            xaxis_title="Date",
            yaxis_title="Close",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("No Close/Adj Close column available to plot.")
else:
    st.info("No historical data returned to plot.")

# ------------------ Futures Curve ------------------
st.markdown("### Futures Curve (simple)")
st.caption("Provide contract symbols to build a curve (Yahoo naming may differ).")
default_curve_syms = {
    "CL=F (WTI Crude)": ["CL=F","CLZ25.NYM","CLF26.NYM","CLH26.NYM"],
    "GC=F (Gold)": ["GC=F","GCZ25.CMX","GCG26.CMX","GCM26.CMX"],
    "SI=F (Silver)": ["SI=F","SIZ25.CMX","SIF26.CMX"],
    "NG=F (Nat Gas)": ["NG=F","NGZ25.NYM","NGF26.NYM","NGH26.NYM"],
    "HG=F (Copper)": ["HG=F","HGZ25.CMX","HGF26.CMX"],
    "ZC=F (Corn)": ["ZC=F","ZCZ25.CBT","ZCH26.CBT","ZCK26.CBT"],
    "ZW=F (Wheat)": ["ZW=F","ZWZ25.CBT","ZWH26.CBT"]
}.get(commodity, [underlying_symbol])
curve_syms = st.text_input("Comma-separated futures symbols:", ", ".join(default_curve_syms))
curve_list = [s.strip() for s in curve_syms.split(",") if s.strip()]
curve_df = get_futures_curve(curve_list)

if not curve_df.empty:
    fig_curve = px.bar(curve_df, x="contract", y="last", title="Quoted Curve (last)")
    st.plotly_chart(fig_curve, use_container_width=True)
    if len(curve_df) >= 2:
        front = curve_df["last"].iloc[0]
        back = curve_df["last"].iloc[-1]
        structure = "Contango" if back > front else "Backwardation"
        st.success(f"Term structure: **{structure}** (front={front:.2f}, back={back:.2f})")
else:
    st.info("Couldn’t fetch any of the provided futures contracts. Adjust symbols and try again.")

# ------------------ Options — IV Smile & Greeks ------------------
st.markdown("### Options — IV Smile & Greeks (ETF proxy)")
expiries = get_options(etf_ticker)
if not expiries:
    st.warning(f"No option expiries found for {etf_ticker}. Try a different ETF.")
else:
    sel_exp = st.selectbox("Choose expiry", expiries, index=0)
    calls, puts = get_option_chain_df(etf_ticker, sel_exp)

    spot_etf = get_spot_price(etf_ticker)
    try:
        dy = yf.Ticker(etf_ticker).info.get("dividendYield") or 0.0
        q = float(dy)
    except Exception:
        q = 0.0

    exp_dt = datetime.strptime(sel_exp, "%Y-%m-%d")
    local = pytz.timezone(tz)
    exp_dt_local = local.localize(exp_dt.replace(hour=16, minute=0))
    now_local = datetime.now(local)
    T = max((exp_dt_local - now_local).total_seconds(), 0) / (365*24*3600)

    def add_greeks(df, opt_type):
        if df.empty:
            return df
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0))/2
        greeks = df.apply(
            lambda rrow: bs_greeks(
                S=spot_etf, K=float(rrow["strike"]), T=T, r=r,
                sigma=float(rrow["impliedVolatility"]) if pd.notna(rrow["impliedVolatility"]) else 0.0001,
                opt_type=opt_type, q=q
            ),
            axis=1, result_type="expand"
        )
        greeks.columns = ["delta","gamma","vega","theta","rho"]
        return pd.concat([df, greeks], axis=1)

    calls_g, puts_g = add_greeks(calls, "call"), add_greeks(puts, "put")

    iv_calls = calls_g[["strike","impliedVolatility"]].rename(columns={"impliedVolatility":"IV"}); iv_calls["type"]="Call"
    iv_puts  = puts_g[["strike","impliedVolatility"]].rename(columns={"impliedVolatility":"IV"});  iv_puts["type"]="Put"
    iv_smile = pd.concat([iv_calls, iv_puts], ignore_index=True).dropna()

    if not iv_smile.empty:
        # No trendline to avoid extra dependency on statsmodels
        fig_smile = px.scatter(iv_smile, x="strike", y="IV", color="type",
                               title=f"{etf_ticker} — IV Smile ({sel_exp})")
        st.plotly_chart(fig_smile, use_container_width=True)
    else:
        st.info("No IV data available to plot a smile.")

    both = pd.concat([calls_g.assign(type="Call"), puts_g.assign(type="Put")], ignore_index=True)
    top = both.sort_values("openInterest", ascending=False).head(20)
    st.dataframe(top[["type","contractSymbol","strike","lastPrice","bid","ask",
                      "openInterest","impliedVolatility","delta","gamma","vega","theta","rho"]])

# ------------------ 3D IV Surface (ETF proxy, interpolated & smoothed) ------------------
st.markdown("### 3D IV Surface (ETF proxy)")
if expiries:
    max_exp = st.slider("Max expiries to include", min_value=3, max_value=min(16, len(expiries)), value=min(10, len(expiries)))
    strike_window = st.slider("Strike window around spot (as % of spot)", 50, 250, 160, step=10)
    iv_cap = st.slider("IV cap (max shown, %)", 50, 300, 200, step=10)
    min_oi = st.slider("Min open interest filter", 0, 5000, 50, step=50)
    max_spread_pct = st.slider("Max bid-ask spread (% of mid)", 1, 200, 50, step=5)
    smooth_sigma = st.slider("Smoothing (Gaussian σ)", 0, 5, 1, help="0 = no smoothing")

    chosen_exps = expiries[:max_exp]
    rows = []
    for e in chosen_exps:
        calls_e, puts_e = get_option_chain_df(etf_ticker, e)

        # Mid price & quality filters
        for df in (calls_e, puts_e):
            df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
            df["spr_pct"] = np.where(df["mid"] > 0, (df["ask"] - df["bid"]) / df["mid"], np.nan)

        both = pd.concat([calls_e, puts_e], ignore_index=True)
        both = both[(both["openInterest"].fillna(0) >= min_oi)]
        both = both[(both["spr_pct"].fillna(0) <= max_spread_pct/100)]

        if both.empty:
            continue

        # Median IV per strike
        iv_med = (both.groupby("strike", as_index=False)["impliedVolatility"].median()
                        .rename(columns={"impliedVolatility":"IV"}))
        iv_med["expiry"] = e
        rows.append(iv_med)

    if rows:
        surf = pd.concat(rows, ignore_index=True).dropna()
        # Keep within strike window
        lo = spot_etf * (1 - strike_window/100)
        hi = spot_etf * (1 + strike_window/100)
        surf = surf[(surf["strike"] >= lo) & (surf["strike"] <= hi)]
        # Sanity cap and positive IV only
        surf = surf[(surf["IV"] > 0) & (surf["IV"] <= iv_cap/100)]

        # Use moneyness so expiries align on x
        surf["moneyness"] = surf["strike"] / max(spot_etf, 1e-9)

        if not surf.empty:
            # Build regular grid
            m_grid = np.linspace(surf["moneyness"].min(), surf["moneyness"].max(), 60)
            # numeric x for expiries, keep labels
            exp_idx = {e:i for i, e in enumerate(sorted(surf["expiry"].unique()))}
            surf["exp_i"] = surf["expiry"].map(exp_idx)
            x_grid = np.linspace(min(exp_idx.values()), max(exp_idx.values()), len(exp_idx))
            Xg, Yg = np.meshgrid(x_grid, m_grid)   # X = expiry index, Y = moneyness

            # Interpolate IV to grid
            points = np.column_stack([surf["exp_i"].values, surf["moneyness"].values])
            values = surf["IV"].values
            Z = griddata(points, values, (Xg, Yg), method="linear")

            # Light smoothing (optional)
            if smooth_sigma > 0:
                Z = gaussian_filter(Z, sigma=smooth_sigma)

            # Plot surface: x axis labeled by expiry strings
            tickvals = list(exp_idx.values())
            ticktext = list(sorted(exp_idx.keys()))
            fig_surface = go.Figure(data=[
                go.Surface(x=Xg, y=Yg, z=Z, colorscale="Viridis", showscale=True)
            ])
            fig_surface.update_layout(
                title=f"{etf_ticker} — IV Surface",
                scene=dict(
                    xaxis_title="Expiry",
                    xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext),
                    yaxis_title="Moneyness (K/S)",
                    zaxis_title="IV",
                    zaxis=dict(tickformat=".0%")
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        else:
            st.info("Not enough data after filters to render a surface. Relax filters or widen the window.")
    else:
        st.info("No expiries returned for surface.")

