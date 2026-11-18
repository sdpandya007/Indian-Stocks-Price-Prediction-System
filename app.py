"""Forecast Desk - stock price prediction with Streamlit.

Run:  pip install -r requirements.txt  &&  streamlit run app.py
Educational use only. Not financial advice.
"""
import io
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

INK, MUTED, TEAL, UP, DOWN, AMBER = "#0E1B2C", "#5B6B7F", "#0A7C86", "#138A5B", "#C8374B", "#E0A030"
ENSEMBLE = ["Ridge", "Random Forest", "Gradient Boosting"]
ANCHOR = 0.06  # long-run market growth assumption (log return per year)
TAPE_INDIA = ("^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS")
LABELS = {"^GSPC": "S&P 500", "^NSEI": "NIFTY 50", "^BSESN": "SENSEX"}
NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
# Used only if NSE's full list can't be downloaded.
FALLBACK = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys", "ICICIBANK": "ICICI Bank", "SBIN": "State Bank of India", "ITC": "ITC",
    "BHARTIARTL": "Bharti Airtel", "LT": "Larsen & Toubro", "HINDUNILVR": "Hindustan Unilever",
    "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank", "BAJFINANCE": "Bajaj Finance",
    "ASIANPAINT": "Asian Paints", "MARUTI": "Maruti Suzuki", "SUNPHARMA": "Sun Pharma",
    "TITAN": "Titan Company", "ULTRACEMCO": "UltraTech Cement", "WIPRO": "Wipro", "HCLTECH": "HCL Technologies",
    "NTPC": "NTPC", "POWERGRID": "Power Grid", "ONGC": "ONGC", "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel", "JSWSTEEL": "JSW Steel", "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports", "COALINDIA": "Coal India", "M&M": "Mahindra & Mahindra",
    "NESTLEIND": "Nestle India", "BAJAJFINSV": "Bajaj Finserv", "BAJAJ-AUTO": "Bajaj Auto",
    "TECHM": "Tech Mahindra", "INDUSINDBK": "IndusInd Bank", "DRREDDY": "Dr Reddy's", "CIPLA": "Cipla",
    "EICHERMOT": "Eicher Motors", "HEROMOTOCO": "Hero MotoCorp", "GRASIM": "Grasim", "HINDALCO": "Hindalco",
    "BRITANNIA": "Britannia", "APOLLOHOSP": "Apollo Hospitals", "DIVISLAB": "Divi's Laboratories",
    "SBILIFE": "SBI Life", "HDFCLIFE": "HDFC Life", "BPCL": "BPCL", "TATACONSUM": "Tata Consumer",
    "ZOMATO": "Zomato", "IRCTC": "IRCTC", "DMART": "Avenue Supermarts", "PIDILITIND": "Pidilite",
    "VEDL": "Vedanta", "YESBANK": "Yes Bank", "IDEA": "Vodafone Idea", "PNB": "Punjab National Bank",
    "BANKBARODA": "Bank of Baroda", "HAL": "Hindustan Aeronautics", "BEL": "Bharat Electronics",
    "TRENT": "Trent", "PAYTM": "One 97 Communications",
}

st.set_page_config(page_title="Forecast Desk", page_icon="📈", layout="wide")

# ----------------------------------------------------------------- styling (CSS)
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: #EDF1F5; color: {INK}; }}
.block-container {{ padding-top: 1rem; max-width: 1280px; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #D9E1EA; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.02em; color: {INK}; }}
.title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.6rem;
          letter-spacing: -0.03em; margin: 0.6rem 0 0.1rem; line-height: 1.05; }}
.sub {{ color: {MUTED}; margin: 0 0 1.1rem; max-width: 60ch; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }}
.kpi {{ background: #fff; border: 1px solid #D9E1EA; border-radius: 10px; padding: 14px 16px; }}
.kpi .l {{ color: {MUTED}; font-size: .82rem; }}
.kpi .v {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }}
.kpi .s {{ color: {MUTED}; font-size: .8rem; }}
.kpi.up .v {{ color: {UP}; }} .kpi.down .v {{ color: {DOWN}; }}
.kpi.main {{ border-left: 4px solid {TEAL}; }}
div[data-testid="stTabs"] button[role="tab"] {{ font-weight: 500; }}
@media (max-width: 760px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} .title {{ font-size: 2rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def nse_universe():
    """All NSE-listed equities (about 2,000). Returns ({symbol: name}, loaded_live)."""
    try:
        r = requests.get(NSE_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = [c.strip() for c in d.columns]
        d = d[d["SERIES"].str.strip().isin(["EQ", "BE"])].dropna(subset=["SYMBOL"])
        out = {str(a).strip(): str(b).strip().title() for a, b in zip(d["SYMBOL"], d["NAME OF COMPANY"])}
        if len(out) > 500:
            return dict(sorted(out.items())), True
    except Exception:
        pass
    return dict(sorted(FALLBACK.items())), False


@st.cache_data(ttl=600, show_spinner=False)
def tape_quotes(syms: tuple) -> list:
    try:
        px = yf.download(list(syms), period="5d", progress=False)["Close"]
        out = []
        for s in syms:
            if s in px:
                col = px[s].dropna()
                if len(col) >= 2:
                    out.append({"s": LABELS.get(s, s.replace(".NS", "")), "p": float(col.iloc[-1]),
                                "c": float(col.iloc[-1] / col.iloc[-2] - 1) * 100})
        return out
    except Exception:
        return []


# ----------------------------------------------------------------- features + models
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = lc.diff(n)
    for n in (5, 10, 20, 50):
        f[f"sma_{n}"] = c / c.rolling(n).mean() - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - (ma - 2 * sd)) / (4 * sd)
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    f["range"] = (df["High"] - df["Low"]) / c
    return f.replace([np.inf, -np.inf], np.nan)


def build(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features=0.5,
                                     n_jobs=-1, random_state=7)
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.04,
                                     subsample=0.8, random_state=7)


def fit_models(names, X, y):
    return [build(n).fit(X, y) for n in names]


def predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def importance(models, cols) -> pd.Series:
    out = []
    for m in models:
        v = np.abs(m.named_steps["ridge"].coef_) if hasattr(m, "named_steps") else m.feature_importances_
        out.append(v / v.sum())
    return pd.Series(np.mean(out, axis=0), index=cols).sort_values(ascending=False)


def limited_forecast(df: pd.DataFrame, horizon: int) -> dict:
    """For new listings: too little history to train a model, so show a volatility range with no trend."""
    close = df["Close"]
    last = float(close.iloc[-1])
    sd = float(np.log(close).diff().dropna().std())
    sig = sd * np.sqrt(np.arange(1, horizon + 1))
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"price": np.full(horizon, last), "lo": last * np.exp(-1.2816 * sig),
                       "hi": last * np.exp(1.2816 * sig)}, index=dates)
    return {"df": df, "feats": make_features(df), "fc": fc, "bt": None, "metrics": None, "imp": None,
            "last": last, "lean": "Neutral", "limited": True}


@st.cache_data(ttl=900, show_spinner="Training models…")
def run_forecast(ticker: str, period: str, horizon: int, model_name: str) -> dict:
    df = load_data(ticker, period)
    if df.empty:
        return {"error": f"No price data found for '{ticker}'. Check the symbol (NSE symbols end in .NS, BSE in .BO) "
                         "or try again later."}
    if len(df) < 20:
        return {"error": f"'{ticker}' has only {len(df)} trading days of prices, too few for any forecast."}
    if len(df) < 200:
        return limited_forecast(df, horizon)
    close, lc = df["Close"], np.log(df["Close"])
    feats = make_features(df).dropna()
    names = ENSEMBLE if model_name == "Ensemble" else [model_name]

    def xy(k):
        y = (lc.shift(-k) - lc).reindex(feats.index)
        d = feats.join(y.rename("y")).dropna()
        return d.drop(columns="y"), d["y"]

    # Honest backtest: train on first 80%, leave a gap of `horizon` days, test on the last 20%.
    X, y = xy(horizon)
    split = int(len(X) * 0.8)
    models = fit_models(names, X.iloc[: split - horizon], y.iloc[: split - horizon])
    Xte, yte = X.iloc[split:], y.iloc[split:]
    pr = predict(models, Xte)
    sigma = float(np.std(yte.values - pr))
    c0 = close.reindex(Xte.index).values
    pos = close.index.get_indexer(Xte.index)
    bt = pd.DataFrame({"actual": c0 * np.exp(yte.values), "model": c0 * np.exp(pr), "naive": c0},
                      index=close.index[pos + horizon])
    mae, mae_n = (bt.actual - bt.model).abs().mean(), (bt.actual - bt.naive).abs().mean()
    metrics = {
        "mae": mae,
        "mape": ((bt.actual - bt.model).abs() / bt.actual).mean() * 100,
        "hit": float(np.mean(np.sign(pr) == np.sign(yte.values)) * 100),
        "skill": (1 - mae / mae_n) * 100,
        "n": len(bt),
    }

    # Final models on all data at a few horizons, interpolated into a path.
    knots = sorted({int(round(k)) for k in np.linspace(1, horizon, min(6, horizon))})
    x_last, rets = feats.iloc[[-1]], []
    for k in knots:
        Xk, yk = xy(k)
        m = fit_models(names, Xk, yk)
        rets.append(float(predict(m, x_last)[0]))
    steps = np.arange(1, horizon + 1)
    path = np.interp(steps, knots, rets)
    sig = sigma * np.sqrt(steps / horizon)
    last = float(close.iloc[-1])
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"price": last * np.exp(path),
                       "lo": last * np.exp(path - 1.2816 * sig),
                       "hi": last * np.exp(path + 1.2816 * sig)}, index=dates)
    lean = "Bullish" if path[-1] > 0.5 * sigma else "Bearish" if path[-1] < -0.5 * sigma else "Neutral"
    return {"df": df, "feats": feats, "fc": fc, "bt": bt, "metrics": metrics,
            "imp": importance(m, list(feats.columns)), "last": last, "lean": lean}



@st.cache_data(ttl=900, show_spinner="Building the outlook…")
def run_longterm(ticker: str, years: int, blend: float) -> dict:
    """Statistical 1-5 year outlook: blended growth assumption + historical volatility (lognormal cone)."""
    df = load_data(ticker, "max")
    if len(df) < 504:
        return {"error": f"'{ticker}' has under 2 years of history, which is too little for a multi-year outlook."}
    close = df["Close"]
    z80, z50 = 1.2816, 0.6745

    def stats(c):
        lr = np.log(c).diff().dropna()
        hist = float(np.log(c.iloc[-1] / c.iloc[0]) / (len(lr) / 252))
        return blend * hist + (1 - blend) * ANCHOR, float(lr.std() * np.sqrt(252)), hist

    mu, sig, hist = stats(close.iloc[-2520:])
    last = float(close.iloc[-1])
    n = int(years * 252)
    t = np.arange(1, n + 1) / 252
    dates = close.index[-1] + pd.to_timedelta(t * 365.25, unit="D")
    fc = pd.DataFrame({"median": last * np.exp(mu * t)}, index=dates)
    for z, a, b in ((z80, "lo80", "hi80"), (z50, "lo50", "hi50")):
        fc[a] = last * np.exp(mu * t - z * sig * np.sqrt(t))
        fc[b] = last * np.exp(mu * t + z * sig * np.sqrt(t))

    # Reality check: how often did past 1-year outcomes land inside the 80% range?
    hits = []
    for i in range(504, len(close) - 252, 21):
        m_, s_, _ = stats(close.iloc[max(0, i - 2520): i + 1])
        c0 = close.iloc[i]
        hits.append(c0 * np.exp(m_ - z80 * s_) <= close.iloc[i + 252] <= c0 * np.exp(m_ + z80 * s_))
    return {"df": df, "fc": fc, "last": last, "mu": mu, "sig": sig, "hist": hist,
            "coverage": float(np.mean(hits) * 100) if hits else None, "tests": len(hits)}


# ----------------------------------------------------------------- UI pieces (HTML + JS)
def ticker_tape(quotes: list):
    html = """
<style>
  body{margin:0;font:500 13px 'IBM Plex Sans',system-ui,sans-serif;color:#0E1B2C;background:#fff}
  .wrap{overflow:hidden;border:1px solid #D9E1EA;border-radius:10px;padding:10px 0;background:#fff}
  .track{display:inline-flex;white-space:nowrap;animation:scroll linear infinite}
  .track:hover{animation-play-state:paused}
  .i{padding:0 22px}.i b{margin-right:6px}.u{color:#138A5B;font-style:normal}.d{color:#C8374B;font-style:normal}
  @keyframes scroll{to{transform:translateX(-50%)}}
  @media (prefers-reduced-motion:reduce){.track{animation:none}.wrap{overflow-x:auto}}
</style>
<div class="wrap"><div class="track" id="t"></div></div>
<script>
  const q = __DATA__, t = document.getElementById('t');
  if (!q.length) { t.style.animation = 'none'; t.textContent = '  Market snapshot unavailable right now.'; }
  else {
    const item = d => `<span class="i"><b>${d.s}</b>${d.p.toLocaleString(undefined,{maximumFractionDigits:2})}
      <em class="${d.c>=0?'u':'d'}">${d.c>=0?'▲':'▼'} ${Math.abs(d.c).toFixed(2)}%</em></span>`;
    const row = q.map(item).join('');
    t.innerHTML = row + row;
    t.style.animationDuration = Math.max(24, q.length * 6) + 's';
  }
</script>"""
    components.html(html.replace("__DATA__", json.dumps(quotes)), height=48)


def kpi(label, value, sub="", tone=""):
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def price_chart(df, fc, last, n, show_ma, show_bb):
    d = df.iloc[-n:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    if show_ma:
        for k, col in ((20, AMBER), (50, "#6C5CE7")):
            fig.add_trace(go.Scatter(x=d.index, y=df.Close.rolling(k).mean().loc[d.index], name=f"SMA {k}",
                                     line=dict(width=1.3, color=col)), 1, 1)
    if show_bb:
        ma, sd = df.Close.rolling(20).mean(), df.Close.rolling(20).std()
        for s, nm in ((2, "Upper band"), (-2, "Lower band")):
            fig.add_trace(go.Scatter(x=d.index, y=(ma + s * sd).loc[d.index], name=nm,
                                     line=dict(width=1, color="#9AA8B8", dash="dot")), 1, 1)
    xs = [d.index[-1]] + list(fc.index)
    hi, lo, mid = [last] + list(fc.hi), [last] + list(fc.lo), [last] + list(fc.price)
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(10,124,134,0.16)",
                             line=dict(width=0), hoverinfo="skip", name="80% range"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=mid, name="Forecast", line=dict(color=TEAL, width=2.6, dash="dash"),
                             mode="lines+markers", marker=dict(size=4)), 1, 1)
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, name="Volume", showlegend=False,
                         marker_color=np.where(d.Close >= d.Open, "rgba(19,138,91,.45)", "rgba(200,55,75,.45)")), 2, 1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(template="plotly_white", height=560, margin=dict(l=8, r=8, t=10, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), font=dict(family="IBM Plex Sans"))
    return fig



def render_long(ticker, years, blend):
    res = run_longterm(ticker, years, blend)
    if "error" in res:
        st.error(res["error"])
        st.stop()
    df, fc, last, mu, hist, sig = res["df"], res["fc"], res["last"], res["mu"], res["hist"], res["sig"]
    end = fc.iloc[-1]
    chg = (end["median"] / last - 1) * 100
    st.markdown(
        '<div class="kpis">'
        + kpi(f"{ticker} last close", f"{CUR}{last:,.2f}", f"as of {df.index[-1]:%b %d, %Y}")
        + kpi(f"Median in {years} yr", f"{CUR}{end['median']:,.2f}", f"{chg:+.0f}% from today", "main")
        + kpi("80% range", f"{CUR}{end['lo80']:,.0f} to {CUR}{end['hi80']:,.0f}", "10th to 90th percentile")
        + kpi("Assumed growth", f"{mu * 100:+.1f}%/yr", f"own history {hist * 100:+.1f}%/yr, volatility {sig * 100:.0f}%/yr",
              "up" if mu > 0 else "down")
        + "</div>",
        unsafe_allow_html=True,
    )
    h, x = df["Close"].iloc[-1260:], list(fc.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h, name="Price", line=dict(color=INK, width=1.8)))
    for lo, hi, a, nm in (("lo80", "hi80", 0.13, "80% range"), ("lo50", "hi50", 0.22, "50% range")):
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(fc[hi]) + list(fc[lo][::-1]), fill="toself",
                                 fillcolor=f"rgba(10,124,134,{a})", line=dict(width=0), hoverinfo="skip", name=nm))
    fig.add_trace(go.Scatter(x=x, y=fc["median"], name="Median path", line=dict(color=TEAL, width=2.6, dash="dash")))
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=8, r=8, t=10, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", hovermode="x unified",
                      legend=dict(orientation="h", y=1.05), font=dict(family="IBM Plex Sans"))
    st.plotly_chart(fig, use_container_width=True)

    t1, t2, t3 = st.tabs(["Year by year", "Reality check", "How it works"])
    with t1:
        rows = []
        for y in range(1, years + 1):
            r = fc.iloc[y * 252 - 1]
            rows.append({"Year": y, "Bear case (10th pct)": f"{CUR}{r['lo80']:,.2f}", "Median": f"{CUR}{r['median']:,.2f}",
                         "Bull case (90th pct)": f"{CUR}{r['hi80']:,.2f}", "Median change": f"{(r['median'] / last - 1) * 100:+.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.download_button("Download outlook (CSV)", fc.iloc[::21].round(2).to_csv().encode(), f"{ticker}_outlook.csv", "text/csv")
    with t2:
        if res["coverage"] is None:
            st.info("Not enough history to run a reality check for this ticker.")
        else:
            st.metric("Past 1-year outcomes inside the 80% range", f"{res['coverage']:.0f}%",
                      f"target 80%, {res['tests']} overlapping tests")
            st.caption("We replayed this method from past dates and checked where the price actually landed one year "
                       "later. Near 80% means the range is honest about uncertainty. Well below means real outcomes "
                       "were wider than the range suggests.")
    with t3:
        st.markdown(
            f"Nobody can predict a stock's price five years out, so this shows a **range of outcomes** instead.\n\n"
            f"- **Growth:** a blend of this stock's own past growth ({hist * 100:+.1f}%/yr) and a market-average "
            f"assumption ({ANCHOR * 100:.0f}%/yr), weighted {blend * 100:.0f}% / {100 - blend * 100:.0f}%.\n"
            f"- **Uncertainty:** the stock's own historical volatility ({sig * 100:.0f}%/yr), which widens with the "
            f"square root of time.\n"
            f"- **Not included:** earnings, news, interest rates, or any company-specific change.")


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Find a stock")
    universe, live = nse_universe()
    opts = [f"{k} ({v})" for k, v in universe.items()]
    default = next((i for i, o in enumerate(opts) if o.startswith("RELIANCE (")), 0)
    pick = st.selectbox(f"Search {len(opts):,} NSE stocks", opts, index=default,
                        help="Click the box and type a company name or symbol.")
    custom = st.text_input("Or type any symbol", "", placeholder="BSE example: 500325.BO").strip().upper()
    ticker = custom or pick.split(" (")[0]
    if "." not in ticker and not ticker.startswith("^"):
        ticker += ".NS"
    if not live:
        st.caption("NSE's full list couldn't be downloaded, so 60 major stocks are listed. "
                   "You can still type any NSE or BSE symbol above.")
    CUR = "₹"
    mode = st.radio("Prediction range", ["Next 1-30 days", "Next 1-5 years"])
    short = mode.startswith("Next 1-30")
    if short:
        with st.form("controls"):
            period = st.selectbox("Training history", ["2y", "5y", "10y"], index=1)
            horizon = st.slider("Forecast horizon (trading days)", 1, 30, 10)
            model_name = st.selectbox("Model", ["Ensemble", "Ridge", "Random Forest", "Gradient Boosting"])
            st.form_submit_button("Run forecast", type="primary", use_container_width=True)
        st.markdown("### Chart")
        window = st.select_slider("Days shown", [60, 120, 250, 500], value=120)
        show_ma = st.toggle("Moving averages", True)
        show_bb = st.toggle("Bollinger bands", False)
    else:
        years = st.slider("Years ahead", 1, 5, 5)
        blend = st.slider("Weight on this stock's own past growth", 0, 100, 50, format="%d%%") / 100
    st.caption("Educational tool. Not financial advice.")

# ----------------------------------------------------------------- main page
ticker_tape(tape_quotes(TAPE_INDIA))
st.markdown('<h1 class="title">Forecast desk</h1>'
            '<p class="sub">Search any NSE-listed stock, then see the next days (machine learning, with a backtest) or the next '
            'five years (a range of outcomes, not a single guess).</p>',
            unsafe_allow_html=True)

if not short:
    render_long(ticker, years, blend)
    st.stop()

res = run_forecast(ticker, period, horizon, model_name)
if "error" in res:
    st.error(res["error"])
    st.stop()

df, fc, bt, m, last = res["df"], res["fc"], res["bt"], res["metrics"], res["last"]
chg = (fc.price.iloc[-1] / last - 1) * 100
if res.get("limited"):
    st.info(f"{ticker} has only {len(df)} trading days of history, too few to train or backtest a model. "
            "This is a volatility range around today's price with no trend assumed. "
            "Full model forecasts unlock at about 200 days of history.")
tone = "up" if res["lean"] == "Bullish" else "down" if res["lean"] == "Bearish" else ""
st.markdown(
    '<div class="kpis">'
    + kpi(f"{ticker} last close", f"{CUR}{last:,.2f}", f"as of {df.index[-1]:%b %d, %Y}")
    + kpi(f"Forecast, {horizon} trading days", f"{CUR}{fc.price.iloc[-1]:,.2f}",
          f"80% range {CUR}{fc.lo.iloc[-1]:,.2f} to {CUR}{fc.hi.iloc[-1]:,.2f}", "main")
    + kpi("Expected change", f"{chg:+.2f}%", f"Model lean: {res['lean']}", tone)
    + (kpi("Mode", "Range only", "limited history, no model") if m is None
       else kpi("Backtest hit rate", f"{m['hit']:.0f}%", f"direction right, {m['n']} test days"))
    + "</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(price_chart(df, fc, last, window, show_ma, show_bb), use_container_width=True)

t1, t2, t3, t4 = st.tabs(["Backtest", "Indicators", "What drives it", "Data"])

with t1:
    if m is None:
        st.info("A backtest needs about 200 days of history. This stock is too new.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt.actual, name="Actual", line=dict(color=INK, width=2)))
        fig.add_trace(go.Scatter(x=bt.index, y=bt.model, name="Model", line=dict(color=TEAL, width=2)))
        fig.add_trace(go.Scatter(x=bt.index, y=bt.naive, name="Naive (no change)", line=dict(color="#9AA8B8", dash="dot")))
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=8, r=8, t=10, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average error", f"{CUR}{m['mae']:,.2f}")
        c2.metric("Error as % of price", f"{m['mape']:.2f}%")
        c3.metric("Skill vs naive", f"{m['skill']:+.1f}%")
        if m["skill"] <= 0:
            st.warning("On unseen data this model did not beat “price stays the same.” Treat the forecast as a rough "
                       "scenario, not a signal. Try another model, horizon, or longer history.")
        else:
            st.success("On unseen data this model beat “price stays the same.” Edges like this are small and can "
                       "disappear, so keep the error range in mind.")

with t2:
    f = res["feats"].loc[df.index[-window:].intersection(res["feats"].index)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("RSI (14)", "MACD histogram"))
    fig.add_trace(go.Scatter(x=f.index, y=f.rsi, line=dict(color=TEAL), name="RSI"), 1, 1)
    fig.add_hline(y=70, line_dash="dot", line_color=DOWN, row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=UP, row=1, col=1)
    fig.add_trace(go.Bar(x=f.index, y=f.macd_hist, name="MACD hist",
                         marker_color=np.where(f.macd_hist >= 0, UP, DOWN)), 2, 1)
    fig.update_layout(template="plotly_white", height=460, showlegend=False, margin=dict(l=8, r=8, t=30, b=8),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    rsi = float(res["feats"].rsi.iloc[-1])
    st.caption(f"RSI is {rsi:.0f}: " + ("above 70, often read as overbought." if rsi > 70 else
               "below 30, often read as oversold." if rsi < 30 else "in the neutral zone."))

with t3:
    if m is None:
        st.info("Model drivers appear once the stock has about 200 days of history.")
    else:
        imp = res["imp"].head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h", marker_color=TEAL))
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=8, r=8, t=10, b=8),
                          xaxis_title="Share of model weight (%)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Features are technical signals: returns, moving-average distance, RSI, MACD, Bollinger position, "
                   "volatility and volume.")

with t4:
    st.dataframe(df.join(res["feats"][["rsi"]]).tail(250).round(2), use_container_width=True)
    st.download_button("Download forecast (CSV)", fc.round(2).to_csv().encode(), f"{ticker}_forecast.csv", "text/csv")
