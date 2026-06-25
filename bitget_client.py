import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from bitget_client import BitgetClient, bitget_positions_to_df

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="TradeSense AI", page_icon="📈", layout="wide")

# --------------------------------------------------
# THEME — light & dark palettes (toggle in sidebar)
# --------------------------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)

PALETTES = {
    "light": {
        "bg": "#F4F9FA", "surface": "#FFFFFF", "border": "#CFE0E3",
        "text": "#0B2127", "muted": "#5A7077", "faint": "#6E8A91",
        "grid": "#DDEBED", "accent": "#0BA6B8", "accent2": "#2BD4D4",
        "win": "#16C784", "loss": "#F6465D", "shadow": "rgba(11,33,39,0.10)",
    },
    "dark": {
        "bg": "#0B1416", "surface": "#13232A", "border": "#21383F",
        "text": "#E6F2F3", "muted": "#8AA3A9", "faint": "#6E8A91",
        "grid": "#1C2E33", "accent": "#19C6D6", "accent2": "#2EE6E6",
        "win": "#16C784", "loss": "#F6465D", "shadow": "rgba(0,0,0,0.35)",
    },
}
P = PALETTES["dark" if dark_mode else "light"]

CHART_BG, GRID_COLOR = P["bg"], P["grid"]
TEXT_COLOR, WIN_COLOR, LOSS_COLOR, LINE_COLOR = P["text"], P["win"], P["loss"], P["accent"]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Source+Sans+3:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; color: __TEXT__; background-color: __BG__; }
.stApp { background-color: __BG__; }
section[data-testid="stSidebar"] { background-color: __SURFACE__; border-right:1px solid __BORDER__; }
[data-testid="metric-container"] { background:__SURFACE__; border:1px solid __BORDER__; border-radius:14px; padding:16px 18px; box-shadow:0 2px 8px __SHADOW__; }
[data-testid="metric-container"] label { color:__MUTED__ !important; font-size:13px !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:__TEXT__ !important; font-family:'Playfair Display', serif !important; font-size:26px !important; }
h1,h2,h3,h4 { font-family:'Playfair Display', serif !important; color:__TEXT__ !important; }
hr { border-color:__BORDER__ !important; }
.ai-box { background:__SURFACE__; padding:20px 22px; border-radius:16px; border:1px solid __BORDER__; margin-bottom:14px; box-shadow:0 2px 8px __SHADOW__; }
.ai-box h3 { font-family:'Playfair Display', serif; color:__ACCENT__ !important; margin-top:0; }
#MainMenu, footer { visibility:hidden; }
</style>
"""
for _k, _v in {
    "__BG__": P["bg"], "__SURFACE__": P["surface"], "__BORDER__": P["border"],
    "__TEXT__": P["text"], "__MUTED__": P["muted"], "__ACCENT__": P["accent"],
    "__SHADOW__": P["shadow"],
}.items():
    _CSS = _CSS.replace(_k, _v)
st.markdown(_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# HERO  (gradient teal title)
# --------------------------------------------------
st.markdown(f"""
<div style='text-align:center; padding:36px 0 16px 0;'>
  <h1 style='font-size:58px; font-weight:800; margin-bottom:10px;'>📈 <span style='background:linear-gradient(90deg, {P['accent']}, {P['accent2']}); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;'>TradeSense AI</span></h1>
  <p style='font-size:20px; color:{P['muted']}; margin-top:0;'>AI-Powered Trading Performance Coach</p>
  <p style='font-size:15px; color:{P['faint']}; letter-spacing:0.12em;'>ANALYZE • IMPROVE • TRADE SMARTER</p>
</div>
""", unsafe_allow_html=True)
st.caption("Built for the Bitget AI Hackathon • AI-powered trade intelligence")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def cream_layout(fig):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, margin=dict(t=48, l=8, r=8, b=8),
        font=dict(family="Source Sans 3, sans-serif", color=TEXT_COLOR),
        title_font=dict(family="Playfair Display, serif", color=TEXT_COLOR, size=17),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor=CHART_BG, bordercolor=GRID_COLOR, borderwidth=1),
    )
    return fig

def money(x, sym="$"):
    try:
        return f"{sym}{x:,.0f}"
    except Exception:
        return f"{sym}0"

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    n = 120
    base = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(n):
        pair = rng.choice(pairs)
        direction = rng.choice(["Long", "Short"])
        edge = {"BTCUSDT": 40, "ETHUSDT": 25, "SOLUSDT": -10, "XRPUSDT": 5, "DOGEUSDT": -30}[pair]
        pnl = rng.normal(edge, 120)
        rows.append({
            "Timestamp": base + pd.Timedelta(hours=int(rng.integers(0, 24 * 60))),
            "Pair": pair, "Direction": direction, "PnL": round(float(pnl), 2),
        })
    return pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)

def ai_coach(stats: dict) -> str:
    """Real LLM coaching; falls back to rule-based tips if no key/SDK."""
    try:
        from openai import OpenAI
        key = st.secrets.get("OPENAI_API_KEY", None)
        if not key:
            raise RuntimeError("no key")
        client = OpenAI(api_key=key)
        prompt = (
            "You are an elite trading performance coach. Given these stats, give 3 "
            "sharp, specific, behavioral insights (not generic advice). Be direct, "
            f"reference the numbers, use short bullet points.\n\nStats: {stats}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception:
        return rule_based_tips(stats)

def rule_based_tips(s: dict) -> str:
    tips = [f"✅ Focus capital on **{s['best_pair']}** — your strongest pair."]
    if abs(s["avg_loss"]) > s["avg_win"]:
        tips.append("⚠️ Average loss exceeds average win — tighten your stop-losses.")
    else:
        tips.append("✅ Good risk/reward ratio — keep protecting it.")
    if s["win_rate"] < 50:
        tips.append("📌 Win rate below 50% — review entry signals, avoid low-conviction trades.")
    if s["profit_factor"] != float("inf") and s["profit_factor"] < 1.5:
        tips.append("📌 Profit factor below 1.5 — reduce size on losing pairs.")
    tips.append(f"🚫 Limit exposure on **{s['worst_pair']}** until you diagnose the losses.")
    return "\n".join(f"- {t}" for t in tips)

def longest_streaks(pnl: pd.Series):
    win_streak = loss_streak = cur_w = cur_l = 0
    for v in pnl:
        if v > 0:
            cur_w += 1; cur_l = 0
        elif v < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        win_streak = max(win_streak, cur_w)
        loss_streak = max(loss_streak, cur_l)
    return win_streak, loss_streak

# --------------------------------------------------
# SIDEBAR — BITGET LIVE SYNC (sponsor integration)
# --------------------------------------------------
with st.sidebar:
    st.header("🔗 Bitget Live Sync")
    st.caption("Use a READ-ONLY API key. Keys are never stored.")
    bg_key = st.text_input("API Key", type="password")
    bg_secret = st.text_input("API Secret", type="password")
    bg_pass = st.text_input("Passphrase", type="password")
    product = st.selectbox("Product", ["USDT-FUTURES", "COIN-FUTURES", "USDC-FUTURES"])
    sync = st.button("⚡ Sync my account", use_container_width=True)

# --------------------------------------------------
# DATA INPUT  (priority: Bitget live sync > CSV upload > sample data)
# --------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    uploaded = st.file_uploader("Upload your trading history CSV", type=["csv"],
                                help="Required: Pair, PnL, Direction. Optional: Timestamp")
with c2:
    currency = st.selectbox("Currency", ["$", "USDT", "€", "£"], index=0)
    use_sample = st.button("✨ Load sample data", use_container_width=True)

df = None
if sync and bg_key and bg_secret and bg_pass:
    try:
        with st.spinner("Syncing from Bitget…"):
            client = BitgetClient(bg_key, bg_secret, bg_pass)
            df = bitget_positions_to_df(client.history_positions(product))
        if df is None or df.empty:
            st.warning("No closed positions found in the last 3 months.")
            df = None
        else:
            st.success(f"Synced {len(df)} closed positions from Bitget 🎉")
    except Exception as e:
        st.error(f"Bitget sync failed: {e}")
elif uploaded is not None:
    try:
        df = load_csv(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
elif use_sample:
    df = sample_data()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if df is not None:
    # ---- Validation ----
    required = {"Pair", "PnL", "Direction"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Data is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce")
    df = df.dropna(subset=["PnL"]).reset_index(drop=True)
    df["Direction"] = df["Direction"].astype(str).str.strip().str.title()

    # Sort chronologically if a timestamp exists
    has_time = "Timestamp" in df.columns
    if has_time:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    if df.empty:
        st.warning("No valid rows after cleaning. Check your PnL column.")
        st.stop()

    # ---- Calculations ----
    total_trades = len(df)
    wins, losses = df[df["PnL"] > 0], df[df["PnL"] < 0]
    win_count, loss_count = len(wins), len(losses)
    decided = win_count + loss_count
    breakeven = total_trades - decided

    win_rate = win_count / decided * 100 if decided else 0
    total_pnl = df["PnL"].sum()
    avg_win = wins["PnL"].mean() if win_count else 0
    avg_loss = losses["PnL"].mean() if loss_count else 0

    gross_profit, gross_loss = wins["PnL"].sum(), abs(losses["PnL"].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")
    pf_display = "∞" if profit_factor == float("inf") else profit_factor

    loss_rate = loss_count / decided if decided else 0
    expectancy = (win_rate / 100 * avg_win) - (loss_rate * abs(avg_loss))

    cum = df["PnL"].cumsum()
    drawdown = cum - cum.cummax()
    max_dd = drawdown.min()

    pair_stats = df.groupby("Pair")["PnL"].sum()
    best_pair, worst_pair = pair_stats.idxmax(), pair_stats.idxmin()
    win_streak, loss_streak = longest_streaks(df["PnL"])

    pf_bonus = 20 if profit_factor == float("inf") else min(20, int(profit_factor * 5))
    score = min(100, int((win_rate * 0.5) + (30 if total_pnl > 0 else 5) + pf_bonus))
    score_label = "Elite 🏆" if score >= 80 else "Good 👍" if score >= 60 else "Needs Work ⚠️"

    stats = {
        "total_trades": total_trades, "win_rate": round(win_rate, 1),
        "total_pnl": round(float(total_pnl), 2), "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2), "profit_factor": profit_factor,
        "expectancy": round(float(expectancy), 2), "max_drawdown": round(float(max_dd), 2),
        "best_pair": best_pair, "worst_pair": worst_pair,
        "longest_win_streak": win_streak, "longest_loss_streak": loss_streak,
    }

    # ---- Metrics ----
    r1 = st.columns(4)
    r1[0].metric("📊 Trades", f"{total_trades}", f"{breakeven} breakeven" if breakeven else None)
    r1[1].metric("🎯 Win Rate", f"{win_rate:.1f}%")
    r1[2].metric("💰 Total PnL", money(total_pnl, currency))
    r1[3].metric("🤖 AI Score", f"{score}/100", score_label)
    r2 = st.columns(4)
    r2[0].metric("📈 Avg Win", money(avg_win, currency))
    r2[1].metric("📉 Avg Loss", money(avg_loss, currency))
    r2[2].metric("⚖️ Expectancy", money(expectancy, currency))
    r2[3].metric("🔻 Max Drawdown", money(max_dd, currency))

    st.write("")
    left, right = st.columns([2.3, 1])

    with left:
        st.subheader("Trading Analytics")
        fig = px.bar(df, x="Pair", y="PnL", color="Direction", title="Profit & Loss by Pair",
                     color_discrete_map={"Long": WIN_COLOR, "Short": LOSS_COLOR})
        st.plotly_chart(cream_layout(fig), use_container_width=True)

        x = df["Timestamp"] if has_time else df.index
        equity = px.line(x=x, y=cum, title="Equity Curve", labels={"y": "Cumulative PnL", "x": ""})
        equity.update_traces(line_color=LINE_COLOR, line_width=2.5)
        st.plotly_chart(cream_layout(equity), use_container_width=True)

        under = go.Figure(go.Scatter(x=list(x), y=drawdown, fill="tozeroy",
                                     line_color=LOSS_COLOR, name="Drawdown"))
        under.update_layout(title="Underwater (Drawdown) Curve")
        st.plotly_chart(cream_layout(under), use_container_width=True)

        if has_time:
            dow = (df.assign(DOW=df["Timestamp"].dt.day_name())
                     .groupby("DOW")["PnL"].sum()
                     .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
                     .dropna())
            dow_fig = px.bar(x=dow.index, y=dow.values, title="PnL by Day of Week",
                             labels={"x": "", "y": "PnL"})
            dow_fig.update_traces(marker_color=LINE_COLOR)
            st.plotly_chart(cream_layout(dow_fig), use_container_width=True)

        st.subheader("Per-Pair Breakdown")
        per_pair = (df.groupby("Pair")
                      .apply(lambda g: pd.Series({
                          "Trades": len(g),
                          "Win Rate %": round((g["PnL"] > 0).sum() / max((g["PnL"] != 0).sum(), 1) * 100, 1),
                          "Total PnL": round(g["PnL"].sum(), 2),
                      }))
                      .sort_values("Total PnL", ascending=False))
        st.dataframe(per_pair, use_container_width=True)

        st.subheader("Trade History")
        st.dataframe(df, use_container_width=True)

    with right:
        st.markdown(f"""
        <div class="ai-box">
            <h3>🤖 AI Coach</h3>
            <p style="color:{P['muted']};">Analysis complete for {total_trades} trades</p>
            <hr style="border-color:{P['border']}; margin:12px 0;">
            <p><strong>Win Rate:</strong> {win_rate:.1f}%</p>
            <p><strong>Expectancy:</strong> {money(expectancy, currency)}/trade</p>
            <p><strong>Profit Factor:</strong> {pf_display}</p>
            <p><strong>Max Drawdown:</strong> {money(max_dd, currency)}</p>
            <p><strong>Best / Worst:</strong> {best_pair} / {worst_pair}</p>
            <p><strong>Win / Loss streak:</strong> {win_streak} / {loss_streak}</p>
        </div>
        """, unsafe_allow_html=True)

        if win_rate >= 60:
            st.success("Strong trader profile. Your strategy shows consistency.")
        elif win_rate >= 50:
            st.warning("Moderate performance. Better risk management could boost returns.")
        else:
            st.error("Performance needs attention. Review your entries and exits.")

        st.markdown("### 💡 AI Recommendations")
        with st.spinner("Coaching…"):
            st.markdown(ai_coach(stats))

        report = pd.DataFrame([stats]).T.rename(columns={0: "Value"})
        st.download_button("⬇️ Download report (CSV)",
                           report.to_csv().encode("utf-8"),
                           file_name="tradesense_report.csv",
                           mime="text/csv", use_container_width=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding:64px 20px; background:{P['surface']}; border-radius:20px; border:1.5px dashed {P['border']}; margin-top:24px;">
      <p style="font-size:52px; margin-bottom:8px;">📂</p>
      <p style="font-size:20px; color:{P['text']}; font-weight:600;">Sync Bitget (sidebar), upload a CSV, or click “Load sample data” to begin</p>
      <p style="font-size:14px; color:{P['muted']};">Required columns: Pair, PnL, Direction • Optional: Timestamp</p>
    </div>
    """, unsafe_allow_html=True)