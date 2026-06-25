# 📈 TradeSense AI — Trading Performance Coach & Risk Dashboard

> Agent monitoring + risk-scoring dashboard for Bitget traders. Built for the Bitget AI Base Camp Hackathon S1 (Track 2 — Trading Infra).

TradeSense AI turns raw trade history into an actionable performance review: it syncs **closed positions directly from Bitget** (read-only API), computes pro-grade risk metrics, scores the account, and uses an LLM to generate specific, behavioral coaching.

## ✨ Features
- **Bitget live sync** — pulls closed futures positions via the Bitget v2 REST API (read-only keys, never stored).
- **CSV upload** + one-click **sample data** for instant demos.
- **Risk metrics** — win rate, total PnL, avg win/loss, profit factor, expectancy, max drawdown, longest win/loss streaks, per-pair breakdown.
- **AI Score (0–100)** — at-a-glance account health rating.
- **AI Coach** — `gpt-4o-mini` generates number-specific insights, with an automatic rule-based fallback (works offline / no key).
- **Visuals** — PnL by pair, equity curve, underwater drawdown curve, PnL by day-of-week.
- **Light / Dark mode** with a Bitget-aligned teal theme.
- **Downloadable CSV report.**

## 🚀 Quickstart
git clone <your-repo-url>

cd tradesense-ai

pip install -r requirements.txt

streamlit run app.py

```
Then either click **Load sample data**, upload a CSV (`Pair, PnL, Direction`, optional `Timestamp`), or enter **read-only** Bitget API keys in the sidebar to sync live.

## 🔑 Optional config
Create `.streamlit/secrets.toml` to enable the real LLM coach:
```
OPENAI_API_KEY = "sk-..."

## 🗂 Project structure
tradesense-ai/

├─ app.py              # Streamlit app

├─ bitget_client.py    # read-only Bitget v2 REST client

├─ requirements.txt

└─ .streamlit/secrets.toml   # optional

## 🔌 Bitget integration
`bitget_client.py` is a minimal, reusable read-only client (correct HMAC-SHA256 + Base64 signing) that other developers can drop into their own Agent projects. It maps Bitget closed positions into a clean, standard schema (`Timestamp, Pair, Direction, PnL`).

## 🔒 Security
API keys are entered at runtime, used read-only, and never persisted.

## 📜 License
MIT
