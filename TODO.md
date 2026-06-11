# Oracle Pythia — Roadmap

Forward-looking work. For what's already shipped see [CHANGELOG.md](CHANGELOG.md).

---

## 1. Wire funding rate into composite score — SHIPPED v9.7 (validation pending)

**Status 2026-06-11:** wired as `W_FUNDING=0.03` with contrarian normalization (±0.1%/8h cap) and active-weight redistribution when data is absent. Remaining before trusting it live: enable `FUNDING_RATES_ENABLED=true`, collect 30+ days of `market_snapshots.funding_rate`, walk-forward validate.

**Why:** v9.5 collects perpetual funding rates into `market_snapshots.funding_rate` but `Strategist.generate_signal` does not yet consume them. Funding is a well-known short-term sentiment proxy (negative rates = shorts paying longs = contrarian bullish).

**What to build:**

- Add a `W_FUNDING` weight (0.03–0.05) to `Strategist.DEFAULT_WEIGHTS` and re-normalize the other seven components.
- Parse `MarketData.funding_rate` into a score in [-1, +1] (negative rate → positive score, capped).
- Extend active-weight normalization to handle missing funding data.
- Walk-forward validate over at least 30 days of collected snapshots before shipping to live.

**Dependencies:** v9.5 funding infrastructure; 30+ days of `market_snapshots.funding_rate` history.

---

## 2. HMM-Based Regime Detection

**Why:** Rule-based regime detection (ADX + EMA + ROC) uses hard thresholds. Hidden Markov Models learn regime transitions from data and adapt to changing market dynamics.

**What to build:**

- HMM with 3–4 hidden states trained on returns + volatility + volume.
- Online regime probability estimation (no future data leakage).
- Regime transition matrix for probabilistic strategy switching.
- Comparison framework: HMM vs rule-based classification accuracy.

**Dependencies:** `hmmlearn` or `pomegranate`, historical OHLCV data.

---

## 3. Volatility-scaled / Kelly position sizing

**Why:** v9.2 delivered conviction tiers (0.6×/1.0×/1.4×/2.0×) + 40% concentration cap. Still flat-dollar per tier; high-volatility tickers carry more risk than low-vol at the same allocation.

**What to build:**

- Volatility-scaled base sizing: `base_size ∝ 1 / ATR_pct` (lower vol = larger position).
- Kelly criterion approximation from v9.4 per-asset edge + win rate.
- Risk parity across portfolio — equal risk contribution per position.
- Maximum portfolio VaR constraint (95% CI, 24h).

**Dependencies:** v9.4 per-asset stats (landed), rolling Sharpe per ticker.

---

## 4. Signal decay and time-of-day edges

**Why:** v9.4 gives us per-asset win rate and factor attribution. Missing: signal half-life (how many candles until predictive power fades) and time-of-day / day-of-week effects.

**What to build:**

- Signal half-life measurement per component.
- Win rate broken down by regime (is BUY in TRENDING_UP better than in RANGING?).
- Hour-of-day / day-of-week edge heatmap.

**Dependencies:** None; v9.4 infrastructure is in place.

---

## 5. Order Book Microstructure Signals

**Why:** Price and volume tell you what happened. Order book tells you what's about to happen.

**What to build:**

- Real-time order book depth monitoring via CCXT WebSocket.
- Bid/ask imbalance ratio as a signal component.
- Large order detection (whale alerts).
- Trade flow imbalance (aggressive buys vs sells).
- Integration as a new weight in the composite score.

**Dependencies:** CCXT WebSocket support, real-time data infrastructure.

---

## 6. Historical Data Feeds for Backtest Fidelity

**Why:** Current backtest uses empty strings for sentiment/consensus/polymarket. Without historical context data, backtests don't reflect real performance.

**v9.5 note:** `market_snapshots` now captures `funding_rate` alongside sentiment/consensus/fear_greed/polymarket_odds/indicators_json/regime — the foundation for a high-fidelity historical replay source is in place. Missing piece: external historical feeds.

**What to build:**

- Historical Fear & Greed Index data feed (CoinMarketCap or Alternative.me API).
- Historical Polymarket resolution data for specific crypto markets.
- LLM sentiment archival: store and replay sentiment scores over time.
- Data quality scoring: measure what % of backtest has real vs synthetic data.

**Dependencies:** External API subscriptions.

---

## 7. Reinforcement Learning Integration

**Why:** Rule-based strategies have fixed behaviour. RL agents learn to adapt their trading policy from experience.

**What to build:**

- Gym-compatible trading environment wrapping the existing strategy.
- DQN or PPO agent for position sizing and entry/exit timing.
- Reward function: risk-adjusted returns (Sortino).
- Paper-trade validation before any live deployment.
- Ensemble: RL agent as one more signal source (10–15% weight), not the sole decision maker.

**Dependencies:** `stable-baselines3` or FinRL, GPU for training, months of historical data.

---

## 8. Multi-Exchange Support

**Why:** Single-exchange concentration risk. Different exchanges have different liquidity profiles.

**What to build:**

- Exchange abstraction layer (partially done via CCXT).
- Cross-exchange price comparison for best execution.
- Exchange-specific fee optimization.
- Failover: if primary exchange is down, route to secondary.

**Dependencies:** Additional API keys, testing infrastructure per exchange.

---

## 9. Portfolio Optimization

**Why:** Current allocation is equal-weight per slot with a concentration cap. Modern portfolio theory provides mathematically optimal allocation strategies.

**What to build:**

- Efficient frontier calculation for the trading universe.
- Hierarchical Risk Parity (HRP) for crypto (handles non-normal returns).
- Dynamic rebalancing based on regime and correlation shifts.
- Transaction-cost-aware optimization.

**Dependencies:** `PyPortfolioOpt` or `riskfolio-lib`, correlation matrix estimation.

---

## 10. Advanced Performance Metrics (remaining)

**v9.3 shipped Sortino, Calmar, Profit Factor.** Still wanted:

- Omega ratio (probability-weighted gains vs losses).
- Rolling metrics (30d, 90d windows).
- Equity curve visualization.
- Monthly/weekly performance attribution.

**Dependencies:** `quantstats` (already in deps), matplotlib for charts.
