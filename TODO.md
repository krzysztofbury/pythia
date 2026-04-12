# Oracle Pythia — Quant Foundation Roadmap

Long-term improvements for transforming the bot into a quantitative trading system.
These items require data infrastructure, historical data feeds, and validation frameworks
that go beyond the current algorithmic scoring engine.

---

## 1. Walk-Forward Backtesting

**Why:** Current backtester uses a single train period. Walk-forward optimization splits
data into multiple train/test windows, re-optimizing parameters at each step.
This reveals overfitting and measures true out-of-sample performance.

**What to build:**
- Rolling window splitter (e.g., 60-day train / 20-day test, rolling forward 20 days)
- Per-window Optuna optimization
- Aggregate out-of-sample performance metrics
- Comparison report: optimized vs static params vs buy-and-hold

**Dependencies:** Optuna integration (Task 9 in Professional Upgrade)

---

## 2. HMM-Based Regime Detection

**Why:** Current rule-based regime detection (ADX + EMA + ROC) is brittle and has
hard thresholds. Hidden Markov Models learn regime transitions from data,
adapting to changing market dynamics.

**What to build:**
- HMM with 3-4 hidden states trained on returns + volatility + volume
- Online regime probability estimation (no future data leakage)
- Regime transition matrix for probabilistic strategy switching
- Comparison framework: HMM vs rule-based classification accuracy

**Dependencies:** hmmlearn or pomegranate library, historical OHLCV data

---

## 3. Risk-Adjusted Position Sizing

**Why:** Current system allocates fixed USD per slot regardless of per-asset risk.
A 1% move in BTC is very different from a 1% move in SHIB.

**What to build:**
- Volatility-scaled position sizing: allocate based on inverse ATR
  (lower volatility = larger position, higher = smaller)
- Kelly criterion approximation: size positions based on edge and win rate
- Risk parity across portfolio: equal risk contribution per position
- Maximum portfolio VaR constraint

**Dependencies:** Per-asset win rate tracking, rolling Sharpe per ticker

---

## 4. Signal Analytics & Decay Measurement

**Why:** We don't know how fast our signals become stale or which signals
actually predict returns. This is the foundation of signal research.

**What to build:**
- Signal half-life measurement: how many candles until signal loses predictive power
- Win rate by regime: measure if BUY signals in TRENDING_UP are better than in RANGING
- Maximum Adverse Excursion (MAE): worst drawdown during each trade
- Maximum Favorable Excursion (MFE): best unrealized profit during each trade
- Factor attribution: which composite components contributed most to winning trades

**Dependencies:** market_snapshots table (already built), trade journal enrichment

---

## 5. Order Book Microstructure Signals

**Why:** Price and volume tell you what happened. Order book tells you what's
about to happen. Bid/ask imbalance, depth, and whale detection provide
short-term predictive edge.

**What to build:**
- Real-time order book depth monitoring via CCXT WebSocket
- Bid/ask imbalance ratio as a signal component
- Large order detection (whale alerts)
- Trade flow imbalance (aggressive buys vs sells)
- Integration as new weight in composite score

**Dependencies:** CCXT WebSocket support, real-time data infrastructure

---

## 6. Historical Data Feeds for Backtest Fidelity

**Why:** Current backtest uses empty strings for sentiment/consensus/polymarket.
Without historical context data, backtests don't reflect real performance.

**What to build:**
- Historical Fear & Greed Index data feed (CoinMarketCap or Alternative.me API)
- Historical Polymarket resolution data for specific crypto markets
- Grok sentiment archival: store and replay sentiment scores over time
- Data quality scoring: measure what % of backtest has real vs synthetic data

**Dependencies:** market_snapshots table (already built), external API subscriptions

---

## 7. Reinforcement Learning Integration

**Why:** Rule-based strategies have fixed behavior. RL agents learn to adapt
their trading policy from experience, potentially finding non-obvious patterns.

**What to build:**
- Gym-compatible trading environment wrapping the existing strategy
- DQN or PPO agent for position sizing and entry/exit timing
- Reward function: risk-adjusted returns (Sortino ratio)
- Paper-trade validation before any live deployment
- Ensemble: RL agent as one more signal source (10-15% weight), not the sole decision maker

**Dependencies:** stable-baselines3 or FinRL, GPU for training, months of historical data

---

## 8. Multi-Exchange Support

**Why:** Single exchange (Kraken) creates concentration risk and limits
arbitrage opportunities. Different exchanges have different liquidity profiles.

**What to build:**
- Exchange abstraction layer (already partially done via CCXT)
- Cross-exchange price comparison for best execution
- Exchange-specific fee optimization
- Failover: if primary exchange is down, route to secondary

**Dependencies:** Additional API keys, testing infrastructure per exchange

---

## 9. Portfolio Optimization

**Why:** Current allocation is equal-weight per slot. Modern portfolio theory
provides mathematically optimal allocation strategies.

**What to build:**
- Efficient frontier calculation for the trading universe
- Hierarchical Risk Parity (HRP) for crypto (handles non-normal returns)
- Dynamic rebalancing based on regime and correlation shifts
- Transaction cost-aware optimization (don't rebalance for tiny improvements)

**Dependencies:** PyPortfolioOpt or riskfolio-lib, correlation matrix estimation

---

## 10. Advanced Performance Metrics

**Why:** Sharpe ratio alone is insufficient. Crypto returns are non-normal
(fat tails, skewness), requiring specialized metrics.

**What to build:**
- Sortino ratio (downside deviation only)
- Calmar ratio (return / max drawdown)
- Omega ratio (probability-weighted gains vs losses)
- Rolling metrics (30d, 90d windows)
- Equity curve visualization
- Monthly/weekly performance attribution

**Dependencies:** quantstats (already in deps), matplotlib for charts
