# Oracle Pythia 10.0.0

An autonomous crypto trading bot on Kraken with rule-based composite strategy,
exact v10 execution accounting, PAPER/LIVE state isolation, quantitative
analysis, and optional off-path LiteLLM commentary. LLM features are disabled
by default and never submit orders.

## What's New in 10.0.0

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

- **Execution-store authority**: `paper-execution-v10.db` or
  `live-execution-v10.db` is the authoritative scoped order, fill, position,
  and protection ledger. `analytics-v10.db` is a derived journal for signals,
  market snapshots, and reporting; it is not execution truth.
- **Recovery-first runtime**: exact identifiers, Decimal-safe accounting,
  single-process locking, bounded drain, and atomic PAPER checkpoints replace
  reconstruction from the v9 journal.
- **Structured health**: an atomic JSON lifecycle document reports version,
  build, boot identity, state, freshness, cycle deadline, reconciliation, and
  bounded error codes. Docker checks it offline with `src/health.py`.
- **Safe release channels**: `main` publishes a PAPER candidate and one-time
  SHA tag after CI; stable promotion retags the exact tested digest without a
  rebuild and requires production approval.
- **Conservative optional inputs**: LLM sentiment and challenger are off by
  default, LIVE influence requires a separate opt-in, and persistent daily,
  monthly, and shadow spend caps apply. DCA and partial exits are disabled in
  the v10 coordinator until exact protection resizing exists.

### v9 Release History

- **v9.0 — modular refactor**: `main.py` split 1995 -> 359 LOC across `risk_guardian`, `data_pipeline`, `order_executor`, `decision_engine`, `portfolio_manager`. All hardcoded strategy thresholds (entry/exit/DCA/mean-reversion, stale-position) moved to `Settings` so operators tune via `.env` without code edits. Legacy `MIN_CONFIDENCE=0.10` (a Gemini-era artifact) removed; replaced by explicit `min_buy_confidence` / `min_sell_confidence` floors (default 0.0 — defer to strategy thresholds).
- **v9.1 — LiteLLM + LLM Challenger**: `xai-sdk` replaced by LiteLLM so any provider (Claude, Gemini, DeepSeek, GPT, Grok) is a one-line config change. New `LLMChallenger` reviews every BUY/SELL the algorithm makes and logs a structured critique — **commentary only, never blocks execution**. ToS-safe across providers. HIGH-concern reviews fire a notifier alert. Default off; enable with `CHALLENGER_ENABLED=true` plus the relevant provider key.
- **v9.2 — conviction-tiered sizing**: Position size now scales with composite score (0.6× / 1.0× / 1.4× / 2.0× tiers). A 40%-of-balance concentration cap binds per position, and an intra-cycle budget tracker prevents two tier-4 entries in one cycle from stacking to 80% exposure. DCA calls are clamped to 1.0× to stop tier amplification on existing positions. Hard cap ceiling raised $100 → $500 to allow full-conviction sizing on proven accounts.
- **v9.3 — walk-forward optimizer**: Optuna trials now actually thread the sampled weights + entry threshold into the backtest (the previous implementation sampled params but evaluated defaults — every trial identical). Added `WalkForwardOptimizer` with rolling 60d-train / 20d-test windows for honest out-of-sample metrics. `BacktestResult` adds Sortino, Calmar, Profit Factor. `Strategist.__init__(weights_override=...)` enables per-trial strategy configuration.
- **v9.4 — signal analytics**: New `SignalAnalyzer` with per-asset P&L breakdown, Maximum Adverse/Favorable Excursion (vs. original position entry — accounts for DCA correctly), and factor attribution that regex-parses composite component scores from signal reasoning and correlates each with realized P&L.
- **v9.7 — funding composite + execution-truth fixes**: funding rate wired as a 3% contrarian component (opt-in), failed sells keep position state, fill-reconciliation dedup window 15→180 min + dust-fill skip, consensus parsing word-boundaries, "insufficient data" responses excluded from weights.
- **v9.6 — LLM resilience (incident-driven)**: After 7 silent weeks (hard BULLISH-sentiment gate + exhausted xAI credits + sub-$10 allocations), sentiment became advisory (blocks only BEARISH; legacy gate via `REQUIRE_BULLISH_SENTIMENT=true`), primary model moved to `gemini/gemini-3.5-flash` (`SENTIMENT_MODEL`), sentiment+consensus responses cached per ticker (`SENTIMENT_CACHE_SECONDS=3600`), NEUTRAL/unavailable sentiment no longer drags the composite, guardian sizes sub-tier-1 entries at tier-1 instead of rejecting, round-up floor lowered to $5, slots capped by `MAX_TOTAL_CRYPTO_POSITIONS`, log retention 7→30 days, and a 72h entry-drought webhook alert added.
- **v9.5 — universe expansion + funding rates**: Default universe now `BTC/USDC, ETH/USDC, SOL/USDC, AVAX/USDC, DOT/USDC, LINK/USDC, XRP/USDC` (USDC quote matches common live setup; SHIB dropped per v9.4 per-asset analysis). Correlation groups regrouped: `btc_core`, `eth_core`, `l1_alt`, `defi`, `payments`, `meme`. New `max_total_crypto_positions=5` portfolio-wide backstop prevents the regrouping from accidentally allowing 7+ correlated positions. Optional `FundingRateFetcher` pulls perpetual funding from a secondary exchange (Binance) as a sentiment proxy; logged to `market_snapshots` for future composite integration.

## Architecture

- **The Oracle (Sentiment)** (`sentiment.py`): LiteLLM-backed chat model (default Gemini Flash) for market sentiment and consensus odds; per-ticker TTL cache. Advisory only — can veto on BEARISH, never authorizes or places trades. Polymarket Gamma API for real prediction market probabilities. CoinMarketCap Fear & Greed Index as contrarian signal. Shared circuit breaker with 5-minute auto-close.
- **The Brain (Strategy)** (`strategy.py`): Algorithmic weighted-composite scoring engine with regime-aware modifiers and multi-timeframe alignment. Thresholds and weights configurable via `Settings`. No LLM on the decision path — all buy/sell/hold signals are rule-based.
- **The Challenger** (`llm_challenger.py`): Reads every BUY/SELL decision the algorithm makes and produces a structured `Critique(concern_level, thesis, concerns)`. ToS-safe — the LLM provides commentary, never trade instructions. Default off.
- **The Regime Detector** (`regime.py`): Rule-based market regime classifier (trending/ranging/breakout) that adjusts strategy behavior, entry thresholds, and stop-loss modes dynamically.
- **The Stop Manager** (`stop_manager.py`): Per-position stop management with three modes (Fixed, Trailing, Trailing+Breakeven) and time-based ROI exits. The v10 coordinator does not execute partial exits pending exact protection resizing.
- **The Guardian (Risk)** (`risk_guardian.py`): Pre-trade gates — liquidity, correlation limits, daily loss circuit breaker, position aging. Conviction-tiered position sizing with a 40%-of-balance concentration cap and intra-cycle budget to prevent multi-entry stacking.
- **The Decision Engine** (`decision_engine.py`): Orchestrates per-ticker exit and entry flows — strategy → guardian → challenger → executor.
- **The Executor** (`order_executor.py`, `execution_coordinator.py`): Exact maker-order submission, recovery, fills, exits, and protection orders. DCA and partial exits are blocked pending exact protection resizing.
- **The Pipeline** (`data_pipeline.py`): Parallel fetch of price, indicators, sentiment, regime, volatility, forecast, funding rates.
- **The Portfolio Manager** (`portfolio_manager.py`): Ticker classification (owned vs. available).
- **The Analyst (Quant)** (`indicators.py`, `volatility.py`, `forecast.py`, `analytics.py`, `signal_analytics.py`): Technical indicators (RSI, MACD, Bollinger Bands, ATR, ADX, ROC, OBV, EMA 21/50), GARCH volatility modeling, ARIMA price forecasting, portfolio analytics, per-asset breakdown, MAE/MFE, factor attribution.
- **The Optimizer** (`optimizer.py`, `walk_forward.py`): Optuna + rolling walk-forward windows. Objective wires sampled params into `CompositeStrategy` via a factory hook so every trial actually uses the sampled parameters.
- **The Proxy (WARP)**: Cloudflare WARP sidecar for Polymarket access from geo-restricted networks.

### Thread Safety

v10 collects market data in parallel via `ThreadPoolExecutor`. All shared state is protected:
- `TradeJournal` — SQLite access serialized via `threading.Lock`
- `CachedExchange` — thread-safe `get_or_compute()` pattern with lock
- `LLMClient` (shared by `SentimentOracle` and `LLMChallenger`) — circuit breaker counter protected by lock

### Modules

| Module | Role |
|---|---|
| `src/main.py` | Boot, logging setup, cycle orchestrator |
| `src/run_cycle` (in `main.py`) | Fast/slow dual cycle wiring; delegates to decision_engine |
| `src/decision_engine.py` | Entry/exit evaluation, strategy + guardian + challenger integration |
| `src/risk_guardian.py` | Liquidity / exposure / correlation / daily loss gates, conviction-tiered sizing |
| `src/order_executor.py` | Sell, partial sell, DCA, protection orders, fill reconciliation |
| `src/data_pipeline.py` | Parallel fetch of price, indicators, sentiment, regime, volatility, forecast, funding |
| `src/portfolio_manager.py` | Ticker classification (owned vs. available) |
| `src/strategy.py` | Regime-aware weighted-composite strategy engine (no LLM). Thresholds + weights from Settings |
| `src/regime.py` | Market regime classifier: TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT |
| `src/stop_manager.py` | Trailing stops, breakeven lock, time-based ROI exits, partial position exits |
| `src/sentiment.py` | Sentiment oracle (LiteLLM-backed), Fear & Greed, Polymarket |
| `src/llm_client.py` | Provider-agnostic LiteLLM wrapper: circuit breaker, fallback model, key redaction in logs |
| `src/llm_challenger.py` | ToS-safe decision review — commentary only, never blocks execution |
| `src/funding.py` | Perpetual funding-rate fetcher (secondary read-only exchange via CCXT) |
| `src/exchange.py` | Kraken integration via CCXT (orders, OHLCV, balances) |
| `src/indicators.py` | Technical analysis: RSI(14), MACD(12/26/9), BB(20,2), ATR(14), ADX(14), ROC(12), OBV, SMA(7), EMA(21/50) |
| `src/volatility.py` | GARCH(1,1) volatility modeling with regime classification |
| `src/forecast.py` | Auto-ARIMA price forecasting with 95% confidence intervals |
| `src/analytics.py` | Portfolio metrics: win rate, P&L, drawdown, daily Sharpe ratio, tax |
| `src/signal_analytics.py` | Per-asset stats, MAE/MFE, factor attribution |
| `src/execution_store.py` | Authoritative scoped v10 orders, fills, positions, and protections |
| `src/journal.py` | Derived analytics journal in `analytics-v10.db`; never execution authority |
| `src/health.py` | Atomic structured lifecycle publication and offline health CLI |
| `src/cache.py` | Thread-safe per-cycle caching proxy for exchange API calls |
| `src/models.py` | Shared types: MarketData, TradeSignal, MarketRegime, StopMode, PositionState, MultiTimeframeContext |
| `src/notifier.py` | Discord/Slack webhook notifications |
| `src/paper_portfolio.py` | Virtual portfolio for paper trading mode |
| `src/backtest.py` | Backtesting framework with live-strategy (`CompositeStrategy`) and factory hook for optimizer |
| `src/walk_forward.py` | Rolling train/test window optimizer with OOS Sharpe aggregation |
| `src/optimizer.py` | Optuna-based parameter optimization (single-split + walk-forward modes) |
| `src/config.py` | Pydantic-based configuration management |

### Strategy Weights (v9.7)

| Signal Source | Weight | Notes |
|---|---|---|
| Technical (RSI, MACD, BB, OBV) | 34% | Price-derived, most reliable |
| Sentiment (LLM) | 19% | Continuous 1-10 scoring |
| Polymarket | 15% | Prediction market implied probabilities |
| Momentum (ROC + ADX) | 10% |  |
| Consensus (LLM) | 9% | Narrative/news aggregation |
| Forecast (ARIMA) | 5% | Weak for 1h crypto; FLAT threshold 0.3% |
| Fear & Greed (contrarian) | 5% | Extreme fear = buy, extreme greed = sell |
| Funding rate (contrarian) | 3% | v9.7 — negative perp funding = bullish; active only with `FUNDING_RATES_ENABLED` |

Weights validated at import time (`sum == 1.0`). Active-weight normalization redistributes missing-data weight to active components. Optimizer can override per-trial via `Strategist(weights_override=...)`.

### Conviction-Tiered Position Sizing (v9.2)

Position size scales with composite score:

| Composite | Multiplier | Interpretation |
|---|---|---|
| < 0.25 | rejected | Strategy's entry gate already refuses |
| [0.25, 0.35) | 0.6× | Marginal edge — small position |
| [0.35, 0.50) | 1.0× | Normal conviction (v9.1 flat behaviour) |
| [0.50, 0.70) | 1.4× | Strong conviction — press the edge |
| ≥ 0.70 | 2.0× | Extreme conviction — full size |

Safety gates (in order of application):
1. **Per-position concentration cap**: `MAX_POSITION_PCT_OF_BALANCE` = 40% default
2. **Absolute hard cap**: `MAX_POSITION_SIZE_USD` = $50 default, ceiling $500
3. **Intra-cycle budget**: tracked across entries so two tier-4 signals in one cycle can't double-spend the cached balance
4. **DCA status**: v10 execution disables DCA until protection quantities can be resized exactly

Set `CONVICTION_SIZING_ENABLED=false` to restore v9.1 flat behaviour.

### Risk Controls

| Control | Description |
|---|---|
| Conviction sizing | Tiered multiplier (0.6× / 1.0× / 1.4× / 2.0×) based on composite score |
| Concentration cap | No single position exceeds 40% of total balance (`MAX_POSITION_PCT_OF_BALANCE`) |
| Total position cap | Max 5 concurrent crypto positions across all correlation groups (`MAX_TOTAL_CRYPTO_POSITIONS`) |
| Intra-cycle budget | Prevents two entries in one cycle from double-spending cached balance |
| Trailing stop-loss | SL ratchets up with price (1.5×ATR below high watermark). Never moves down |
| Breakeven lock | After +1×ATR profit, SL moves to entry price (capital protection) |
| Time-based ROI | Profit targets decay over time: 4% at 0 min, 3% at 60 min, 2% at 120 min, 1% at 240 min |
| Partial exits | Disabled in v10 execution pending exact protection resizing |
| ATR-based stops | SL = entry − 2×ATR, TP = entry + 3×ATR (fallback for FIXED mode) |
| SL floor clamp | Stop-loss never exceeds 50% below entry (protects low-price assets) |
| Regime-aware stops | Trailing in trends, breakeven in downtrends, fixed in ranges |
| TRENDING_DOWN block | Hard-blocks all new entries in TRENDING_DOWN regime |
| Volatility gating | HIGH vol requires confidence >0.45, EXTREME >0.65 |
| Mean reversion bonus | +0.10 composite boost when RSI<28 AND ADX<30 in RANGING regime |
| Per-group correlation limit | Max 3 positions in same correlation group (`MAX_CORRELATED_POSITIONS`) |
| Daily loss limit | Pauses new entries if daily realized losses exceed $20 (`DAILY_LOSS_LIMIT_USD`) |
| Position aging | Forces exit review after 120h with <2% P&L |
| Per-cycle entry limit | Max 2 new positions per 15-min cycle (`MAX_ENTRIES_PER_CYCLE`) |
| Regulatory circuit breaker | Aborts entry if "DELISTED", "BANNED" detected in sentiment |
| LLM circuit breaker | Skips primary LLM after 3 consecutive failures; auto-closes after 5 minutes |
| Stale order cleanup | Cancels unfilled limit orders after 2 hours |
| LLM Challenger | Observational second opinion — HIGH concern triggers a notifier alert (never blocks execution) |

### Market Regimes

| Regime | Detection | Strategy Adjustment |
|---|---|---|
| TRENDING_UP | ADX > 25, EMA21 > EMA50, ROC > 0 | 10% composite boost, trailing stops, easier entry threshold |
| TRENDING_DOWN | ADX > 25, EMA21 < EMA50, ROC < 0 | **Hard block on new entries** (v8.2); breakeven lock on existing |
| RANGING | ADX < 20 | 5% boost (mean reversion), fixed stops, easier entry |
| BREAKOUT | BB%B > 1 or < 0, |OBV slope| > 0.3 | 10% penalty (wait for confirmation), breakeven lock |

### Correlation Groups (v9.5)

Assets in the same group count toward the per-group `MAX_CORRELATED_POSITIONS=3` limit. A portfolio-wide `MAX_TOTAL_CRYPTO_POSITIONS=5` backstop caps total concurrent exposure regardless of group.

| Group | Assets |
|---|---|
| `btc_core` | BTC, LTC |
| `eth_core` | ETH |
| `l1_alt` | SOL, AVAX, DOT |
| `defi` | LINK |
| `payments` | XRP |
| `meme` | DOGE, SHIB |

Assets not in the table get their own unique group (no correlation limit, but still subject to total cap).

## Setup

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your keys.
   ```bash
   cp .env.example .env
   ```
   *Required keys:*
   - `KRAKEN_API_KEY` & `KRAKEN_PRIVATE_KEY` (Permissions: Query Funds, Modify Orders. **NO Withdrawal**).

   *Optional keys:*
   - `GEMINI_API_KEY` - required only when an enabled model uses Gemini.
   - `CMC_API_KEY` — CoinMarketCap Fear & Greed Index.
   - `XAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — enable LLM Challenger and/or a sentiment fallback on a different provider (recommended: `SENTIMENT_FALLBACK_MODEL=xai/grok-4-1-fast-reasoning`).
   - `WEBHOOK_URL` / `ERROR_WEBHOOK_URL` — Discord/Slack notifications.

4. **Run in PAPER mode**:
   ```bash
   PYTHONPATH=src uv run python src/main.py
   ```

   LIVE requires the account identifier, acknowledgement, and explicit entry
   switch documented in [deploy/qnap/README.md](deploy/qnap/README.md). Validate
   the exact candidate under representative PAPER load before promotion.

5. **Run Backtester** (standalone):
   ```bash
   PYTHONPATH=src uv run python src/backtest.py
   ```

6. **Run Optimizer** (standalone):
   ```bash
   # Single 70/30 split (fast)
   PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --trials 50

   # Walk-forward (honest out-of-sample metrics)
   PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --walk-forward \
       --trials 50 --train-size 1440 --test-size 480 --step 480
   ```

## Configuration

All settings can be overridden via environment variables or `.env`.

### Core

| Setting | Default | Description |
|---|---|---|
| `TRADING_MODE` | `PAPER` | `PAPER` or `LIVE` |
| `TRADING_UNIVERSE` | `BTC/USDC,ETH/USDC,SOL/USDC,AVAX/USDC,DOT/USDC,LINK/USDC,XRP/USDC` | Comma-separated trading pairs |
| `QUOTE_CURRENCY` | `USDC` | Quote currency for balances |
| `MAX_POSITION_SIZE_USD` | `50.0` | Hard cap per position (ceiling raised to $500 in v9.2) |
| `SENTIMENT_ENABLED` | `false` | Enable paid LLM sentiment calls; disabled is free and deterministic |
| `LLM_LIVE_INFLUENCE_ENABLED` | `false` | Separate LIVE opt-in; otherwise enabled LLM output remains shadow-only |
| `SENTIMENT_MODEL` | `gemini/gemini-3.5-flash` | Sentiment/consensus model, LiteLLM format (bare legacy names auto-prefix `xai/`) |
| `SENTIMENT_FALLBACK_MODEL` | *(unset)* | Fallback model on a different provider — strongly recommended |
| `SENTIMENT_CACHE_SECONDS` | `3600` | Per-ticker TTL cache for sentiment+consensus LLM calls (0 disables) |
| `LLM_MONTHLY_BUDGET_USD` | `1.0` | Persistent monthly cap; also bounds monthly shadow spend |
| `LLM_DAILY_SHADOW_BUDGET_USD` | `0.05` | Persistent UTC daily runtime cap |
| `LLM_MAX_TOKENS` | `512` | Per-call output-token cap |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-call timeout in seconds |
| `REQUIRE_BULLISH_SENTIMENT` | `false` | `true` restores the legacy hard gate (LLM outage disables all entries) |
| `NO_ENTRY_ALERT_HOURS` | `72` | Webhook alert when a LIVE bot opens nothing for N hours (0 disables) |

### Cycle Timing

| Setting | Default | Description |
|---|---|---|
| `ENTRY_CYCLE_SECONDS` | `900` | Fast cycle: stop management, entry timing (15 min) |
| `FULL_CYCLE_SECONDS` | `3600` | Full cycle: sentiment, polymarket, GARCH analysis (1 hour) |

### Strategy Thresholds (v9.0 — all configurable via .env)

| Setting | Default | Description |
|---|---|---|
| `ENTRY_THRESHOLD` | `0.25` | Min composite to open a new position |
| `EXIT_THRESHOLD` | `-0.05` | Bearish composite threshold for strategic exit |
| `DCA_THRESHOLD` | `0.30` | Min composite for DCA (stricter than entry) |
| `STRONG_SELL_THRESHOLD` | `-0.25` | Composite below this triggers cut-loss SELL |
| `PROFIT_TAKE_PCT` | `5.0` | Profit % that triggers momentum-weakness exit |
| `MOMENTUM_WEAK_THRESHOLD` | `0.1` | Composite below this + profit >`PROFIT_TAKE_PCT` → lock in gains |
| `SMA_OVEREXTENDED_PCT` | `15.0` | Block entry if price >N% above SMA-7 |
| `MEAN_REVERSION_RSI` | `28.0` | RSI below this = oversold trigger |
| `MEAN_REVERSION_ADX` | `30.0` | ADX below this = weak trend (mean-reversion eligible) |
| `MEAN_REVERSION_BONUS` | `0.10` | Composite bonus when mean-reversion conditions met |
| `STALE_POSITION_HOURS` | `120.0` | Force exit review if position older than N hours |
| `STALE_EXIT_PNL_THRESHOLD_PCT` | `2.0` | Only force-exit stale if |P&L| below this |
| `MIN_BUY_CONFIDENCE` | `0.0` | Guardian-level floor above strategy threshold (opt-in) |
| `MIN_SELL_CONFIDENCE` | `0.0` | Guardian-level floor above strategy threshold (opt-in) |

### Conviction Sizing (v9.2)

| Setting | Default | Description |
|---|---|---|
| `CONVICTION_SIZING_ENABLED` | `true` | Scale position size by composite score tier |
| `MAX_POSITION_PCT_OF_BALANCE` | `0.40` | Per-position concentration cap (5%–100%) |
| `SIZING_TIER_1_THRESHOLD` | `0.25` | Tier-1 lower bound (tiers must be ascending) |
| `SIZING_TIER_2_THRESHOLD` | `0.35` |  |
| `SIZING_TIER_3_THRESHOLD` | `0.50` |  |
| `SIZING_TIER_4_THRESHOLD` | `0.70` |  |
| `SIZING_TIER_1_MULTIPLIER` | `0.6` | Weak conviction sizing |
| `SIZING_TIER_2_MULTIPLIER` | `1.0` | Normal (v9.1 behaviour) |
| `SIZING_TIER_3_MULTIPLIER` | `1.4` | Strong conviction |
| `SIZING_TIER_4_MULTIPLIER` | `2.0` | Full size |

### LLM Challenger (v9.1 — opt-in)

| Setting | Default | Description |
|---|---|---|
| `CHALLENGER_ENABLED` | `false` | Enable per-decision LLM review (commentary only) |
| `CHALLENGER_MODEL` | `anthropic/claude-sonnet-4-5` | Primary challenger model (LiteLLM format) |
| `CHALLENGER_FALLBACK_MODEL` | `gemini/gemini-3.5-flash` | Fallback model if primary fails |
| `SENTIMENT_FALLBACK_MODEL` | *(unset)* | Fallback for Grok sentiment if primary fails |
| `ANTHROPIC_API_KEY` | *(unset)* | Required when using `anthropic/*` models |
| `GEMINI_API_KEY` | *(unset)* | Required when using `gemini/*` models |
| `OPENAI_API_KEY` | *(unset)* | Required when using `openai/*` models |

### Funding Rates (v9.5 — opt-in)

| Setting | Default | Description |
|---|---|---|
| `FUNDING_RATES_ENABLED` | `false` | Fetch perp funding from secondary read-only exchange |
| `FUNDING_RATES_EXCHANGE` | `binance` | CCXT exchange id for funding lookup (all symbols map to `/USDT:USDT` perps) |

### Risk Management

| Setting | Default | Description |
|---|---|---|
| `USE_ATR_STOPS` | `true` | Use ATR-based dynamic SL/TP |
| `ATR_SL_MULTIPLIER` | `2.0` | Stop-loss distance in ATR multiples |
| `ATR_TP_MULTIPLIER` | `3.0` | Take-profit distance in ATR multiples |
| `TRAILING_STOP_ENABLED` | `true` | Enable trailing stop-loss |
| `TRAILING_SL_ATR_MULTIPLIER` | `1.5` | Trailing SL distance below high watermark |
| `BREAKEVEN_ATR_THRESHOLD` | `1.0` | Move SL to entry after this much ATR profit |
| `ROI_ENABLED` | `true` | Enable time-based ROI exits |
| `ROI_SCHEDULE` | `0:4.0,60:3.0,120:2.0,240:1.0` | Time-decaying profit targets (min:pct) |
| `PARTIAL_EXIT_ENABLED` | `true` | Legacy strategy knob; v10 coordinator blocks partial execution pending exact protection resizing |
| `PARTIAL_EXIT_PCT` | `0.50` | Percentage of position to sell (50%) |
| `PARTIAL_EXIT_ATR_THRESHOLD` | `2.0` | Trigger partial exit after 2×ATR profit |
| `STOP_LOSS_PCT` | `0.02` | Fallback fixed SL (2%) when ATR unavailable |
| `TAKE_PROFIT_PCT` | `0.04` | Fallback fixed TP (4%) when ATR unavailable |
| `MAX_SPREAD_PCT` | `0.005` | Max spread to enter (0.5%) |
| `LIMIT_ORDER_DISCOUNT_PCT` | `0.002` | Limit buy discount below market (0.2%) |
| `MAX_CORRELATED_POSITIONS` | `3` | Max positions in same correlation group |
| `MAX_TOTAL_CRYPTO_POSITIONS` | `5` | Portfolio-wide cap across all groups (v9.5) |
| `DAILY_LOSS_LIMIT_USD` | `20.0` | Pause entries if daily losses exceed this |
| `POSITION_MAX_AGE_HOURS` | `120` | Review stale positions after N hours |
| `MAX_DCA_PER_POSITION` | `1` | Legacy limit; v10 coordinator blocks DCA pending exact protection resizing |
| `MAX_ENTRIES_PER_CYCLE` | `2` | Max new positions per 15-min cycle |
| `STALE_ORDER_MAX_AGE_SECONDS` | `7200` | Cancel unfilled orders after N seconds |
| `DYNAMIC_RISK_ENABLED` | `false` | Apply GARCH vol multipliers to SL/TP |

### Multi-Timeframe Analysis

| Setting | Default | Description |
|---|---|---|
| `MTF_ENABLED` | `true` | Enable multi-timeframe analysis |
| `MTF_SHORT_TIMEFRAME` | `15m` | Short timeframe for entry timing |
| `MTF_LONG_TIMEFRAME` | `4h` | Long timeframe for trend context |

### Technical Analysis

| Setting | Default | Description |
|---|---|---|
| `TA_CANDLE_TIMEFRAME` | `1h` | Timeframe for primary technical indicators |
| `TA_CANDLE_LIMIT` | `100` | Number of candles for indicators |
| `GARCH_ENABLED` | `true` | Enable GARCH volatility modeling |
| `GARCH_CANDLE_TIMEFRAME` | `4h` | Timeframe for GARCH model |
| `ARIMA_ENABLED` | `true` | Enable ARIMA price forecasting |
| `POLYMARKET_ENABLED` | `true` | Enable Polymarket prediction market odds |
| `WARP_PROXY` | *(unset)* | SOCKS5 proxy for Polymarket |

### Other

| Setting | Default | Description |
|---|---|---|
| `TAX_RATE` | `0.19` | Tax rate on realized profits (19% PL) |
| `WEBHOOK_URL` | *(unset)* | Discord/Slack webhook for trade notifications |
| `ERROR_WEBHOOK_URL` | *(unset)* | Separate webhook for errors |
| `PAPER_STATE_DB_PATH` | `logs/paper-execution-v10.db` | Authoritative PAPER execution store |
| `LIVE_STATE_DB_PATH` | `logs/live-execution-v10.db` | Authoritative LIVE execution store |
| `ANALYTICS_DB_PATH` | `logs/analytics-v10.db` | Derived analytics journal; not execution authority |
| `LLM_BUDGET_DB_PATH` | `logs/llm-budget.db` | Persistent LLM reservation and spend ledger |
| `HEALTHCHECK_FILE` | `logs/healthcheck` | Atomic structured JSON health document |
| `LOG_RETENTION_DAYS` | `30` | Daily log rotation retention (v9.6: 7 destroyed incident evidence) |

## Docker Deployment

The local stack runs the PAPER-only bot and a Cloudflare WARP sidecar. The
image healthcheck runs `python /app/src/health.py` against the configured
structured health document. An `unhealthy` status is an operator signal:
Docker does not restart a running process solely because a healthcheck fails.
`restart: unless-stopped` applies when the process exits.

| Service | Image | Purpose |
|---|---|---|
| `oracle-pythia` | `oracle-pythia:dev` | Locally built PAPER trading bot |
| `warp-proxy` | `ghcr.io/mon-ius/docker-warp-socks:v6` | SOCKS5 proxy for Polymarket (internal port 9091) |

Only Polymarket API calls route through the WARP proxy. All other traffic (Kraken, Grok/LiteLLM providers, CMC) uses direct connections.

### Quick Start (Docker)

```bash
# 1. Clone the repo
git clone https://github.com/krzysztofbury/pythia.git
cd pythia

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start in paper mode first (default)
docker compose up -d

# 4. Check logs
docker logs -f oracle-pythia

# This local Compose service is forced to PAPER.
# Use the reviewed candidate/stable QNAP flow for LIVE.
```

### Step-by-Step Docker Installation

#### Prerequisites

- Docker Engine 24+ and Docker Compose v2+
- Kraken account with API key (Query Funds + Modify Orders permissions only — **never enable Withdrawal**)
- A provider API key only if an optional LLM feature is enabled

#### 1. Clone and Configure

```bash
git clone https://github.com/krzysztofbury/pythia.git
cd pythia
cp .env.example .env
```

Edit `.env` with your editor:

```bash
# Required
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_PRIVATE_KEY=your_kraken_private_key

# Start in PAPER mode to validate
TRADING_MODE=PAPER

# Quote + universe (USDC is the v9.5 default)
QUOTE_CURRENCY=USDC
TRADING_UNIVERSE=BTC/USDC,ETH/USDC,SOL/USDC,AVAX/USDC,DOT/USDC,LINK/USDC,XRP/USDC

# Set position size (hard cap ceiling $500 in v9.2; default 50)
MAX_POSITION_SIZE_USD=50.0

# Optional paid LLM features remain shadow-only unless separately armed
# SENTIMENT_ENABLED=true
# CHALLENGER_ENABLED=true
# ANTHROPIC_API_KEY=your_anthropic_key
```

Secure the file:

```bash
chmod 600 .env
```

#### 2. Build and Start

```bash
docker compose up -d
```

#### 3. Verify

```bash
docker logs -f oracle-pythia
# Expect:
# Oracle Pythia v10 booting mode=PAPER
# Starting FULL cycle ... mode=PAPER
```

The structured health document carries `app_version=10.0.0`, the image
`BUILD_VERSION`, lifecycle state, freshness, and cycle deadline.

#### 4. Monitor

```bash
# Live logs
docker logs -f oracle-pythia

# Exact offline health command used by the image HEALTHCHECK
docker compose exec oracle-pythia \
  /app/.venv/bin/python /app/src/health.py /app/logs/healthcheck \
  --max-age-seconds 4500
```

#### 5. Switch to Live Trading

The local Compose definition intentionally overrides `TRADING_MODE=PAPER` and
`LIVE_ENTRIES_ENABLED=false`. Follow the candidate, production approval,
stable-digest promotion, and four-part LIVE arming procedure in
[deploy/qnap/README.md](deploy/qnap/README.md).

### Logs & Data

| File | Description |
|---|---|
| `logs/bot.log` | Main bot log (time-rotated daily, 30-day retention) |
| `logs/paper-execution-v10.db` | Authoritative PAPER execution store |
| `logs/live-execution-v10.db` | Authoritative LIVE execution store, isolated from PAPER |
| `logs/analytics-v10.db` | Derived trades, signals, snapshots, and optimization analytics |
| `logs/llm-budget.db` | Persistent paid-LLM budget ledger |
| `logs/paper-exchange-state.json` | Atomic PAPER venue checkpoint |
| `logs/healthcheck` | Atomic structured health JSON checked by Docker |

The selected v10 execution store is the sole local authority for orders, fills,
positions, and protections. A reviewed analytics rebuild command is future
work; no operator rebuild command exists today. Do not infer positions or order
state from the analytics `trades` table.

A v9 `trades.db` is a read-only legacy archive, never v10 execution state. Keep
it outside the writable runtime state volume; the QNAP deployment exposes it
only at `/archive/v9:ro` and does not configure v10 to open it.

### Candidate and Stable

`Publish candidate` runs quality checks on `main`, builds `linux/amd64` once,
and publishes `candidate` plus a one-time `sha-<git-sha>` tag. Its registry
digest is the authority; publication refuses to overwrite an existing SHA tag.
`Promote stable` verifies that digest against the tested SHA tag and retags the
same manifest as `stable` without checking out or rebuilding source.

Watchtower is optional and explicitly profiled on QNAP. It provides neither a
health-gated rollout nor automatic rollback; see the retained-image and manual
digest rollback procedure in the QNAP guide.

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/ --ignore-missing-imports

# Run optimizer (single-split)
PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --trials 50

# Run optimizer (walk-forward)
PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --walk-forward

# Run P&L reconciliation migration
PYTHONPATH=src uv run python src/migrate_reconcile_pnl.py --dry-run
```

### Design Docs

Per-release design specifications live in `docs/superpowers/specs/`:

- `2026-04-18-v9.0-refactor-design.md` — main.py split, threshold-to-config, MIN_CONFIDENCE cleanup
- `2026-04-18-v9.1-litellm-challenger-design.md` — LiteLLM migration + LLM Challenger
- `2026-04-18-v9.2-conviction-sizing-design.md` — Conviction-tiered sizing + concentration cap
- `2026-04-18-v9.3-walkforward-design.md` — Walk-forward optimizer + live-strategy backtest
- `2026-04-18-v9.4-analytics-design.md` — Per-asset stats + MAE/MFE + factor attribution
- `2026-04-18-v9.5-universe-funding-design.md` — Universe expansion + funding rates

### Pre-commit Hooks

```bash
uv tool install pre-commit
pre-commit install
```

Hooks run automatically on commit: ruff linting/formatting, trailing whitespace, secret detection.

## License

MIT License. See [LICENSE](LICENSE) for details.
