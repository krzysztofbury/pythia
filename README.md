# Oracle Pythia v8.0

An autonomous crypto trading bot on Kraken, driven by algorithmic strategy (weighted composite scoring), Grok (Sentiment), Polymarket (Prediction Markets), quantitative analysis (Technical Indicators, GARCH Volatility, ARIMA Forecasting), and market regime detection.

## What's New in v8.0

- **Trailing Stop-Loss**: Dynamic SL that ratchets up with price. Breakeven lock mode moves SL to entry after 1xATR profit. Regime-aware mode selection (trailing in trends, fixed in ranges).
- **Time-Based ROI Decay**: Configurable profit targets that decrease over time (e.g., 4% at entry, 1% after 4 hours). Prevents holding winners too long.
- **Multi-Timeframe Analysis**: 15m candles for entry timing, 1h for primary signals, 4h for trend context. EMA(21/50) crossover detection on longer timeframes.
- **Market Regime Detection**: Rule-based classifier (TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT) using ADX, EMA crossover, ROC, Bollinger Bands, and OBV. Strategy adjusts weights, thresholds, and stop modes per regime.
- **Partial Position Exits**: Sell 50% at 2xATR profit, let the rest trail. Locks in gains while keeping upside exposure.
- **Fear & Greed Contrarian Signal**: Extreme Fear = buy opportunity, Extreme Greed = sell signal. 10% weight in composite scoring.
- **Fast/Slow Dual Cycle**: 15-minute fast cycle for stop management and entry timing, 1-hour full cycle for sentiment and deep analysis.
- **Hyperparameter Optimization**: Optuna-based parameter optimization with 70/30 train/validate split. Results saved to JSON for human review (never auto-applied).
- **Market Snapshot Persistence**: Every cycle saves full market context to SQLite for realistic backtesting with historical sentiment data.
- **Active Weight Normalization**: Missing data (no Polymarket, ARIMA FLAT) no longer silently wastes scoring weight — redistributed to active components.

## Architecture

- **The Oracle (Sentiment)**: Grok (xAI) for market sentiment (continuous 1-10 scoring) and consensus odds. Polymarket Gamma API for real prediction market probabilities. CoinMarketCap Fear & Greed Index as contrarian signal. Circuit breaker skips Grok after 3 consecutive failures.
- **The Brain (Strategy)**: Algorithmic weighted-composite scoring engine with regime-aware modifiers and multi-timeframe alignment. No LLM in the decision path — all buy/sell/hold signals are rule-based.
- **The Regime Detector**: Rule-based market regime classifier (trending/ranging/breakout) that adjusts strategy behavior, entry thresholds, and stop-loss modes dynamically.
- **The Stop Manager**: Per-position stop management with three modes (Fixed, Trailing, Trailing+Breakeven), time-based ROI exits, and partial position exits.
- **The Guardian (Risk)**: Hard-coded risk management — ATR-based dynamic stops (clamped at 50% max loss), correlation limits across asset groups, daily loss circuit breaker, position aging, spread checks, exposure limits.
- **The Executor**: CCXT (Kraken) with strict "Maker-Only" enforcement and automatic retries on transient errors.
- **The Analyst (Quant)**: Technical indicators (RSI, MACD, Bollinger Bands, ATR, ADX, ROC, OBV, EMA 21/50), GARCH volatility modeling, ARIMA price forecasting, portfolio analytics.
- **The Proxy (WARP)**: Cloudflare WARP sidecar for Polymarket access from geo-restricted networks.

### Thread Safety

v8.0 collects market data in parallel via `ThreadPoolExecutor`. All shared state is protected:
- `TradeJournal` — SQLite access serialized via `threading.Lock`
- `CachedExchange` — thread-safe `get_or_compute()` pattern with lock
- `SentimentOracle` — circuit breaker counter protected by lock

### Modules

| Module | Role |
|---|---|
| `src/main.py` | Main loop, Risk Guardian, trade execution, fast/slow dual cycle, stop management integration, market snapshots |
| `src/strategy.py` | Regime-aware weighted-composite strategy engine with multi-timeframe alignment (no LLM) |
| `src/regime.py` | Market regime classifier: TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT |
| `src/stop_manager.py` | Trailing stops, breakeven lock, time-based ROI exits, partial position exits |
| `src/sentiment.py` | Grok sentiment (continuous scoring), Fear & Greed, Polymarket, circuit breaker |
| `src/exchange.py` | Kraken integration via CCXT (orders, OHLCV, balances) |
| `src/indicators.py` | Technical analysis: RSI(14), MACD(12/26/9), BB(20,2), ATR(14), ADX(14), ROC(12), OBV, SMA(7), EMA(21/50) |
| `src/volatility.py` | GARCH(1,1) volatility modeling with regime classification |
| `src/forecast.py` | Auto-ARIMA price forecasting with 95% confidence intervals |
| `src/analytics.py` | Portfolio metrics: win rate, P&L, drawdown, daily Sharpe ratio, tax |
| `src/journal.py` | Thread-safe SQLite trade journal with position state tracking, market snapshots, and daily P&L |
| `src/cache.py` | Thread-safe per-cycle caching proxy for exchange API calls |
| `src/models.py` | Shared types: MarketData, TradeSignal, MarketRegime, StopMode, PositionState, MultiTimeframeContext |
| `src/notifier.py` | Discord/Slack webhook notifications |
| `src/paper_portfolio.py` | Virtual portfolio for paper trading mode |
| `src/backtest.py` | Backtesting framework with historical snapshot replay support |
| `src/optimizer.py` | Optuna-based hyperparameter optimization with train/validate split |
| `src/config.py` | Pydantic-based configuration management |

### Strategy Weights (v8.0)

| Signal Source | Weight | Notes |
|---|---|---|
| Technical (RSI, MACD, BB, OBV) | 30% | Price-derived, most reliable. Increased from 25%. |
| Sentiment (Grok) | 20% | Continuous 1-10 scoring. Reduced from 25% (hallucination risk). |
| Polymarket | 15% | Prediction market implied probabilities |
| Momentum (ROC + ADX) | 10% | Reduced from 15% — over-punished in sideways markets |
| Fear & Greed (contrarian) | 10% | NEW — extreme fear = buy, extreme greed = sell |
| Consensus (Grok) | 10% | Reduced weight — LLM consensus is noisy |
| ARIMA Forecast | 5% | Reduced from 10% — weak for 1h crypto. FLAT threshold widened to 0.3%. |

Weights are validated at import time (`assert sum == 1.0`). Active-weight normalization redistributes missing data weight to active components.

### Risk Controls

| Control | Description |
|---|---|
| Trailing stop-loss | SL ratchets up with price (1.5xATR below high watermark). Never moves down. |
| Breakeven lock | After +1xATR profit, SL moves to entry price (capital protection) |
| Time-based ROI | Profit targets decay over time: 4% at 0min, 3% at 60min, 2% at 120min, 1% at 240min |
| Partial exits | Sell 50% at 2xATR profit, let remaining 50% trail |
| ATR-based stops | SL = entry - 2xATR, TP = entry + 3xATR (fallback for FIXED mode) |
| SL floor clamp | Stop-loss never exceeds 50% below entry (protects low-price assets) |
| Regime-aware stops | Trailing in trends, breakeven in downtrends, fixed in ranges |
| Volatility gating | HIGH vol requires confidence >0.45, EXTREME >0.65 |
| Mean reversion bonus | +0.10 composite boost when RSI<28 AND ADX<30 (oversold in range) |
| Correlation limit | Max 3 positions in same asset group |
| Daily loss limit | Pauses new entries if daily realized losses exceed $20 |
| Position aging | Forces exit review after 72h with <2% P&L |
| Regulatory circuit breaker | Aborts entry if "DELISTED", "BANNED" detected in sentiment |
| Grok circuit breaker | Skips Grok API calls after 3 consecutive failures |
| Stale order cleanup | Cancels unfilled limit orders after 2 hours |

### Market Regimes

| Regime | Detection | Strategy Adjustment |
|---|---|---|
| TRENDING_UP | ADX > 25, EMA21 > EMA50, ROC > 0 | 10% composite boost, trailing stops, easier entry threshold |
| TRENDING_DOWN | ADX > 25, EMA21 < EMA50, ROC < 0 | 20% composite penalty, breakeven lock, harder entry threshold |
| RANGING | ADX < 20 | 5% boost (mean reversion), fixed stops, easier entry |
| BREAKOUT | BB%B > 1 or < 0, |OBV slope| > 0.3 | 10% penalty (wait for confirmation), breakeven lock |

### Correlation Groups

Assets in the same group count toward the `MAX_CORRELATED_POSITIONS` limit:

| Group | Assets |
|---|---|
| `btc_ecosystem` | BTC, LTC |
| `eth_ecosystem` | ETH, SOL, LINK, DOT |
| `meme` | DOGE, SHIB |
| `alt_major` | XRP |

Assets not in the table get their own unique group (no correlation limit).

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
   *Required Keys:*
   - `KRAKEN_API_KEY` & `KRAKEN_PRIVATE_KEY` (Permissions: Query Funds, Modify Orders. NO Withdrawal).
   - `XAI_API_KEY` (Grok).

   *Optional Keys:*
   - `CMC_API_KEY` (CoinMarketCap Fear & Greed Index).
   - `WEBHOOK_URL` / `ERROR_WEBHOOK_URL` (Discord/Slack notifications).

4. **Run**:
   ```bash
   # Paper Mode (Default)
   PYTHONPATH=src uv run python src/main.py

   # Live Mode
   TRADING_MODE=LIVE PYTHONPATH=src uv run python src/main.py
   ```

5. **Run Backtester** (standalone):
   ```bash
   PYTHONPATH=src uv run python src/backtest.py
   ```

6. **Run Optimizer** (standalone):
   ```bash
   PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --trials 200
   ```

## Configuration

All settings can be overridden via environment variables or `.env`:

### Core

| Setting | Default | Description |
|---|---|---|
| `TRADING_MODE` | `PAPER` | `PAPER` or `LIVE` |
| `TRADING_UNIVERSE` | `BTC/USDT,ETH/USDT,...` | Comma-separated trading pairs |
| `QUOTE_CURRENCY` | `USDT` | Quote currency for balances |
| `MAX_POSITION_SIZE_USD` | `50.0` | Hard cap per position (max 100) |
| `XAI_MODEL` | `grok-4-1-fast-reasoning` | Grok model for sentiment/consensus |

### Cycle Timing

| Setting | Default | Description |
|---|---|---|
| `ENTRY_CYCLE_SECONDS` | `900` | Fast cycle: stop management, entry timing (15 min) |
| `FULL_CYCLE_SECONDS` | `3600` | Full cycle: sentiment, polymarket, GARCH analysis (1 hour) |
| `TIMEOUT_SECONDS` | `3600` | Legacy: sleep between cycles |

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
| `PARTIAL_EXIT_ENABLED` | `true` | Enable partial position exits |
| `PARTIAL_EXIT_PCT` | `0.50` | Percentage of position to sell (50%) |
| `PARTIAL_EXIT_ATR_THRESHOLD` | `2.0` | Trigger partial exit after 2xATR profit |
| `STOP_LOSS_PCT` | `0.02` | Fallback fixed SL (2%) when ATR unavailable |
| `TAKE_PROFIT_PCT` | `0.04` | Fallback fixed TP (4%) when ATR unavailable |
| `MAX_SPREAD_PCT` | `0.005` | Max spread to enter (0.5%) |
| `LIMIT_ORDER_DISCOUNT_PCT` | `0.002` | Limit buy discount below market (0.2%) |
| `MAX_CORRELATED_POSITIONS` | `3` | Max positions in same correlation group |
| `DAILY_LOSS_LIMIT_USD` | `20.0` | Pause entries if daily losses exceed this |
| `POSITION_MAX_AGE_HOURS` | `72` | Review stale positions after N hours |
| `MAX_DCA_PER_POSITION` | `1` | Max additional DCA buys per position |
| `STALE_ORDER_MAX_AGE_SECONDS` | `7200` | Cancel unfilled orders after N seconds |
| `DYNAMIC_RISK_ENABLED` | `false` | Apply GARCH vol multipliers to SL/TP |

### Multi-Timeframe Analysis

| Setting | Default | Description |
|---|---|---|
| `MTF_ENABLED` | `true` | Enable multi-timeframe analysis |
| `MTF_SHORT_TIMEFRAME` | `15m` | Short timeframe for entry timing (RSI, BB) |
| `MTF_LONG_TIMEFRAME` | `4h` | Long timeframe for trend context (EMA crossover) |

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

### Optimizer

| Setting | Default | Description |
|---|---|---|
| `OPTIMIZED_PARAMS_FILE` | *(unset)* | Path to JSON file with optimized parameters (never auto-applied) |

### Other

| Setting | Default | Description |
|---|---|---|
| `TAX_RATE` | `0.19` | Tax rate on realized profits (19% PL) |
| `WEBHOOK_URL` | *(unset)* | Discord/Slack webhook for trade notifications |
| `ERROR_WEBHOOK_URL` | *(unset)* | Separate webhook for errors |
| `HEALTHCHECK_FILE` | `logs/healthcheck` | Path for Docker healthcheck file |
| `LOG_MAX_BYTES` | `10485760` | Max log file size before rotation (10MB) |
| `LOG_BACKUP_COUNT` | `5` | Number of rotated log files to keep |

## Docker

The bot runs as two containers: the trading bot and a Cloudflare WARP sidecar that proxies Polymarket traffic. Includes Docker HEALTHCHECK that auto-restarts if no cycle completes within 2 hours.

```bash
docker compose up -d
```

| Service | Image | Purpose |
|---|---|---|
| `oracle-pythia` | `uv:python3.13-bookworm-slim` | Trading bot |
| `warp-proxy` | `ghcr.io/mon-ius/docker-warp-socks:v5` | SOCKS5 proxy for Polymarket (port 9091) |

Only Polymarket API calls route through the WARP proxy. All other traffic (Kraken, Grok, CMC) uses direct connections.

### Logs & Data

| File | Description |
|---|---|
| `logs/bot.log` | Main bot log (rotating, 10MB) — the only file-based log |
| `logs/trades.db` | SQLite database — single source of truth for all structured data |
| `logs/healthcheck` | Healthcheck timestamp (used by Docker) |

All structured data lives in SQLite (`trades.db`):

| Table | Contents |
|---|---|
| `trades` | Trade journal (BUY/SELL/CANCEL records with P&L and tax) |
| `signals` | All trading signals with composite scores (replaces signals.jsonl) |
| `position_state` | Per-position stop tracking (high watermark, SL/TP, partial exits) |
| `market_snapshots` | Full market context per cycle for backtest replay |
| `optimization_runs` | Optuna optimization results (replaces optimization.json) |

## Development

- **Linting**:
  ```bash
  uv run ruff check .
  uv run mypy .
  ```
- **Tests** (137+ tests):
  ```bash
  PYTHONPATH=src uv run pytest
  ```
- **Optimizer**:
  ```bash
  PYTHONPATH=src uv run python src/optimizer.py --symbol BTC/USDC --trials 200
  ```
