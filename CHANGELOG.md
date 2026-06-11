# Changelog

All notable changes to Oracle Pythia are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [9.7.0] — 2026-06-11

### Added

- **Funding rate wired into the composite** (TODO #1): new `W_FUNDING=0.03`
  component (technical 0.35→0.34, sentiment 0.20→0.19, consensus 0.10→0.09).
  Contrarian sign: negative funding scores bullish, saturating at ±0.1%/8h.
  Active only when `FUNDING_RATES_ENABLED` supplies data — absent data
  redistributes the weight, so default behaviour is unchanged. Factor
  attribution and the Optuna optimizer sample the new `fund=` component.

### Fixed

- **Failed sells no longer orphan positions**: `execute_sell` returns success;
  StopManager/strategic-exit/take-profit paths delete `position_state` only
  when the order was actually placed. Previously a Kraken error during a stop
  sell dropped all stop tracking while the coins were still held.
- **Fill-reconciliation double counting**: dedup window widened 15→180 min
  (maker orders live up to 2h before stale cleanup — fills landing later were
  journaled on top of the placement row, inflating cost basis; observed live
  on 2026-04-22 LINK). Sub-$1 partial-fill residues are skipped entirely.
- **`UPDATE` is not `UP`**: consensus direction parsing now uses word
  boundaries; prose like "UPDATE:" or "DOWNTREND" no longer false-positives a
  direction and (with a stray percentage) blocks or authorizes entries.
- **"Insufficient data" LLM responses count as unavailable**, excluded from
  the composite's effective weight (observed live: Gemini/Grok answer
  "Insufficient recent data available" when they cannot browse).

---

## [9.6.0] — 2026-06-11

Incident-driven release. The LIVE bot executed **zero trades from 2026-04-23 to
2026-06-11**: the entry gate hard-required BULLISH LLM sentiment, xAI credits
ran out on 2026-05-15 (and again 2026-06-10 after ~1,300 uncached reasoning
calls/day), and sub-$10 guardian allocations silently killed the BUY signals
that did fire. No alert ever triggered.

### Changed

- **Sentiment is now advisory, not a hard gate** (`strategy.py`): entries are
  blocked only on BEARISH sentiment or a strong DOWN consensus (>60%). The
  legacy require-BULLISH gate is available via `REQUIRE_BULLISH_SENTIMENT=true`.
  The LLM still never issues buy/sell instructions — it remains commentary on
  top of the rule-based composite (ToS-safe).
- **Primary sentiment model → `gemini/gemini-3.5-flash`** via new
  `SENTIMENT_MODEL` setting. `XAI_API_KEY` is now optional; `XAI_MODEL` is no
  longer wired (set `SENTIMENT_MODEL=xai/<model>` instead). Provider keys
  (Gemini/Anthropic/OpenAI/xAI) are passed through to LiteLLM.
- **NEUTRAL sentiment scores 0.0** in the composite (was −0.111 — the failure
  fallback string permanently dragged every composite down).
- **Unavailable sentiment/consensus excluded from effective weight**
  (`_sentiment_available`): failure fallbacks redistribute their weight like
  missing Polymarket data instead of masquerading as a neutral opinion.
- **Guardian no longer rejects sub-tier-1 entry composites**: regime modifiers
  can lower the strategy's entry threshold below the 0.25 tier-1 floor; such
  entries are sized at the tier-1 multiplier (DCAs still hard-reject). 447 BUY
  signals died this way during the incident.
- **`ROUND_UP_THRESHOLD` lowered 8.3 → 5.0** and the round-up now checks both
  balance and intra-cycle budget. On a ~$100 account with 7 slots, tier-1
  allocations (~$6-8) previously fell below Kraken's $10 minimum every time.
- **Entry slots capped by `max_total_crypto_positions`**
  (`compute_available_slots`): capital is no longer split across 7 universe
  slots when at most 5 positions may exist.
- **`LOG_RETENTION_DAYS` default 7 → 30** — the 7-day window destroyed the
  logs needed to diagnose the incident.
- `CHALLENGER_FALLBACK_MODEL` default bumped to `gemini/gemini-3.5-flash`.

### Added

- **Per-ticker TTL cache for sentiment + consensus LLM calls**
  (`SENTIMENT_CACHE_SECONDS`, default 3600). Cuts ~1,344 calls/day to ~350.
  Failures are never cached, so a degraded provider is retried next cycle.
- **Entry-drought alert** (`NO_ENTRY_ALERT_HOURS`, default 72): when a LIVE
  bot opens no position for N hours, a webhook error fires (throttled to once
  per 24h). Silence is a failure mode.
- `TradeJournal.get_last_entry_timestamp()` and an optional `timestamp`
  override on `log_trade` (tests/backfills).
- Boot config line now prints sentiment model, fallback, cache TTL, and gate
  mode.

---

## [9.5.0] — 2026-04-18

### Added

- **Universe expansion**: default `TRADING_UNIVERSE` now `BTC/USDC, ETH/USDC, SOL/USDC, AVAX/USDC, DOT/USDC, LINK/USDC, XRP/USDC` (7 liquid majors). All pairs verified live on Kraken. `QUOTE_CURRENCY` default switched `USDT → USDC` to match common live operator setup.
- **Perpetual funding-rate fetcher** (`src/funding.py`): `FundingRateFetcher` pulls perp funding from a secondary read-only CCXT client (Binance by default). All symbols map to `/USDT:USDT` perps for deepest coverage (funding is quote-agnostic). Opt-in via `FUNDING_RATES_ENABLED=true`.
- **`market_snapshots.funding_rate`** column (idempotent migration). Populated on every cycle when funding is enabled, persisted for future composite-integration analysis.
- **`MarketData.funding_rate`** field (informational only; not yet consumed by `Strategist.generate_signal`).
- **`max_total_crypto_positions: int = 5`**: portfolio-wide concurrent-position backstop. Prevents the v9.5 correlation-group refinement from accidentally allowing 7+ correlated-crypto positions. Applied before the per-group `max_correlated_positions` gate.
- **Boot-time effective-config log line**: prints mode, quote, universe, and feature flags at startup so deploys with unexpected `.env` state are visible immediately.

### Changed

- **Correlation groups** tightened/refined in `risk_guardian.CORRELATION_GROUPS`:
  - `btc_core = {BTC, LTC}` (was `btc_ecosystem`)
  - `eth_core = {ETH}` (broken out of the old ETH cluster)
  - `l1_alt = {SOL, AVAX, DOT}` (new L1-alts group)
  - `defi = {LINK}` (separated from ETH cluster)
  - `payments = {XRP}` (no longer `alt_major`)
  - `meme = {DOGE, SHIB}` (unchanged)
- **`SHIB` dropped** from default universe (poor historical edge per v9.4 per-asset analysis).

### Fixed

- **C3 (factor-attribution regex never matched real reasoning)**: `Strategist.generate_signal` now appends a machine-readable `tech=+X.XXX mom=+X.XXX ...` token string to `TradeSignal.reasoning`. Factor-attribution regex now finds real data. Reasoning column truncation bumped 300 → 600 chars to accommodate.
- **C2 (MAE/MFE paired SELL with DCA add, not original entry)**: `compute_mae_mfe` now finds the earliest BUY after the most recent prior SELL for the ticker, giving MAE/MFE vs. the position's true cost basis.
- **C3 (`DOT/USDC:USDC` perp does not exist on Binance)**: `FundingRateFetcher._to_perp` always maps to `/USDT:USDT` perps regardless of spot quote. Funding signal is quote-agnostic; USDT perps have the deepest cross-exchange coverage.
- **M3 (regex word boundaries)**: `_COMPONENT_PATTERN` anchored with `\b` so tokens embedded in other words can't false-positive.

---

## [9.4.0] — 2026-04-18

### Added

- **`src/signal_analytics.py`** — new module. `SignalAnalyzer` with:
  - `per_asset_metrics()`: aggregate SELL-side P&L, win rate, avg/best/worst per ticker.
  - `compute_mae_mfe(ticker=None, limit=50)`: Maximum Adverse / Favorable Excursion per round-trip trade, using `market_snapshots` price stream between entry and exit.
  - `factor_attribution()`: Pearson correlation between each composite component (technical, momentum, forecast, sentiment, consensus, polymarket, fear_greed) and realized P&L.
  - `run()` and `log_summary()` for one-shot reports.

---

## [9.3.0] — 2026-04-18

### Added

- **`src/walk_forward.py`** — `WalkForwardOptimizer`:
  - Rolling train/test windows (default 60d / 20d, step 20d) over a candle series.
  - `run(optimize_fn, evaluate_fn)` — callables injected so the module has no hard dependency on optuna.
  - Aggregates out-of-sample Sharpe, stability (stddev), overfit gap, worst-window Sharpe, aggregate return.
- **`Strategist.__init__(weights_override=...)`** — accepts a dict of per-component weights to replace `DEFAULT_WEIGHTS`. Required keys enforced; must sum to 1.0 ± 1e-9.
- **`CompositeStrategy._strategist_factory`** hook in `backtest.py`. When set, `CompositeStrategy.init()` constructs the strategist via the factory — the optimizer uses this to thread sampled parameters into each trial.
- **`BacktestResult`**: new `sortino_ratio`, `calmar_ratio`, `profit_factor` fields. Crypto returns have fat tails; Sharpe alone is misleading.
- **`StrategyOptimizer.walk_forward()`** CLI mode: `--walk-forward --trials N --train-size N --test-size N --step N`. Old single-split mode retained.

### Fixed

- **Optuna trials were evaluating identical strategies** (prior versions sampled parameters but called `BacktestRunner().run()` without threading the params into `CompositeStrategy`). Every trial now uses the sampled weights + entry threshold.

---

## [9.2.0] — 2026-04-18

### Added

- **Conviction-tiered position sizing** in `RiskGuardian.check_buy`:
  - `compute_conviction_multiplier(composite, settings) -> float` walks the tier thresholds.
  - Default tiers: `<0.25 reject`, `[0.25,0.35) 0.6×`, `[0.35,0.50) 1.0×`, `[0.50,0.70) 1.4×`, `≥0.70 2.0×`.
  - All tiers + multipliers operator-tunable via `SIZING_TIER_{1..4}_THRESHOLD` / `..._MULTIPLIER`.
- **40%-of-balance concentration cap** (`MAX_POSITION_PCT_OF_BALANCE=0.40`, range 5%–100%). No single position can exceed this fraction of total balance regardless of tier multiplier.
- **Intra-cycle `budget_usd`** parameter on `check_buy` + `evaluate_entry`. `run_cycle` tracks remaining balance across entries so two tier-4 signals in one cycle can't both allocate 40% of cached balance (the cache doesn't refresh between entries).
- **`is_dca=True` clamp**: DCA calls reduce multiplier to 1.0 to prevent tier amplification on existing positions.
- **`MAX_POSITION_SIZE_USD` ceiling raised** `le=100.0` → `le=500.0`. Default still $50; operators must explicitly opt into larger sizes.
- **`CONVICTION_SIZING_ENABLED: bool = True`** feature flag. Set `false` to restore v9.1 flat behaviour.

### Fixed

- **C1 (cached balance → 2× 40% in one cycle)**: `run_cycle` tracks `cycle_budget_usd` and decrements on successful entries; passed to `check_buy` as an additional ceiling.
- **C2 (DCA stacking on existing positions)**: `execute_dca` now calls `guardian.check_buy(..., is_dca=True)`.

---

## [9.1.0] — 2026-04-18

### Added

- **`src/llm_client.py`** — `LLMClient`: thin LiteLLM wrapper with circuit breaker (3 failures → open), 5-minute auto-close cooldown, automatic fallback to a secondary model on primary failure, API-key redaction in log/exception strings.
- **`src/llm_challenger.py`** — `LLMChallenger`:
  - Reviews every BUY/SELL decision; returns `Critique(concern_level, thesis, concerns)`.
  - **Never blocks execution** — commentary only, ToS-safe across providers.
  - `concern_level="HIGH"` fires `notifier.notify_error` as an alert.
  - Degrades gracefully to `Critique(concern_level="NONE")` on any failure.
- **`build_challenger_from_settings(settings)`** factory in `llm_challenger.py` — wires provider keys from `Settings`.
- **`journal.log_signal(llm_critique=...)`** kwarg + new `signals.llm_critique` column (idempotent migration).
- **`decision_engine._run_challenger_and_format`** helper — wraps `challenger.review` defensively so a buggy subclass that violates the no-raise contract cannot kill a trading cycle.
- **`SENTIMENT_FALLBACK_MODEL`**, **`CHALLENGER_*`** config fields. `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` optional secrets.

### Changed

- **`sentiment.py`** — `SentimentOracle` now dispatches through `LLMClient` instead of `xai_sdk` directly. Legacy bare model names (`grok-4-1-fast-reasoning`) auto-prefix with `xai/`. Public interface unchanged so `main.py` wiring is preserved.

### Removed

- Direct `xai_sdk` imports from `sentiment.py`. The LiteLLM abstraction handles Grok identically to any other provider.

---

## [9.0.0] — 2026-04-18

### Changed

- **`src/main.py` split from 1995 LOC into focused modules**:
  - `src/risk_guardian.py` — `RiskGuardian`, `check_correlation_limit`, `check_daily_loss_limit`, `CORRELATION_GROUPS`, `REGULATORY_DANGER_PHRASES`.
  - `src/data_pipeline.py` — parallel fetch of price, indicators, sentiment, regime, volatility, forecast, multi-timeframe context. `build_market_data`, `collect_all_market_data`.
  - `src/order_executor.py` — `execute_sell`, `execute_partial_sell`, `execute_dca`, `compute_protection_prices`, `check_take_profit`, `ensure_stop_loss`, `reconcile_stale_orders`, `reconcile_fills`.
  - `src/decision_engine.py` — `evaluate_entry`, `evaluate_exit`.
  - `src/portfolio_manager.py` — `classify_tickers`.
  - `src/main.py` now 359 LOC: boot, logging, `run_cycle` orchestrator only.

### Added

- **Strategy thresholds moved from module constants to `Settings`** (configurable via `.env`): `ENTRY_THRESHOLD`, `EXIT_THRESHOLD`, `STRONG_SELL_THRESHOLD`, `DCA_THRESHOLD`, `PROFIT_TAKE_PCT`, `MOMENTUM_WEAK_THRESHOLD`, `SMA_OVEREXTENDED_PCT`, `MEAN_REVERSION_RSI/ADX/BONUS`, `STALE_POSITION_HOURS`, `STALE_EXIT_PNL_THRESHOLD_PCT`, `MIN_BUY_CONFIDENCE`, `MIN_SELL_CONFIDENCE`.
- **`Strategist.__init__(settings=None)`** — reads thresholds from the passed `Settings` or falls back to module defaults. Module constants retained and guarded by an import-time drift assertion so tuning via `.env` cannot silently diverge from the test fallbacks.

### Removed

- **`MIN_CONFIDENCE = 0.10`** constant (a Gemini-era artifact where confidence ∈ [0,1] was a raw LLM score). Dead for the BUY path (entry_threshold 0.25 > 0.10 always) and unreachable on the SELL path (all SELL branches set confidence ≥ 0.15 by construction). Replaced with explicit `min_buy_confidence` / `min_sell_confidence` floors (default 0.0, opt-in).
- **Dead `RiskGuardian.check_exposure` method** (unused since v8; callers rely on `exchange.has_open_position` + `get_open_orders` directly).

### Fixed

- **Empty universe crashed `collect_all_market_data`** (`ThreadPoolExecutor(max_workers=min(0, 4))` raised `ValueError`). Now returns an empty dict on empty input.
- **Boot-log version string** updated to v9.0 (was pinned to v8.2).

---

## [8.2.0] — 2026-04-12

v8.2 delivered the P&L integrity fix and strategy-tuning overhaul tracked in `docs/superpowers/specs/2026-04-12-pnl-fix-strategy-tuning-design.md`:

- Kraken fill reconciliation (`fetch_my_trades` + `_reconcile_fills`): captures SL/TP fills and actual fees.
- `exchange_fill_id` column + `fee` column + `fill_source` ("bot" vs "exchange") on `trades` table.
- `compute_pnl_with_fees`: FIFO cost basis with proportional fee deduction.
- Strategy discipline: hard-block entries in TRENDING_DOWN, entry threshold 0.15 → 0.25, per-cycle limit (max 2 new positions), mean reversion gated to RANGING regime only, DOGE removed from default universe.

## [8.1.0] — 2026-03-26

Post-deployment tuning based on live performance data:

- Entry threshold 0.10 → 0.15 (too much noise at 0.10).
- Fear & Greed weight 0.10 → 0.05 (permanently bullish in fear markets without counterweight).
- Stale position timeout 72h → 120h (stale exits were the #1 loss source in v8.0).
- Gate entry on negative technicals (prevents buying overbought).
- Gate entry on MTF bearish conflict (stops buying into downtrends).

## [8.0.0] — 2026-03-26

"Professional upgrade":

- Market regime classifier (`regime.py`): TRENDING_UP/DOWN, RANGING, BREAKOUT.
- Stop manager (`stop_manager.py`): trailing SL, ROI decay, partial exits, breakeven lock.
- Multi-timeframe analysis: 15m entry timing, 1h primary, 4h trend context.
- Fear & Greed contrarian signal (CoinMarketCap).
- Fast/slow dual cycle: 15 min fast (stops + entries) / 1 h full (sentiment + GARCH + Polymarket).
- Optuna-based hyperparameter optimization (70/30 split, JSON output for human review).
- Thread-safe market-data collection via `ThreadPoolExecutor`.
- Active-weight normalization for missing signal components.

## Earlier

v1 – v7 history not reconstructed; see `git log --oneline` for the full record.
