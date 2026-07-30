# Oracle Pythia — Roadmap

Forward-looking work. For what's already shipped see [CHANGELOG.md](CHANGELOG.md).

---

## v10 Pre-LIVE Gates — MUST fix before arming LIVE

Findings from the 2026-07-24 adversarial review of the v10 rewrite. All are
fail-closed today (they wedge readiness, never corrupt money state), which is
acceptable on the PAPER candidate channel but not for LIVE operation.

1. **Issue-lifecycle remediation path.** `HALT_UNKNOWN_ORDER_FILL` and
   `INVALID_OPEN_ORDER` are recorded but have no resolve path and there is no
   operator CLI for `resolve_reconciliation_issue` — one manual trade in the
   Kraken app (fill reconciliation is account-wide) permanently halts the bot
   until manual SQLite surgery. Build a reviewed operator remediation CLI.
2. **Fill re-attribution after late order binding.** `_fill_hash` includes
   `order_id`, so a fill first stored with `order_id NULL` (entry response
   lost + order lookup failing in the same pass) raises `PayloadConflictError`
   on re-reconciliation and freezes the fill cursor. Exclude mutable owner
   columns from the hash and match the fetched order's `cl_ord_id` before the
   NULL fallback (`execution_coordinator.py` `_apply_exchange_fill`).
3. **Bounded UNKNOWN→FAILED path.** A submission Kraken never admitted stays
   UNKNOWN forever (transition graph allows UNKNOWN→FAILED, nothing takes it);
   the ticker stays reserved by `reserved_entry_tickers`. Needs a bounded
   non-admission criterion (completed fill window + closed-orders check past
   the submission timestamp).
4. **Ambiguous stop-amend adoption.** Amend admitted + read-back failed leaves
   the store expecting the old trigger; `verify_protective_stop` then loops
   `PROTECTION_MISMATCH` forever even though the exchange trigger is strictly
   better. Adopt the observed trigger when it is the invariant-safe direction.
5. **Same-millisecond fill ordering.** Sort keys use `(timestamp, fill_id)`
   in the coordinator, store rebuild (`ORDER BY executed_at_ms, fill_id`), and
   `rebuild_v10.py`; Kraken txids are not chronological, so an opposite-side
   pair in one millisecond can abort with `FillApplicationError`. Use Kraken's
   monotonic `trade_id` (present in the raw payload) as the tiebreaker —
   needs a persisted column in `fills`.
6. **Universe-scoped position management.** A position whose symbol is removed
   from `ORACLE_UNIVERSE` is silently unmanaged (no stop verify, no ratchet,
   no exits, no issue recorded). Drive recovery/protection/exit sweeps from
   `store.list_open_positions()` instead of the settings list.
7. **Mode-scoped shared state.** `analytics_db_path`, `healthcheck_file`, and
   `llm_budget_db_path` are not mode-scoped while the singleton lock is —
   dual-mode in one directory lets a PAPER process mask a dead LIVE process's
   healthcheck. Scope the defaults per mode or reject dual-mode same-dir.
8. **Account-scoped mutual exclusion.** The singleton lock is
   filesystem-scoped; two LIVE containers on different volumes can trade the
   same Kraken account with a real double-exposure window before reconciliation
   halts them. Derive the lock identity from the account digest and/or add an
   exchange-side mutual-exclusion check.
9. **Quality digest ≠ published digest.** The CI smoke tests a different image
   build than `publish-candidate` pushes (base is digest-pinned now, but build
   args differ and the published digest itself is never executed in CI). Either
   run the smoke against the pushed candidate digest or accept the QNAP PAPER
   soak as the first execution.
10. **Paper-venue fill eviction.** `PaperExchange` silently evicts fills beyond
    `max_fill_events` while the live exchange refuses truncation
    (`FillPaginationLimitError`); the cursor can advance past evicted fills
    with `complete=True`. Raise when `since_ms` predates retained history.

---

## Supply chain — land the outstanding dependency security updates

**74 Dependabot alerts (2 critical, 31 high)** are open across `litellm`,
`pillow`, `setuptools`, `soupsieve`, `cryptography`, `aiohttp`, `urllib3`,
`requests`, `protobuf`, and others — a backlog accumulated while a hardcoded
`exclude-newer = "2026-05-13T00:00:00Z"` made every later patch unresolvable.
Dependabot failed with `dependency_file_not_resolvable` on each attempt
(observed 2026-07-30 for litellm, pillow, setuptools).

The cutoff is now a rolling `exclude-newer = "14 days"`, so patches older than
the window resolve normally and Dependabot can open per-package PRs, each with
its own CI run. Remaining work is to review and land those PRs.

Two things to keep in mind while doing it:

- A fix published inside the last 14 days is still outside the window by
  design. If a critical advisory lands and must ship immediately, use
  `exclude-newer-package` for that one package rather than widening the window.
- The test suite mocks `litellm.completion`, so a green suite does **not**
  validate a real litellm upgrade. Exercise the actual provider path before
  trusting a litellm bump in LIVE.

Deliberately kept out of the v10.0.0 release: bumping the dependency tree of an
already-reviewed safety release would have invalidated its verification.

---

## v10 Release Follow-ups

- **Exact protection resizing:** DCA and partial exits remain disabled on the
  v10 coordinator path. Implement cancel/replace recovery and exact reserved
  quantity accounting before either feature can be enabled.
- **Reviewed rebuild tooling:** v9 `trades.db` remains a read-only archive and
  `analytics-v10.db` is not execution authority. No supported rebuild/import
  command exists. Specify provenance, scope checks, idempotency, dry-run output,
  and recovery behavior before adding an operator command.
- **Representative candidate validation:** exercise the exact candidate digest
  under representative PAPER load before stable promotion; a quiet window is
  not verification.
- **Watchtower decision:** keep it profile-disabled unless the operator accepts
  daemon-equivalent socket access, private-registry credential mounting, and
  manual retained-image/digest rollback without health gating.

v10 LLM defaults are intentionally conservative: sentiment and challenger are
off, LIVE influence is separately gated, and enabled calls are bounded by the
persistent daily/monthly/shadow budget ledger plus token and timeout caps.

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

## 1b. Order-identity fill reconciliation - SUPERSEDED v10

The v10 execution store records exact order and fill identities and is the
execution authority. Legacy fuzzy matching in the analytics journal must not be
used to reconstruct runtime positions or order state.

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
