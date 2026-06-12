# Contributing to Oracle Pythia

Thank you for your interest in contributing to Oracle Pythia.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-user>/pythia.git`
3. Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. Install dependencies: `uv sync`
5. Copy `.env.example` to `.env` and configure (PAPER mode for development)
6. Run tests: `uv run pytest tests/ -v` (357+ tests as of v9.5)

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Write tests first (TDD)
3. Implement your changes
4. Run linting: `uv run ruff check src/ tests/`
5. Run type checking: `uv run mypy src/ --ignore-missing-imports`
6. Run tests: `uv run pytest tests/ -v`
7. Commit with a clear message
8. Open a pull request

## Code Style

- **Linter**: ruff (config in `pyproject.toml`)
- **Line length**: 100 characters
- **Quotes**: double
- **Type hints**: required for public functions
- Follow existing patterns in the codebase

## Architecture Principles

- **Safety > Performance > DX** (TigerStyle)
- **No LLM in the decision path** -- all buy/sell/hold signals are rule-based. The v9.1 LLM Challenger produces commentary only and never blocks execution.
- **ToS compliance** -- any LLM integration must keep the LLM observational (never issuing trade instructions) so the bot stays compatible with all major provider ToS.
- **Thread safety** -- all shared state is protected by locks
- **Assertions for invariants** -- use `assert` for logic errors, exceptions for runtime errors

## Testing

- Tests live in `tests/` and use pytest
- Use the `conftest.py` fixtures for common test data
- Mock external APIs (Kraken, Grok, Polymarket) -- never hit live APIs in tests
- Test names should describe the behavior: `test_trending_down_blocks_buy`

## What We're Looking For

See `TODO.md` for the roadmap. v9.0–v9.5 landed the modular refactor, LLM Challenger, conviction sizing, walk-forward optimizer, signal analytics, and universe expansion. Remaining high-impact contributions:

- Wiring the v9.5 funding rate into the composite score (needs walk-forward validation)
- HMM-based regime detection (replaces rule-based thresholds with learned transitions)
- Volatility-scaled / Kelly-approximated position sizing on top of v9.2 conviction tiers
- Signal half-life and time-of-day edge measurements (v9.4 infrastructure in place)
- Order book microstructure signals (requires CCXT WebSocket)
- Historical sentiment/Polymarket replay for high-fidelity backtests
- Portfolio optimization (HRP, efficient frontier)

## Financial Safety

This bot trades real money. All contributions must:

- Never weaken existing risk controls (stop-loss, daily loss limit, position caps)
- Never auto-apply optimized parameters without human review
- Include tests that verify safety invariants
- Use PAPER mode during development -- never test against live Kraken

## Questions?

Open an issue for discussion before starting large changes.
