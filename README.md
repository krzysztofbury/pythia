# Oracle Pythia

An autonomous crypto trading bot specializing in Kraken, driven by Gemini 3 Pro (Strategy) and Grok (Sentiment).

## Architecture

- **The Oracle (Sentiment)**: Grok (xAI) + Polymarket prediction markets.
- **The Brain (Strategy)**: Google Gemini 3 Pro.
- **The Guardian (Risk)**: Hard-coded risk management (Tuition Cap).
- **The Executor**: CCXT (Kraken) with strict "Maker-Only" enforcement.

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
   - `GOOGLE_API_KEY` (Gemini).
   - `XAI_API_KEY` (Grok).

4. **Run**:
   ```bash
   # Paper Mode (Default)
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   uv run python src/main.py

   # Live Mode
   TRADING_MODE=LIVE uv run python src/main.py
   ```

## Development

- **Linting**:
  ```bash
  uv run ruff check .
  uv run mypy .
  ```
- **Tests**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src && uv run pytest
  ```