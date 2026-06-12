"""Tests for v9.5 — universe expansion, correlation groups, funding rate fetcher."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
    "XAI_API_KEY": "test",
}


def _make_settings():
    from config import Settings

    with patch.dict("os.environ", _ENV, clear=False):
        return Settings()


def _default_universe() -> list[str]:
    """Return the class-level default universe (ignores a live .env override)."""
    from config import Settings

    raw = Settings.model_fields["trading_universe"].default
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Universe default
# ---------------------------------------------------------------------------


class TestUniverseDefault:
    """Assert the class-level Settings defaults; live .env overrides are ignored."""

    def test_default_uses_usdc_quote(self):
        for ticker in _default_universe():
            assert ticker.endswith("/USDC"), f"Expected USDC quote, got {ticker}"

    def test_default_quote_currency_is_usdc(self):
        from config import Settings

        assert Settings.model_fields["quote_currency"].default == "USDC"

    def test_default_includes_expected_tickers(self):
        universe = _default_universe()
        assert "BTC/USDC" in universe
        assert "ETH/USDC" in universe
        assert "SOL/USDC" in universe
        assert "AVAX/USDC" in universe
        assert "DOT/USDC" in universe
        assert "LINK/USDC" in universe
        assert "XRP/USDC" in universe

    def test_default_excludes_shib_and_doge(self):
        bases = [s.split("/")[0] for s in _default_universe()]
        assert "SHIB" not in bases
        assert "DOGE" not in bases


# ---------------------------------------------------------------------------
# Correlation groups
# ---------------------------------------------------------------------------


class TestCorrelationGroups:
    def test_btc_and_ltc_grouped_together(self):
        from risk_guardian import CORRELATION_GROUPS

        assert CORRELATION_GROUPS["BTC"] == CORRELATION_GROUPS["LTC"]

    def test_eth_in_own_group(self):
        from risk_guardian import CORRELATION_GROUPS

        assert CORRELATION_GROUPS["ETH"] != CORRELATION_GROUPS["BTC"]
        assert CORRELATION_GROUPS["ETH"] != CORRELATION_GROUPS["SOL"]

    def test_l1_alts_grouped(self):
        from risk_guardian import CORRELATION_GROUPS

        group = CORRELATION_GROUPS["SOL"]
        assert CORRELATION_GROUPS["AVAX"] == group
        assert CORRELATION_GROUPS["DOT"] == group

    def test_meme_group_retained(self):
        from risk_guardian import CORRELATION_GROUPS

        assert CORRELATION_GROUPS["DOGE"] == CORRELATION_GROUPS["SHIB"] == "meme"

    def test_correlation_limit_blocks_four_l1_alts(self):
        from risk_guardian import check_correlation_limit

        settings = _make_settings()
        # max_correlated_positions default = 3
        owned = ["SOL/USDC", "AVAX/USDC", "DOT/USDC"]
        # Adding a 4th L1 alt — but we only have SOL, AVAX, DOT on Kraken USDC,
        # so fake a 4th by adding another that maps to same group. AVAX again.
        # The helper just counts same-group positions in the owned list.
        assert check_correlation_limit("SOL/USDC", owned, settings) is False

    def test_correlation_limit_allows_cross_group(self):
        from risk_guardian import check_correlation_limit

        settings = _make_settings()
        settings.max_total_crypto_positions = 10  # relax the portfolio cap for this test
        owned = ["SOL/USDC", "AVAX/USDC", "DOT/USDC"]  # 3 L1 alts
        # BTC is a different group (btc_core) → allowed.
        assert check_correlation_limit("BTC/USDC", owned, settings) is True


class TestPortfolioWideCap:
    """Portfolio-wide max_total_crypto_positions backstop (v9.5 C4 fix)."""

    def test_total_cap_blocks_entry_even_when_group_has_room(self):
        from risk_guardian import check_correlation_limit

        settings = _make_settings()
        settings.max_total_crypto_positions = 3
        settings.max_correlated_positions = 3
        # 3 positions total — at the cap. A new entry in a *different* group
        # is still blocked because total is already maxed.
        owned = ["BTC/USDC", "ETH/USDC", "XRP/USDC"]  # all different groups
        assert check_correlation_limit("SOL/USDC", owned, settings) is False

    def test_total_cap_allows_entry_below_total(self):
        from risk_guardian import check_correlation_limit

        settings = _make_settings()
        settings.max_total_crypto_positions = 5
        settings.max_correlated_positions = 3
        owned = ["BTC/USDC", "ETH/USDC"]
        assert check_correlation_limit("SOL/USDC", owned, settings) is True

    def test_total_cap_default_is_five(self):
        from config import Settings

        assert Settings.model_fields["max_total_crypto_positions"].default == 5  # noqa: PLR2004


# ---------------------------------------------------------------------------
# FundingRateFetcher
# ---------------------------------------------------------------------------


class TestFundingRateFetcher:
    def test_unknown_exchange_disables_fetcher(self):
        from funding import FundingRateFetcher

        f = FundingRateFetcher(exchange_id="not_a_real_exchange")
        assert f.enabled is False
        assert f.fetch("BTC/USDC") is None

    def test_fetch_returns_none_when_client_raises(self):
        from funding import FundingRateFetcher

        f = FundingRateFetcher.__new__(FundingRateFetcher)
        f._exchange_id = "binance"  # noqa: SLF001
        f._client = MagicMock()  # noqa: SLF001
        f._client.fetch_funding_rate.side_effect = RuntimeError("down")  # noqa: SLF001
        assert f.fetch("BTC/USDT") is None

    def test_fetch_parses_funding_rate(self):
        from funding import FundingRateFetcher

        f = FundingRateFetcher.__new__(FundingRateFetcher)
        f._exchange_id = "binance"  # noqa: SLF001
        f._client = MagicMock()  # noqa: SLF001
        f._client.fetch_funding_rate.return_value = {
            "fundingRate": 0.0001,
            "symbol": "BTC/USDT:USDT",
        }
        rate = f.fetch("BTC/USDT")
        assert rate == pytest.approx(0.0001)

    def test_fetch_maps_spot_to_perp_symbol(self):
        from funding import FundingRateFetcher

        f = FundingRateFetcher.__new__(FundingRateFetcher)
        f._exchange_id = "binance"  # noqa: SLF001
        f._client = MagicMock()  # noqa: SLF001
        f._client.fetch_funding_rate.return_value = {"fundingRate": 0.0}  # noqa: SLF001
        f.fetch("BTC/USDT")
        called_symbol = f._client.fetch_funding_rate.call_args.args[0]  # noqa: SLF001
        assert called_symbol == "BTC/USDT:USDT"

    def test_to_perp_passes_through_already_perp_symbol(self):
        from funding import FundingRateFetcher

        assert FundingRateFetcher._to_perp("BTC/USDT:USDT") == "BTC/USDT:USDT"  # noqa: SLF001

    def test_to_perp_handles_malformed_symbol(self):
        from funding import FundingRateFetcher

        assert FundingRateFetcher._to_perp("BTCUSDT") == "BTCUSDT"  # noqa: SLF001


# ---------------------------------------------------------------------------
# MarketData and Journal integration
# ---------------------------------------------------------------------------


class TestMarketDataFundingField:
    def test_funding_rate_defaults_to_none(self):
        from models import MarketData

        data = MarketData(ticker="BTC/USDC", current_price=100.0)
        assert data.funding_rate is None

    def test_funding_rate_round_trips_through_snapshot(self, tmp_path):
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        journal.save_market_snapshot(
            ticker="BTC/USDC",
            price=100.0,
            sentiment="",
            consensus="",
            fear_greed="",
            polymarket_odds="",
            indicators_json="{}",
            regime="RANGING",
            funding_rate=0.0002,
        )
        row = journal._conn.execute(  # noqa: SLF001
            "SELECT funding_rate FROM market_snapshots WHERE ticker=?",
            ("BTC/USDC",),
        ).fetchone()
        assert row[0] == pytest.approx(0.0002)

    def test_snapshot_migration_idempotent_on_existing_db(self, tmp_path):
        """A legacy market_snapshots table without funding_rate must be migrated."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE market_snapshots ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp TEXT NOT NULL,"
            " ticker TEXT NOT NULL,"
            " price REAL NOT NULL,"
            " sentiment TEXT NOT NULL DEFAULT '',"
            " consensus TEXT NOT NULL DEFAULT '',"
            " fear_greed TEXT NOT NULL DEFAULT '',"
            " polymarket_odds TEXT NOT NULL DEFAULT '',"
            " indicators_json TEXT NOT NULL DEFAULT '{}',"
            " regime TEXT NOT NULL DEFAULT 'RANGING'"
            ")",
        )
        conn.commit()
        conn.close()

        from journal import TradeJournal

        j = TradeJournal(db_path=str(db_path))
        cols = [row[1] for row in j._conn.execute("PRAGMA table_info(market_snapshots)").fetchall()]  # noqa: SLF001
        assert "funding_rate" in cols
