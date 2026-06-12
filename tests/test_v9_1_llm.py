"""Tests for v9.1 — LiteLLM migration, LLMClient, LLMChallenger, schema migration."""

import json
import sqlite3
import types
from unittest.mock import MagicMock, patch

import pytest

_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
    "XAI_API_KEY": "test",
}


def _make_settings(**overrides):
    from config import Settings

    env = {**_ENV, **{k.upper(): str(v) for k, v in overrides.items()}}
    with patch.dict("os.environ", env, clear=False):
        return Settings()


def _fake_litellm_response(content: str):
    """Build a litellm-shaped response object with a single choice."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_chat_happy_path(self):
        from llm_client import LLMClient

        client = LLMClient(primary_model="anthropic/claude-sonnet-4-5")
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response("hello from LLM")
            out = client.chat("system", "user")
        assert out == "hello from LLM"
        assert mock_completion.call_count == 1

    def test_falls_back_to_secondary_on_primary_failure(self):
        from llm_client import LLMClient

        client = LLMClient(
            primary_model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-flash",
        )
        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"].startswith("anthropic/"):
                raise RuntimeError("primary down")
            return _fake_litellm_response("fallback response")

        with patch("litellm.completion", side_effect=side_effect):
            out = client.chat("system", "user")
        assert out == "fallback response"
        assert calls == ["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-flash"]

    def test_circuit_breaker_opens_after_three_failures(self):
        from llm_client import LLMClient

        client = LLMClient(
            primary_model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-flash",
        )
        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"].startswith("anthropic/"):
                raise RuntimeError("primary down")
            return _fake_litellm_response("fallback ok")

        with patch("litellm.completion", side_effect=side_effect):
            for _ in range(3):
                client.chat("system", "user")

        # After 3 primary failures, circuit is open — next chat should skip primary.
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response("fallback ok")
            client.chat("system", "user")
            called_models = [c.kwargs["model"] for c in mock_completion.call_args_list]
            assert all(m.startswith("gemini/") for m in called_models)

    def test_raises_when_no_fallback_configured_and_primary_fails(self):
        from llm_client import LLMClient

        client = LLMClient(primary_model="anthropic/claude-sonnet-4-5")
        with patch("litellm.completion", side_effect=RuntimeError("down")):
            with pytest.raises(RuntimeError):
                client.chat("system", "user")


# ---------------------------------------------------------------------------
# LLMChallenger
# ---------------------------------------------------------------------------


def _make_signal_and_data(action="BUY", composite=0.31):
    from models import MarketData, TradeSignal

    signal = TradeSignal(
        action=action,
        confidence=abs(composite),
        composite=composite,
        reasoning="bot reasoning",
    )
    data = MarketData(
        ticker="BTC/USDT",
        current_price=50000.0,
        indicators=None,
        volatility=None,
        forecast=None,
        sentiment="BULLISH (Score: 8/10)",
        consensus="UP, 65%",
        fear_greed="30 (Fear)",
        polymarket_odds="Polymarket: No data available.",
        has_position=False,
    )
    return signal, data


class TestLLMChallenger:
    def test_disabled_returns_none(self):
        from llm_challenger import LLMChallenger

        challenger = LLMChallenger(client=None, enabled=False)
        signal, data = _make_signal_and_data()
        assert challenger.review(signal, data) is None
        assert challenger.enabled is False

    def test_hold_signal_returns_none_even_when_enabled(self):
        from llm_challenger import LLMChallenger

        client = MagicMock()
        challenger = LLMChallenger(client=client, enabled=True)
        signal, data = _make_signal_and_data(action="HOLD", composite=0.1)
        assert challenger.review(signal, data) is None
        assert client.chat.call_count == 0

    def test_happy_path_parses_json_response(self):
        from llm_challenger import LLMChallenger

        client = MagicMock()
        client._primary = "anthropic/claude-sonnet-4-5"  # noqa: SLF001
        client.chat.return_value = json.dumps(
            {
                "concern_level": "MEDIUM",
                "thesis": "Oversold bounce with weak volume — needs confirmation.",
                "concerns": ["low volume", "regime still bearish"],
            },
        )
        challenger = LLMChallenger(client=client, enabled=True)
        signal, data = _make_signal_and_data(action="BUY")
        critique = challenger.review(signal, data)
        assert critique is not None
        assert critique.concern_level == "MEDIUM"
        assert "Oversold" in critique.thesis
        assert critique.concerns == ["low volume", "regime still bearish"]

    def test_malformed_response_returns_safe_fallback(self):
        from llm_challenger import LLMChallenger

        client = MagicMock()
        client._primary = "anthropic/claude-sonnet-4-5"  # noqa: SLF001
        client.chat.return_value = "no JSON here just prose"
        challenger = LLMChallenger(client=client, enabled=True)
        signal, data = _make_signal_and_data(action="BUY")
        critique = challenger.review(signal, data)
        assert critique is not None
        assert critique.concern_level == "NONE"

    def test_llm_exception_returns_safe_fallback(self):
        from llm_challenger import LLMChallenger

        client = MagicMock()
        client._primary = "anthropic/claude-sonnet-4-5"  # noqa: SLF001
        client.chat.side_effect = RuntimeError("provider down")
        challenger = LLMChallenger(client=client, enabled=True)
        signal, data = _make_signal_and_data(action="BUY")
        critique = challenger.review(signal, data)
        assert critique is not None
        assert critique.concern_level == "NONE"

    def test_invalid_concern_level_rejected(self):
        from llm_challenger import LLMChallenger

        client = MagicMock()
        client._primary = "anthropic/claude-sonnet-4-5"  # noqa: SLF001
        client.chat.return_value = json.dumps(
            {
                "concern_level": "CRITICAL",
                "thesis": "...",
                "concerns": [],
            },
        )
        challenger = LLMChallenger(client=client, enabled=True)
        signal, data = _make_signal_and_data(action="BUY")
        critique = challenger.review(signal, data)
        assert critique is not None
        assert critique.concern_level == "NONE"

    def test_critique_format_for_journal(self):
        from llm_challenger import Critique

        c = Critique(
            concern_level="HIGH",
            thesis="A risky long with macro headwinds.",
            concerns=["Fed meeting tomorrow", "ETF outflows 3d"],
            model="anthropic/claude-sonnet-4-5",
        )
        out = c.format_for_journal()
        assert out.startswith("[HIGH]")
        assert "risky long" in out
        assert "Fed meeting tomorrow" in out


class TestBuildChallengerFromSettings:
    def test_disabled_when_challenger_enabled_false(self):
        from llm_challenger import build_challenger_from_settings

        settings = _make_settings()
        assert settings.challenger_enabled is False
        challenger = build_challenger_from_settings(settings)
        assert challenger.enabled is False

    def test_enabled_when_flag_true_and_keys_present(self):
        from llm_challenger import build_challenger_from_settings

        settings = _make_settings()
        # Simulate operator setting via .env
        settings.challenger_enabled = True
        challenger = build_challenger_from_settings(settings)
        assert challenger.enabled is True


# ---------------------------------------------------------------------------
# Schema migration — llm_critique column
# ---------------------------------------------------------------------------


class TestSignalsSchemaMigration:
    def test_fresh_journal_has_llm_critique_column(self, tmp_path):
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "fresh.db"))
        cursor = journal._conn.execute("PRAGMA table_info(signals)")  # noqa: SLF001
        cols = [row[1] for row in cursor.fetchall()]
        assert "llm_critique" in cols

    def test_existing_journal_gets_column_migrated(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        # Manually create an old-schema signals table without llm_critique.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE signals ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp TEXT NOT NULL,"
            " ticker TEXT NOT NULL,"
            " action TEXT NOT NULL,"
            " confidence REAL NOT NULL,"
            " composite REAL NOT NULL,"
            " vol_regime TEXT NOT NULL DEFAULT 'NORMAL',"
            " reasoning TEXT NOT NULL DEFAULT ''"
            ")",
        )
        conn.commit()
        conn.close()

        from journal import TradeJournal

        journal = TradeJournal(db_path=str(db_path))
        cursor = journal._conn.execute("PRAGMA table_info(signals)")  # noqa: SLF001
        cols = [row[1] for row in cursor.fetchall()]
        assert "llm_critique" in cols

    def test_log_signal_persists_critique(self, tmp_path):
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        journal.log_signal(
            ticker="BTC/USDT",
            action="BUY",
            confidence=0.31,
            composite=0.31,
            vol_regime="NORMAL",
            reasoning="test",
            llm_critique="[MEDIUM] risky but OK",
        )
        row = journal._conn.execute(  # noqa: SLF001
            "SELECT llm_critique FROM signals WHERE ticker=?",
            ("BTC/USDT",),
        ).fetchone()
        assert row[0] == "[MEDIUM] risky but OK"

    def test_log_signal_defaults_critique_to_empty(self, tmp_path):
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        journal.log_signal(
            ticker="ETH/USDT",
            action="SELL",
            confidence=0.2,
            composite=-0.2,
            vol_regime="NORMAL",
            reasoning="test",
        )
        row = journal._conn.execute(  # noqa: SLF001
            "SELECT llm_critique FROM signals WHERE ticker=?",
            ("ETH/USDT",),
        ).fetchone()
        assert row[0] == ""


# ---------------------------------------------------------------------------
# Sentiment migration — LLM calls route through LLMClient
# ---------------------------------------------------------------------------


class TestSentimentLiteLLMMigration:
    def test_sentiment_oracle_uses_litellm_client(self):
        from pydantic import SecretStr

        from sentiment import SentimentOracle

        oracle = SentimentOracle(
            xai_key=SecretStr("fake"),
            model_name="grok-4-1-fast-reasoning",
        )
        # Legacy bare model name should auto-prefix with xai/
        assert oracle._llm._primary == "xai/grok-4-1-fast-reasoning"  # noqa: SLF001

    def test_sentiment_analysis_returns_litellm_output(self):
        from pydantic import SecretStr

        from sentiment import SentimentOracle

        oracle = SentimentOracle(
            xai_key=SecretStr("fake"),
            model_name="xai/grok-4-1",
        )
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "BULLISH (Score: 8/10): strong uptrend",
            )
            out = oracle.analyze_ticker_sentiment("BTC/USDT")
        assert out.startswith("BULLISH")
        assert "8/10" in out

    def test_sentiment_failure_returns_neutral_fallback(self):
        from pydantic import SecretStr

        from sentiment import SentimentOracle

        oracle = SentimentOracle(
            xai_key=SecretStr("fake"),
            model_name="xai/grok-4-1",
        )
        with patch("litellm.completion", side_effect=RuntimeError("down")):
            out = oracle.analyze_ticker_sentiment("BTC/USDT")
        assert out.startswith("NEUTRAL")


# ---------------------------------------------------------------------------
# decision_engine integration — challenger is called for BUY, logged to journal
# ---------------------------------------------------------------------------


class TestDecisionEngineChallengerWiring:
    def test_run_challenger_helper_passes_through_disabled(self):
        from decision_engine import _run_challenger_and_format

        out = _run_challenger_and_format(
            challenger=None,
            signal=MagicMock(),
            data=MagicMock(),
            notifier=MagicMock(),
        )
        assert out == ""

    def test_run_challenger_helper_notifies_on_high(self):
        from decision_engine import _run_challenger_and_format
        from llm_challenger import Critique

        challenger = MagicMock()
        challenger.review.return_value = Critique(
            concern_level="HIGH",
            thesis="Bad entry; macro risk imminent.",
            concerns=["Fed", "ETF outflow"],
            model="test/model",
        )
        data = MagicMock()
        data.ticker = "BTC/USDT"
        signal = MagicMock()
        signal.action = "BUY"
        notifier = MagicMock()

        out = _run_challenger_and_format(challenger, signal, data, notifier)
        assert out.startswith("[HIGH]")
        notifier.notify_error.assert_called_once()
        assert "HIGH concern" in notifier.notify_error.call_args.args[0]

    def test_run_challenger_helper_suppresses_notifier_exceptions(self):
        from decision_engine import _run_challenger_and_format
        from llm_challenger import Critique

        challenger = MagicMock()
        challenger.review.return_value = Critique(
            concern_level="HIGH",
            thesis="risky",
            concerns=[],
            model="test/model",
        )
        data = MagicMock()
        data.ticker = "BTC/USDT"
        signal = MagicMock()
        signal.action = "BUY"
        notifier = MagicMock()
        notifier.notify_error.side_effect = RuntimeError("webhook down")

        # Must not raise even if notifier blows up.
        out = _run_challenger_and_format(challenger, signal, data, notifier)
        assert out.startswith("[HIGH]")

    def test_run_challenger_helper_suppresses_review_exceptions(self):
        """Even if ``review`` itself raises, the bot keeps trading."""
        from decision_engine import _run_challenger_and_format

        challenger = MagicMock()
        challenger.review.side_effect = RuntimeError("review crashed")
        data = MagicMock()
        data.ticker = "BTC/USDT"
        signal = MagicMock()
        signal.action = "BUY"
        notifier = MagicMock()

        out = _run_challenger_and_format(challenger, signal, data, notifier)
        assert out == ""


class TestCircuitBreakerAutoClose:
    """Breaker must auto-close after cooldown so primary isn't lost forever."""

    def test_cooldown_elapsed_reopens_probe(self, monkeypatch):
        import llm_client
        from llm_client import LLMClient

        client = LLMClient(
            primary_model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-flash",
        )
        # Force breaker open.
        client._consecutive_failures = 3  # noqa: SLF001
        client._opened_at = 0.0  # noqa: SLF001

        # Pretend clock is 1 second past cooldown.
        monkeypatch.setattr(
            llm_client.time,
            "monotonic",
            lambda: llm_client._CIRCUIT_OPEN_COOLDOWN_SECONDS + 1.0,
        )
        assert client.is_circuit_open() is False

    def test_within_cooldown_stays_open(self, monkeypatch):
        import llm_client
        from llm_client import LLMClient

        client = LLMClient(
            primary_model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-flash",
        )
        client._consecutive_failures = 3  # noqa: SLF001
        client._opened_at = 100.0  # noqa: SLF001

        monkeypatch.setattr(llm_client.time, "monotonic", lambda: 101.0)
        assert client.is_circuit_open() is True


class TestKeyRedactionInErrors:
    """API keys echoed in exception messages must be redacted before logging/re-raise."""

    def test_raised_runtime_error_does_not_contain_api_key(self):
        from llm_client import LLMClient

        client = LLMClient(
            primary_model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-flash",
        )
        leaked_primary = "Invalid API key: sk-ant-SECRET123456789012"
        leaked_fallback = "Invalid API key: AIzaSECRET9876543210"

        def side_effect(**kwargs):
            if kwargs["model"].startswith("anthropic/"):
                raise RuntimeError(leaked_primary)
            raise RuntimeError(leaked_fallback)

        with patch("litellm.completion", side_effect=side_effect):
            with pytest.raises(RuntimeError) as excinfo:
                client.chat("system", "user")

        err = str(excinfo.value)
        assert "sk-ant-SECRET" not in err, f"Primary key leaked: {err!r}"
        assert "AIzaSECRET" not in err, f"Fallback key leaked: {err!r}"
        assert "[REDACTED]" in err


class TestSignalsReadsLlmCritique:
    """get_recent_signals must surface the new column."""

    def test_get_recent_signals_includes_llm_critique(self, tmp_path):
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        journal.log_signal(
            ticker="BTC/USDT",
            action="BUY",
            confidence=0.31,
            composite=0.31,
            vol_regime="NORMAL",
            reasoning="test",
            llm_critique="[MEDIUM] caution",
        )
        rows = journal.get_recent_signals(ticker="BTC/USDT")
        assert len(rows) == 1
        assert rows[0]["llm_critique"] == "[MEDIUM] caution"
