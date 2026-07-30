"""Tests for the Exchange Layer."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from exchange import KrakenExchange

REQUEST_TIMEOUT_MS = 12_345
EXPECTED_PAGINATED_FILLS = 51


def _market() -> dict:
    return {
        "limits": {
            "amount": {"min": 0.0001, "max": 10},
            "price": {"min": 1, "max": 1_000_000},
            "cost": {"min": 10, "max": 1_000_000},
        },
    }


def test_place_maker_buy_enforces_limit_and_post_only() -> None:
    """Verify that place_maker_buy calls ccxt with correct safety parameters."""
    # Setup
    mock_ccxt = MagicMock()
    mock_ccxt.price_to_precision.return_value = "50000.0"
    mock_ccxt.amount_to_precision.return_value = "0.001"
    mock_ccxt.markets = {"BTC/USD": _market()}

    # Patch the internal client creation or just mock the instance
    # Since we can't easily patch ccxt.kraken inside the __init__ without patching the module,
    # we will rely on patching the _client attribute after instantiation or patching the class.
    # Simpler: Subclass for test or just patch ccxt.kraken

    # Proper patching
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ccxt.kraken", lambda _config: mock_ccxt)

        exchange = KrakenExchange(SecretStr("dummy_key"), SecretStr("dummy_secret"))

        # Act
        exchange.place_maker_buy("BTC/USD", 50000.0, 0.001)

        # Assert
        mock_ccxt.create_order.assert_called_once()
        _args, kwargs = mock_ccxt.create_order.call_args

        # Verify positional args or kwargs
        assert kwargs["symbol"] == "BTC/USD"
        assert kwargs["type"] == "limit"  # MUST BE LIMIT
        assert kwargs["side"] == "buy"
        assert kwargs["params"] == {"postOnly": True}  # MUST BE POST-ONLY


def test_timeout_client_id_and_atomic_conditional_stop_are_explicit() -> None:
    """The entry and its Kraken conditional close share one non-retried mutation."""
    mock_ccxt = MagicMock()
    mock_ccxt.markets = {"BTC/USD": _market()}
    mock_ccxt.price_to_precision.side_effect = lambda _symbol, value: str(value)
    mock_ccxt.amount_to_precision.side_effect = lambda _symbol, value: str(value)
    captured_config: dict = {}

    with pytest.MonkeyPatch.context() as monkeypatch:

        def client_factory(config: dict) -> MagicMock:
            captured_config.update(config)
            return mock_ccxt

        monkeypatch.setattr("ccxt.kraken", client_factory)
        exchange = KrakenExchange(
            SecretStr("dummy_key"),
            SecretStr("dummy_secret"),
            request_timeout_ms=REQUEST_TIMEOUT_MS,
        )
        exchange.place_maker_buy(
            "BTC/USD",
            Decimal(50000),
            Decimal("0.001"),
            client_order_id="exact-client-1",
            stop_loss_price=Decimal(45000),
        )

    assert captured_config["timeout"] == REQUEST_TIMEOUT_MS
    assert mock_ccxt.create_order.call_count == 1
    params = mock_ccxt.create_order.call_args.kwargs["params"]
    assert params == {
        "clientOrderId": "exact-client-1",
        "postOnly": True,
        "close": {"ordertype": "stop-loss", "price": "45000"},
    }


def test_market_limits_are_checked_after_precision_rounding() -> None:
    """An amount rounded below Kraken's minimum is rejected before mutation."""
    mock_ccxt = MagicMock()
    mock_ccxt.markets = {"BTC/USD": _market()}
    mock_ccxt.price_to_precision.return_value = "50000"
    mock_ccxt.amount_to_precision.return_value = "0.00009"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("ccxt.kraken", lambda _config: mock_ccxt)
        exchange = KrakenExchange(SecretStr("key"), SecretStr("secret"))
        with pytest.raises(ValueError, match="below market minimum"):
            exchange.place_maker_buy(
                "BTC/USD",
                Decimal(50000),
                Decimal("0.000099"),
                client_order_id="exact-client-2",
            )

    mock_ccxt.create_order.assert_not_called()


def test_normalization_is_side_effect_free_and_matches_submission() -> None:
    """Coordinator can persist exact venue values before the first mutation."""
    mock_ccxt = MagicMock()
    mock_ccxt.markets = {"BTC/USD": _market()}
    mock_ccxt.price_to_precision.side_effect = lambda _symbol, _value: "50000.12"
    mock_ccxt.amount_to_precision.side_effect = lambda _symbol, _value: "0.001"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("ccxt.kraken", lambda _config: mock_ccxt)
        exchange = KrakenExchange(SecretStr("key"), SecretStr("secret"))
        normalized = exchange.normalize_limit_order(
            "BTC/USD",
            Decimal("0.00109"),
            Decimal("50000.123"),
            stop_loss_price=Decimal("45000.126"),
        )

    assert normalized == (Decimal("0.001"), Decimal("50000.12"), Decimal("50000.12"))
    mock_ccxt.create_order.assert_not_called()


def test_account_fill_pagination_uses_fixed_end_and_offsets() -> None:
    """Account history is bounded and every page uses the same end boundary."""
    mock_ccxt = MagicMock()
    mock_ccxt.markets = {"BTC/USD": _market()}
    mock_ccxt.fetch_my_trades.side_effect = [
        [{"id": f"fill-{index}", "order": f"order-{index}"} for index in range(50)],
        [{"id": "fill-50", "order": "order-50"}],
    ]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("ccxt.kraken", lambda _config: mock_ccxt)
        exchange = KrakenExchange(SecretStr("key"), SecretStr("secret"))
        fills = exchange.fetch_account_fills(1_000, 9_999, max_pages=3)

    assert len(fills) == EXPECTED_PAGINATED_FILLS
    assert fills[-1]["id"] == "fill-50"
    assert mock_ccxt.fetch_my_trades.call_args_list[0].kwargs == {
        "since": 1_000,
        "limit": 50,
        "params": {"end": 9, "ofs": 0},
    }
    assert mock_ccxt.fetch_my_trades.call_args_list[1].kwargs["params"] == {
        "end": 9,
        "ofs": 50,
    }
