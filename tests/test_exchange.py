"""Tests for the Exchange Layer."""
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from exchange import KrakenExchange


def test_place_maker_buy_enforces_limit_and_post_only() -> None:
    """Verify that place_maker_buy calls ccxt with correct safety parameters."""
    # Setup
    mock_ccxt = MagicMock()

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
