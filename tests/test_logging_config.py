"""Tests for time-based log rotation and config changes."""

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from unittest.mock import patch

import pytest

from config import Settings


@pytest.fixture
def _clean_root_logger():
    """Remove all handlers from root logger before/after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(original_handlers)


_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
    "XAI_API_KEY": "test",
}


class TestLogRetentionConfig:
    """Config-level tests for log_retention_days."""

    def test_default_retention_days(self):
        with patch.dict("os.environ", _ENV):
            settings = Settings()
        assert settings.log_retention_days == 7

    def test_custom_retention_days(self):
        with patch.dict("os.environ", {**_ENV, "LOG_RETENTION_DAYS": "14"}):
            settings = Settings()
        assert settings.log_retention_days == 14

    def test_old_size_config_removed(self):
        """log_max_bytes and log_backup_count should no longer exist on Settings."""
        with patch.dict("os.environ", _ENV):
            settings = Settings()
        assert not hasattr(settings, "log_max_bytes")
        assert not hasattr(settings, "log_backup_count")

    def test_zero_retention_rejected(self):
        """backupCount=0 would disable deletion, must be rejected."""
        from pydantic import ValidationError

        with patch.dict("os.environ", {**_ENV, "LOG_RETENTION_DAYS": "0"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_negative_retention_rejected(self):
        from pydantic import ValidationError

        with patch.dict("os.environ", {**_ENV, "LOG_RETENTION_DAYS": "-1"}):
            with pytest.raises(ValidationError):
                Settings()


class TestSetupLogging:
    """Integration tests for setup_logging producing a TimedRotatingFileHandler."""

    @pytest.mark.usefixtures("_clean_root_logger")
    def test_creates_timed_rotating_handler(self, tmp_path):
        """setup_logging should attach a TimedRotatingFileHandler to root logger."""
        from main import setup_logging

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict("os.environ", _ENV):
            settings = Settings()

        with patch("os.getcwd", return_value=str(tmp_path)):
            setup_logging(settings)

        root = logging.getLogger()
        timed_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(timed_handlers) >= 1, "Expected at least one TimedRotatingFileHandler"

    @pytest.mark.usefixtures("_clean_root_logger")
    def test_handler_uses_utc(self, tmp_path):
        """Handler should rotate at midnight UTC."""
        from main import setup_logging

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict("os.environ", _ENV):
            settings = Settings()

        with patch("os.getcwd", return_value=str(tmp_path)):
            setup_logging(settings)

        root = logging.getLogger()
        timed_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert timed_handlers, "No TimedRotatingFileHandler found"
        assert timed_handlers[0].utc is True

    @pytest.mark.usefixtures("_clean_root_logger")
    def test_handler_backup_count_matches_config(self, tmp_path):
        """BackupCount should equal settings.log_retention_days."""
        from main import setup_logging

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict("os.environ", {**_ENV, "LOG_RETENTION_DAYS": "3"}):
            settings = Settings()

        assert settings.log_retention_days == 3

        with patch("os.getcwd", return_value=str(tmp_path)):
            setup_logging(settings)

        root = logging.getLogger()
        timed_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert timed_handlers, "No TimedRotatingFileHandler found"
        assert timed_handlers[0].backupCount == 3

    @pytest.mark.usefixtures("_clean_root_logger")
    def test_no_size_based_rotating_handler(self, tmp_path):
        """RotatingFileHandler (size-based) should not be used."""
        from main import setup_logging

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict("os.environ", _ENV):
            settings = Settings()

        with patch("os.getcwd", return_value=str(tmp_path)):
            setup_logging(settings)

        root = logging.getLogger()
        rotating_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert rotating_handlers == [], "RotatingFileHandler (size-based) should not be used"

    @pytest.mark.usefixtures("_clean_root_logger")
    def test_handler_rotation_interval_is_midnight(self, tmp_path):
        """Handler when parameter should be midnight (daily rotation)."""
        from main import setup_logging

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        with patch.dict("os.environ", _ENV):
            settings = Settings()

        with patch("os.getcwd", return_value=str(tmp_path)):
            setup_logging(settings)

        root = logging.getLogger()
        timed_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert timed_handlers, "No TimedRotatingFileHandler found"
        # TimedRotatingFileHandler stores when="midnight" as when="MIDNIGHT"
        assert timed_handlers[0].when == "MIDNIGHT"
