"""Tests for config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.common.errors import ConfigError
from app.config.loader import load_config
from app.config.settings import TradingConfig


class TestTradingConfig:
    """Unit tests for TradingConfig Pydantic model."""

    def test_defaults_load(self):
        """Default config.yaml should load without error."""
        cfg = load_config()
        assert cfg.mode.value == "paper"
        assert cfg.risk_per_trade_pct == 0.75
        assert cfg.max_total_exposure_pct == 70.0

    def test_risk_per_trade_too_low(self):
        with pytest.raises(ValidationError):
            TradingConfig(risk_per_trade_pct=0.05)

    def test_risk_per_trade_too_high(self):
        with pytest.raises(ValidationError):
            TradingConfig(risk_per_trade_pct=3.0)

    def test_max_total_exposure_too_high(self):
        with pytest.raises(ValidationError):
            TradingConfig(max_total_exposure_pct=95)

    def test_min_stop_atr_greater_than_max(self):
        with pytest.raises(ValueError, match="min_stop_atr"):
            TradingConfig(min_stop_atr=5.0, max_stop_atr=2.0)

    def test_single_position_gt_total_exposure(self):
        with pytest.raises(ValueError, match="max_single_position_pct"):
            TradingConfig(max_single_position_pct=80, max_total_exposure_pct=70)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            TradingConfig(bogus_field=42)

    def test_universe_empty_rejected(self):
        with pytest.raises(ValidationError):
            TradingConfig(universe=[])


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("foo: [unclosed bracket", encoding="utf-8")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(f)

    def test_non_dict_yaml(self, tmp_path: Path):
        f = tmp_path / "list.yaml"
        f.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(f)

    def test_empty_yaml(self, tmp_path: Path):
        """Empty YAML should produce default config."""
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        cfg = load_config(f)
        assert cfg.mode.value == "paper"

    def test_partial_override(self, tmp_path: Path):
        f = tmp_path / "partial.yaml"
        f.write_text(
            textwrap.dedent("""\
                risk_per_trade_pct: 1.5
                max_open_positions: 3
            """),
            encoding="utf-8",
        )
        cfg = load_config(f)
        assert cfg.risk_per_trade_pct == 1.5
        assert cfg.max_open_positions == 3
        assert cfg.starting_equity == 10_000  # default untouched
