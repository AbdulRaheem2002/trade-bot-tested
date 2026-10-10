"""Tests for the decision validator – covers Rules 1-5, 8, 10, 11, 13, 14."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from app.common.enums import Action, ConvictionTier, RejectionReason
from app.common.models import BreakerState
from app.config.settings import TradingConfig
from app.risk.validator import validate_item, validate_response
from tests.conftest import (
    make_cycle_state,
    make_decision_item,
    make_portfolio,
    make_position,
    make_snapshot,
)

# ════════════════════════════════════════════════════════════════════════
# Rule 1: validate_response – parse / schema
# ════════════════════════════════════════════════════════════════════════


class TestValidateResponse:
    """Rule 1: malformed/unparsable/empty → hold-all (None)."""

    def test_none_input(self):
        assert validate_response(None) is None

    def test_empty_string(self):
        assert validate_response("") is None

    def test_malformed_json(self):
        assert validate_response("{not json!!!}") is None

    def test_not_a_dict(self):
        assert validate_response("[1, 2, 3]") is None

    def test_wrong_types_in_decision(self):
        """decisions must be a list of objects."""
        raw = json.dumps({"reasoning": "test", "decisions": "not a list"})
        assert validate_response(raw) is None

    def test_empty_decisions_list(self):
        raw = json.dumps({"reasoning": "no trades", "decisions": []})
        assert validate_response(raw) is None

    def test_valid_hold_decision(self):
        raw = json.dumps(
            {
                "reasoning": "market uncertain",
                "decisions": [{"action": "hold", "symbol": "BTC/USDT"}],
            }
        )
        result = validate_response(raw)
        assert result is not None
        assert len(result.decisions) == 1
        assert result.decisions[0].action == Action.HOLD

    def test_valid_open_long(self):
        raw = {
            "reasoning": "bullish",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 65000,
                    "conviction": "A",
                    "confidence": 0.85,
                    "thesis": "trend up",
                    "rationale": "EMA setup",
                }
            ],
        }
        result = validate_response(raw)
        assert result is not None
        assert result.decisions[0].conviction == ConvictionTier.A

    def test_extra_fields_tolerated(self):
        """Extra keys in the LLM response should not cause a rejection."""
        raw = {
            "reasoning": "test",
            "extra_key": True,
            "decisions": [
                {"action": "hold", "symbol": "BTC/USDT", "mood": "nervous"},
            ],
        }
        result = validate_response(raw)
        assert result is not None

    def test_invalid_action_value(self):
        """An unknown action value fails schema validation → None."""
        raw = json.dumps(
            {
                "reasoning": "test",
                "decisions": [{"action": "yolo_buy", "symbol": "BTC/USDT"}],
            }
        )
        assert validate_response(raw) is None

    def test_confidence_out_of_range(self):
        raw = json.dumps(
            {
                "reasoning": "test",
                "decisions": [
                    {
                        "action": "open_long",
                        "symbol": "BTC/USDT",
                        "invalidation_price": 65000,
                        "conviction": "A",
                        "confidence": 1.5,
                    }
                ],
            }
        )
        assert validate_response(raw) is None


# ════════════════════════════════════════════════════════════════════════
# Rule 3: required fields for open/add
# ════════════════════════════════════════════════════════════════════════


class TestRule3RequiredFields:
    def test_symbol_not_in_universe(self, cfg: TradingConfig):
        item = make_decision_item(symbol="DOGE/USDT")
        rej = validate_item(
            item, make_portfolio(), make_snapshot(symbol="DOGE/USDT"), make_cycle_state(), cfg
        )
        assert rej is not None
        assert rej.reason == RejectionReason.SYMBOL_NOT_IN_UNIVERSE

    def test_missing_invalidation_price(self, cfg: TradingConfig):
        item = make_decision_item(invalidation_price=None)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.MISSING_INVALIDATION_PRICE

    def test_missing_conviction(self, cfg: TradingConfig):
        item = make_decision_item(conviction=None)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.MISSING_CONVICTION

    def test_missing_confidence(self, cfg: TradingConfig):
        item = make_decision_item(confidence=None)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.MISSING_CONFIDENCE


# ════════════════════════════════════════════════════════════════════════
# Rule 4: invalidation geometry
# ════════════════════════════════════════════════════════════════════════


class TestRule4Invalidation:
    def test_long_invalidation_above_entry(self, cfg: TradingConfig):
        """Long stop must be BELOW entry."""
        item = make_decision_item(invalidation_price=72_000.0)
        snap = make_snapshot(current_price=70_000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.INVALIDATION_WRONG_SIDE

    def test_short_invalidation_below_entry(self, cfg: TradingConfig):
        """Short stop must be ABOVE entry."""
        cfg_short = TradingConfig(shorts_enabled=True, min_equity_for_shorts=1.0)
        snap = make_snapshot(
            current_price=70_000.0,
            last_close=67_000.0,
            ema200_4h=68_000.0,
        )
        item = make_decision_item(
            action=Action.OPEN_SHORT,
            invalidation_price=69_000.0,
        )
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg_short)
        assert rej is not None
        assert rej.reason == RejectionReason.INVALIDATION_WRONG_SIDE

    def test_invalidation_too_tight(self, cfg: TradingConfig):
        """Stop distance < min_stop_atr (1.0 ATR)."""
        # ATR=1000, stop distance = 500 → 0.5 ATR < 1.0
        item = make_decision_item(invalidation_price=69_500.0)
        snap = make_snapshot(current_price=70_000.0, atr14_4h=1000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.INVALIDATION_TOO_TIGHT

    def test_invalidation_too_wide(self, cfg: TradingConfig):
        """Stop distance > max_stop_atr (4.0 ATR)."""
        # ATR=1000, stop distance = 5000 → 5.0 ATR > 4.0
        item = make_decision_item(invalidation_price=65_000.0)
        snap = make_snapshot(current_price=70_000.0, atr14_4h=1000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.INVALIDATION_TOO_WIDE

    def test_invalidation_just_right(self, cfg: TradingConfig):
        """Stop distance within [1.0, 4.0] ATR → accepted."""
        # ATR=1000, stop at 68000, entry 70000 → 2.0 ATR ✓
        item = make_decision_item(invalidation_price=68_000.0)
        snap = make_snapshot(current_price=70_000.0, atr14_4h=1000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is None


# ════════════════════════════════════════════════════════════════════════
# Rule 5: setup gates
# ════════════════════════════════════════════════════════════════════════


class TestRule5SetupGates:
    def test_long_close_below_ema200(self, cfg: TradingConfig):
        """Long gate: last close must be above EMA200."""
        snap = make_snapshot(last_close=67_000.0, ema200_4h=68_000.0, current_price=67_000.0)
        item = make_decision_item(invalidation_price=65_000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.CLOSE_NOT_ABOVE_EMA200

    def test_short_close_above_ema200(self, cfg: TradingConfig):
        """Short gate: last close must be below EMA200."""
        cfg_short = TradingConfig(shorts_enabled=True, min_equity_for_shorts=1.0)
        snap = make_snapshot(
            last_close=69_000.0,
            ema200_4h=68_000.0,
            current_price=69_000.0,
        )
        item = make_decision_item(
            action=Action.OPEN_SHORT,
            invalidation_price=71_000.0,
        )
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg_short)
        assert rej is not None
        assert rej.reason == RejectionReason.CLOSE_NOT_BELOW_EMA200

    def test_spread_too_wide(self, cfg: TradingConfig):
        snap = make_snapshot(spread_bps=15.0)
        item = make_decision_item(invalidation_price=68_000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.SPREAD_TOO_WIDE

    def test_volume_too_low(self, cfg: TradingConfig):
        snap = make_snapshot(volume_24h=10_000_000.0)
        item = make_decision_item(invalidation_price=68_000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.VOLUME_TOO_LOW

    def test_stale_data(self, cfg: TradingConfig):
        snap = make_snapshot(
            candle_timestamp=datetime(2025, 6, 14, 20, 0, 0, tzinfo=UTC),
        )
        item = make_decision_item(invalidation_price=68_000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.STALE_DATA

    def test_no_snapshot_is_stale(self, cfg: TradingConfig):
        item = make_decision_item(invalidation_price=68_000.0)
        rej = validate_item(item, make_portfolio(), None, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.STALE_DATA


# ════════════════════════════════════════════════════════════════════════
# Rule 8: no reopening a closed symbol
# ════════════════════════════════════════════════════════════════════════


class TestRule8NoReopen:
    def test_reopen_rejected(self, cfg: TradingConfig):
        cs = make_cycle_state(symbols_closed={"BTC/USDT"})
        item = make_decision_item()
        rej = validate_item(item, make_portfolio(), make_snapshot(), cs, cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.SYMBOL_CLOSED_THIS_CYCLE


# ════════════════════════════════════════════════════════════════════════
# Rule 10: add constraints
# ════════════════════════════════════════════════════════════════════════


class TestRule10AddConstraints:
    def _portfolio_with_position(self, unrealized_r: float, stop_at_be: bool):
        """Build a portfolio with a long BTC position at given R and stop state."""
        entry = 65_000.0
        risk = 20.0
        # unrealized_pnl = unrealized_r * risk
        # (current - entry) * qty = unrealized_r * risk
        # current = entry + unrealized_r * risk / qty
        qty = 0.01
        current = entry + unrealized_r * risk / qty
        stop = entry if stop_at_be else entry - 2000.0
        pos = make_position(
            entry_price=entry,
            quantity=qty,
            invalidation_price=stop,
            risk_amount=risk,
        )
        return make_portfolio(positions={"BTC/USDT": pos}), current

    def test_add_unrealized_r_too_low(self, cfg: TradingConfig):
        """Add rejected when position's unrealized R < +1.0."""
        portfolio, current = self._portfolio_with_position(0.5, stop_at_be=True)
        snap = make_snapshot(
            current_price=current,
            last_close=current,
            ema200_4h=current - 5000.0,
            atr14_4h=1000.0,
        )
        item = make_decision_item(
            action=Action.ADD,
            invalidation_price=portfolio.positions["BTC/USDT"].invalidation_price,
        )
        rej = validate_item(item, portfolio, snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.ADD_UNREALIZED_R_TOO_LOW

    def test_add_stop_not_at_breakeven(self, cfg: TradingConfig):
        """Add rejected when existing stop is below entry (long)."""
        portfolio, current = self._portfolio_with_position(1.5, stop_at_be=False)
        snap = make_snapshot(
            current_price=current,
            last_close=current,
            ema200_4h=current - 5000.0,
            atr14_4h=1000.0,
        )
        item = make_decision_item(
            action=Action.ADD,
            invalidation_price=65_000.0,  # at entry = tightened from below
        )
        rej = validate_item(item, portfolio, snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.ADD_STOP_NOT_AT_BREAKEVEN

    def test_add_profitable_and_be_stop_accepted(self, cfg: TradingConfig):
        """Add accepted when R >= +1.0 and stop at breakeven."""
        portfolio, current = self._portfolio_with_position(1.5, stop_at_be=True)
        snap = make_snapshot(
            current_price=current,
            last_close=current,
            ema200_4h=current - 5000,
            atr14_4h=1000.0,
        )
        # New invalidation must be >= existing (tightened); existing = entry = 65000
        # Also distance in ATR must be valid: |current - 65000| / 1000
        # current = 65000 + 1.5*20/0.01 = 65000 + 3000 = 68000
        # distance = 68000 - 65000 = 3000; 3000/1000 = 3.0 ATR ✓
        item = make_decision_item(
            action=Action.ADD,
            invalidation_price=65_000.0,
        )
        rej = validate_item(item, portfolio, snap, make_cycle_state(), cfg)
        assert rej is None


# ════════════════════════════════════════════════════════════════════════
# Rule 11: shorts
# ════════════════════════════════════════════════════════════════════════


class TestRule11Shorts:
    def test_shorts_disabled(self, cfg: TradingConfig):
        item = make_decision_item(action=Action.OPEN_SHORT, invalidation_price=72_000.0)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.SHORTS_DISABLED

    def test_shorts_equity_too_low(self):
        cfg = TradingConfig(shorts_enabled=True, min_equity_for_shorts=25_000)
        item = make_decision_item(action=Action.OPEN_SHORT, invalidation_price=72_000.0)
        portfolio = make_portfolio(equity=10_000.0)
        rej = validate_item(item, portfolio, make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.SHORTS_EQUITY_TOO_LOW


# ════════════════════════════════════════════════════════════════════════
# Rule 13: breaker / kill switch
# ════════════════════════════════════════════════════════════════════════


class TestRule13BreakerActive:
    def test_open_blocked_by_breaker(self, cfg: TradingConfig):
        cs = make_cycle_state(breaker_state=BreakerState(daily_loss_blocked=True))
        item = make_decision_item()
        rej = validate_item(item, make_portfolio(), make_snapshot(), cs, cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.BREAKER_ACTIVE

    def test_close_allowed_during_breaker(self, cfg: TradingConfig):
        cs = make_cycle_state(breaker_state=BreakerState(daily_loss_blocked=True))
        pos = make_position()
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        item = make_decision_item(action=Action.CLOSE)
        rej = validate_item(item, portfolio, make_snapshot(), cs, cfg)
        assert rej is None

    def test_hold_allowed_during_breaker(self, cfg: TradingConfig):
        cs = make_cycle_state(breaker_state=BreakerState(weekly_loss_blocked=True))
        item = make_decision_item(action=Action.HOLD)
        rej = validate_item(item, make_portfolio(), make_snapshot(), cs, cfg)
        assert rej is None

    def test_open_blocked_by_kill_switch(self, cfg: TradingConfig):
        cs = make_cycle_state(kill_switch=True)
        item = make_decision_item()
        rej = validate_item(item, make_portfolio(), make_snapshot(), cs, cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.BREAKER_ACTIVE

    def test_add_blocked_by_breaker(self, cfg: TradingConfig):
        cs = make_cycle_state(breaker_state=BreakerState(peak_drawdown_blocked=True))
        pos = make_position()
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        item = make_decision_item(action=Action.ADD, invalidation_price=64_000.0)
        rej = validate_item(item, portfolio, make_snapshot(), cs, cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.BREAKER_ACTIVE


# ════════════════════════════════════════════════════════════════════════
# Rule 14: invalidation may only be tightened
# ════════════════════════════════════════════════════════════════════════


class TestRule14InvalidationTightening:
    def test_long_invalidation_loosened(self, cfg: TradingConfig):
        """Moving a long stop DOWN is loosening → rejected."""
        pos = make_position(
            entry_price=65_000.0,
            invalidation_price=65_000.0,  # at breakeven
            risk_amount=20.0,
        )
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        # Position R must be >= 1.0
        current = 65_000.0 + 1.5 * 20.0 / 0.01  # 68000
        snap = make_snapshot(
            current_price=current,
            last_close=current,
            ema200_4h=current - 5000,
            atr14_4h=1000.0,
        )
        # Loosening: new stop (64000) < existing stop (65000) for long
        item = make_decision_item(action=Action.ADD, invalidation_price=64_000.0)
        rej = validate_item(item, portfolio, snap, make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.INVALIDATION_LOOSENED

    def test_long_invalidation_tightened_accepted(self, cfg: TradingConfig):
        """Moving a long stop UP is tightening → accepted."""
        pos = make_position(
            entry_price=65_000.0,
            invalidation_price=65_000.0,
            risk_amount=20.0,
        )
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        current = 65_000.0 + 1.5 * 20.0 / 0.01  # 68000
        snap = make_snapshot(
            current_price=current,
            last_close=current,
            ema200_4h=current - 5000,
            atr14_4h=1000.0,
        )
        # Tightening: new stop (66000) > existing (65000) for long
        # distance = |68000 - 66000| = 2000; 2000/1000 = 2.0 ATR ✓
        item = make_decision_item(action=Action.ADD, invalidation_price=66_000.0)
        rej = validate_item(item, portfolio, snap, make_cycle_state(), cfg)
        assert rej is None


# ════════════════════════════════════════════════════════════════════════
# Position existence checks
# ════════════════════════════════════════════════════════════════════════


class TestPositionExistence:
    def test_open_when_position_exists(self, cfg: TradingConfig):
        pos = make_position()
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        item = make_decision_item()
        rej = validate_item(item, portfolio, make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.POSITION_ALREADY_EXISTS

    def test_close_when_no_position(self, cfg: TradingConfig):
        item = make_decision_item(action=Action.CLOSE)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.NO_OPEN_POSITION

    def test_add_when_no_position(self, cfg: TradingConfig):
        item = make_decision_item(action=Action.ADD)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is not None
        assert rej.reason == RejectionReason.NO_OPEN_POSITION


# ════════════════════════════════════════════════════════════════════════
# Valid open_long passes all checks
# ════════════════════════════════════════════════════════════════════════


class TestValidDecisions:
    def test_valid_open_long(self, cfg: TradingConfig):
        item = make_decision_item(invalidation_price=68_000.0)
        snap = make_snapshot(current_price=70_000.0, atr14_4h=1000.0)
        rej = validate_item(item, make_portfolio(), snap, make_cycle_state(), cfg)
        assert rej is None

    def test_valid_close(self, cfg: TradingConfig):
        pos = make_position()
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        item = make_decision_item(action=Action.CLOSE)
        rej = validate_item(item, portfolio, make_snapshot(), make_cycle_state(), cfg)
        assert rej is None

    def test_valid_hold(self, cfg: TradingConfig):
        item = make_decision_item(action=Action.HOLD)
        rej = validate_item(item, make_portfolio(), make_snapshot(), make_cycle_state(), cfg)
        assert rej is None
