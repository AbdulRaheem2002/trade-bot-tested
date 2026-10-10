"""Tests for the risk engine – Rules 7, 9, 12 and full pipeline orchestration.

Tests:
- validate_response integration (Rule 1 hold-all on bad payload)
- Close-first ordering and symbols_closed_this_cycle tracking
- Rule 7: at most one new position per cycle (conviction priority, list order tiebreak)
- Rule 7: new_position_opened_this_cycle flag blocks subsequent opens
- Rule 8: reopening a symbol closed in the same cycle is rejected
- Rule 9: max open positions cap (and freeing a slot via close in same cycle)
- Sizing integration: sized orders converted to ApprovedOrder; sizing failures to Rejection
- Add order processing
"""

from __future__ import annotations

from app.common.enums import Action, ConvictionTier, RejectionReason
from app.config.settings import TradingConfig
from app.risk.risk_engine import process_decisions
from tests.conftest import (
    make_cycle_state,
    make_portfolio,
    make_position,
    make_snapshot,
)


class TestRiskEngineParsing:
    """Rule 1 integration with risk engine."""

    def test_malformed_input_results_in_hold_all(self, cfg: TradingConfig):
        portfolio = make_portfolio()
        cycle_state = make_cycle_state()
        approved, rejected = process_decisions(
            "{bad json}",
            portfolio,
            {},
            cycle_state,
            cfg,
        )
        assert approved == []
        assert rejected == []

    def test_none_input_results_in_hold_all(self, cfg: TradingConfig):
        portfolio = make_portfolio()
        cycle_state = make_cycle_state()
        approved, rejected = process_decisions(
            None,
            portfolio,
            {},
            cycle_state,
            cfg,
        )
        assert approved == []
        assert rejected == []


class TestRiskEngineRule7OneNewPositionPerCycle:
    """Rule 7: At most one new position opened per 4h cycle."""

    def test_two_valid_opens_tier_a_wins_over_tier_b(self, cfg: TradingConfig):
        """When multiple opens are valid, tier A is chosen over tier B."""
        portfolio = make_portfolio()
        cycle_state = make_cycle_state()
        snapshots = {
            "BTC/USDT": make_snapshot(
                "BTC/USDT", last_close=70_000.0, ema200_4h=65_000.0, atr14_4h=1_000.0
            ),
            "ETH/USDT": make_snapshot(
                "ETH/USDT", last_close=3_500.0, ema200_4h=3_200.0, atr14_4h=50.0
            ),
        }

        # Decisions list: ETH first with tier B, BTC second with tier A
        payload = {
            "reasoning": "Market looks bullish across both majors.",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "ETH/USDT",
                    "invalidation_price": 3_400.0,  # 2 ATR below 3500
                    "conviction": "B",
                    "confidence": 0.8,
                    "thesis": "ETH thesis",
                    "invalidation": "ETH invalidation",
                    "rationale": "ETH rationale",
                },
                {
                    "action": "open_long",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 68_000.0,  # 2 ATR below 70000
                    "conviction": "A",
                    "confidence": 0.9,
                    "thesis": "BTC thesis",
                    "invalidation": "BTC invalidation",
                    "rationale": "BTC rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 1
        assert approved[0].symbol == "BTC/USDT"
        assert approved[0].conviction == ConvictionTier.A
        assert cycle_state.new_position_opened_this_cycle is True

        assert len(rejected) == 1
        assert rejected[0].symbol == "ETH/USDT"
        assert rejected[0].reason == RejectionReason.MAX_ONE_NEW_POSITION_PER_CYCLE

    def test_two_valid_opens_same_tier_first_listed_wins(self, cfg: TradingConfig):
        """When both opens have the same conviction tier, the first listed wins."""
        portfolio = make_portfolio()
        cycle_state = make_cycle_state()
        snapshots = {
            "BTC/USDT": make_snapshot(
                "BTC/USDT", last_close=70_000.0, ema200_4h=65_000.0, atr14_4h=1_000.0
            ),
            "ETH/USDT": make_snapshot(
                "ETH/USDT", last_close=3_500.0, ema200_4h=3_200.0, atr14_4h=50.0
            ),
        }

        payload = {
            "reasoning": "Two Tier A setups.",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 68_000.0,
                    "conviction": "A",
                    "confidence": 0.9,
                    "thesis": "BTC thesis",
                    "invalidation": "BTC invalidation",
                    "rationale": "BTC rationale",
                },
                {
                    "action": "open_long",
                    "symbol": "ETH/USDT",
                    "invalidation_price": 3_400.0,
                    "conviction": "A",
                    "confidence": 0.9,
                    "thesis": "ETH thesis",
                    "invalidation": "ETH invalidation",
                    "rationale": "ETH rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 1
        assert approved[0].symbol == "BTC/USDT"

        assert len(rejected) == 1
        assert rejected[0].symbol == "ETH/USDT"
        assert rejected[0].reason == RejectionReason.MAX_ONE_NEW_POSITION_PER_CYCLE

    def test_already_opened_this_cycle_rejects_new_open(self, cfg: TradingConfig):
        """If new_position_opened_this_cycle is already True, all open proposals are rejected."""
        portfolio = make_portfolio()
        cycle_state = make_cycle_state(new_opened=True)
        snapshots = {
            "BTC/USDT": make_snapshot(
                "BTC/USDT", last_close=70_000.0, ema200_4h=65_000.0, atr14_4h=1_000.0
            ),
        }

        payload = {
            "reasoning": "Another setup.",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 68_000.0,
                    "conviction": "A",
                    "confidence": 0.9,
                    "thesis": "BTC thesis",
                    "invalidation": "BTC invalidation",
                    "rationale": "BTC rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 0
        assert len(rejected) == 1
        assert rejected[0].reason == RejectionReason.MAX_ONE_NEW_POSITION_PER_CYCLE


class TestRiskEngineRule8CloseAndReopen:
    """Rule 8: Closes processed first; reopening a symbol closed this cycle rejected."""

    def test_close_processed_first_and_tracked(self, cfg: TradingConfig):
        """Closing an existing position succeeds and registers in symbols_closed_this_cycle."""
        pos = make_position(symbol="BTC/USDT", entry_price=65_000.0, quantity=0.01)
        portfolio = make_portfolio(positions={"BTC/USDT": pos})
        cycle_state = make_cycle_state()
        snapshots = {
            "BTC/USDT": make_snapshot("BTC/USDT", last_close=70_000.0),
        }

        payload = {
            "reasoning": "Taking profit.",
            "decisions": [
                {
                    "action": "close",
                    "symbol": "BTC/USDT",
                    "thesis": "Target reached",
                    "rationale": "Exit trade",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 1
        assert approved[0].action == Action.CLOSE
        assert approved[0].symbol == "BTC/USDT"
        assert approved[0].quantity == 0.01
        assert "BTC/USDT" in cycle_state.symbols_closed_this_cycle
        assert len(rejected) == 0

    def test_reopen_symbol_closed_this_cycle_rejected(self, cfg: TradingConfig):
        """Reopening a symbol that was closed earlier in this cycle is rejected."""
        portfolio = make_portfolio()
        cycle_state = make_cycle_state(symbols_closed={"BTC/USDT"})
        snapshots = {
            "BTC/USDT": make_snapshot(
                "BTC/USDT", last_close=70_000.0, ema200_4h=65_000.0, atr14_4h=1_000.0
            ),
        }

        payload = {
            "reasoning": "Re-entering immediately.",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 68_000.0,
                    "conviction": "A",
                    "confidence": 0.8,
                    "thesis": "Re-entry",
                    "invalidation": "Stop",
                    "rationale": "Re-entry rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 0
        assert len(rejected) == 1
        assert rejected[0].reason == RejectionReason.SYMBOL_CLOSED_THIS_CYCLE


class TestRiskEngineRule9MaxOpenPositions:
    """Rule 9: Maximum open positions limit."""

    def test_max_positions_reached_rejects_new_open(self):
        """When max_open_positions is reached, opening another is rejected."""
        cfg = TradingConfig(
            universe=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
            max_open_positions=3,
        )
        pos1 = make_position(symbol="BTC/USDT")
        pos2 = make_position(symbol="ETH/USDT")
        pos3 = make_position(symbol="SOL/USDT")
        portfolio = make_portfolio(positions={"BTC/USDT": pos1, "ETH/USDT": pos2, "SOL/USDT": pos3})
        cycle_state = make_cycle_state()
        snapshots = {
            "AVAX/USDT": make_snapshot("AVAX/USDT", last_close=30.0, ema200_4h=25.0, atr14_4h=1.0),
        }

        payload = {
            "reasoning": "Open 4th coin.",
            "decisions": [
                {
                    "action": "open_long",
                    "symbol": "AVAX/USDT",
                    "invalidation_price": 28.0,
                    "conviction": "A",
                    "confidence": 0.8,
                    "thesis": "AVAX thesis",
                    "invalidation": "AVAX stop",
                    "rationale": "AVAX rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 0
        assert len(rejected) == 1
        assert rejected[0].reason == RejectionReason.MAX_POSITIONS_REACHED

    def test_close_frees_slot_for_new_open(self):
        """If one of the 3 positions is closed in the same cycle, a new open is allowed."""
        cfg = TradingConfig(
            universe=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
            max_open_positions=3,
        )
        pos1 = make_position(symbol="BTC/USDT", entry_price=65_000.0, quantity=0.01)
        pos2 = make_position(symbol="ETH/USDT")
        pos3 = make_position(symbol="SOL/USDT")
        portfolio = make_portfolio(positions={"BTC/USDT": pos1, "ETH/USDT": pos2, "SOL/USDT": pos3})
        cycle_state = make_cycle_state()
        snapshots = {
            "BTC/USDT": make_snapshot("BTC/USDT", last_close=70_000.0),
            "AVAX/USDT": make_snapshot("AVAX/USDT", last_close=30.0, ema200_4h=25.0, atr14_4h=1.0),
        }

        payload = {
            "reasoning": "Rotate from BTC to AVAX.",
            "decisions": [
                {
                    "action": "close",
                    "symbol": "BTC/USDT",
                    "thesis": "Close BTC",
                    "rationale": "Take profit",
                },
                {
                    "action": "open_long",
                    "symbol": "AVAX/USDT",
                    "invalidation_price": 28.0,
                    "conviction": "A",
                    "confidence": 0.8,
                    "thesis": "AVAX thesis",
                    "invalidation": "AVAX stop",
                    "rationale": "AVAX rationale",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        # BTC close approved AND AVAX open approved
        assert len(approved) == 2
        assert approved[0].action == Action.CLOSE
        assert approved[0].symbol == "BTC/USDT"
        assert approved[1].action == Action.OPEN_LONG
        assert approved[1].symbol == "AVAX/USDT"
        assert len(rejected) == 0


class TestRiskEngineAddAndHold:
    """Add and Hold behavior in risk engine."""

    def test_hold_action_ignored_without_orders_or_rejections(self, cfg: TradingConfig):
        portfolio = make_portfolio()
        cycle_state = make_cycle_state()
        payload = {
            "reasoning": "Nothing to do.",
            "decisions": [
                {
                    "action": "hold",
                    "symbol": "BTC/USDT",
                    "rationale": "Patiently waiting",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            {},
            cycle_state,
            cfg,
        )
        assert approved == []
        assert rejected == []

    def test_valid_add_approved(self, cfg: TradingConfig):
        """Add to profitable position >= +1R with breakeven stop is approved."""
        # Entry 60,000, stop originally 58,000 (R = 2,000)
        # Current price 65,000 (+2.5R)
        # Invalidation tightened to 61,000 (above entry)
        pos = make_position(
            symbol="BTC/USDT",
            entry_price=60_000.0,
            quantity=0.01,
            invalidation_price=61_000.0,
            risk_amount=20.0,
        )
        portfolio = make_portfolio(
            equity=10_000.0,
            cash=8_000.0,
            positions={"BTC/USDT": pos},
        )
        cycle_state = make_cycle_state()
        snapshots = {
            "BTC/USDT": make_snapshot(
                "BTC/USDT",
                last_close=65_000.0,
                current_price=65_000.0,
                ema200_4h=55_000.0,
                atr14_4h=1_500.0,
            ),
        }

        payload = {
            "reasoning": "Pyramiding winner.",
            "decisions": [
                {
                    "action": "add",
                    "symbol": "BTC/USDT",
                    "invalidation_price": 62_000.0,  # 2 ATR below 65000 is 62000
                    "conviction": "A",
                    "confidence": 0.85,
                    "thesis": "Trend continuation",
                    "invalidation": "Stop 62000",
                    "rationale": "Add at support",
                },
            ],
        }

        approved, rejected = process_decisions(
            payload,
            portfolio,
            snapshots,
            cycle_state,
            cfg,
        )

        assert len(approved) == 1
        assert approved[0].action == Action.ADD
        assert approved[0].symbol == "BTC/USDT"
        assert approved[0].quantity > 0
        assert len(rejected) == 0
