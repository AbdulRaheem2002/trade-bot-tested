"""Tests for the position sizer – Rule 6.

Validates the pure-function position sizer:
- Risk formula: equity * risk_pct * tier_mult / effective_distance
- Effective stop distance: abs(entry - inv) + slippage + fees
- Caps: single-position cap, exposure cap, cash cap
- Lot-size rounding (floor) and min-notional check
- Seeded randomized test: loss at invalidation never exceeds risk budget
"""

from __future__ import annotations

import math
import random

import pytest

from app.common.enums import Action, ConvictionTier, RejectionReason, Side
from app.common.models import Rejection, SizedOrder
from app.config.settings import TradingConfig
from app.risk.sizer import size_order


class TestSizerBasicFormulas:
    """Basic sizing calculations without caps."""

    def test_tier_a_sizing(self, cfg: TradingConfig):
        """Tier A uses 1.0x multiplier."""
        # equity = 10,000, risk = 0.75% = 75.0 USDT
        # entry = 50,000, inv = 48,000 -> raw distance = 2,000
        # slippage = 50,000 * 5/10,000 = 25
        # fees = 50,000 * 10/10,000 * 2 = 100
        # effective distance = 2,000 + 25 + 100 = 2,125
        # raw qty = 75 / 2,125 = ~0.035294...
        # floored to lot 0.001 -> 0.035
        res = size_order(
            symbol="BTC/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=50_000.0,
            invalidation_price=48_000.0,
            conviction=ConvictionTier.A,
            confidence=0.85,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=10_000.0,
            config=cfg,
            lot_size=0.001,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.quantity == 0.035
        assert res.conviction == ConvictionTier.A
        assert res.notional == pytest.approx(0.035 * 50_000.0)
        # Sized risk amount = qty * effective_distance = 0.035 * 2,125 = 74.375 <= 75.0
        assert res.risk_amount <= 75.0 + 1e-6

    def test_tier_b_sizing(self, cfg: TradingConfig):
        """Tier B uses 0.6x multiplier (0.45% risk)."""
        # equity = 10,000 -> risk = 45.0 USDT
        # effective distance = 2,125
        # raw qty = 45.0 / 2,125 = ~0.021176...
        # floored to lot 0.001 -> 0.021
        res = size_order(
            symbol="BTC/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=50_000.0,
            invalidation_price=48_000.0,
            conviction=ConvictionTier.B,
            confidence=0.7,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=10_000.0,
            config=cfg,
            lot_size=0.001,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.quantity == 0.021
        assert res.conviction == ConvictionTier.B
        assert res.risk_amount <= 45.0 + 1e-6

    def test_short_sizing(self, cfg: TradingConfig):
        """Short sizing with invalidation above entry."""
        # entry = 3,000, inv = 3,150 -> raw distance = 150
        # slippage = 3,000 * 5/10,000 = 1.5
        # fees = 3,000 * 10/10,000 * 2 = 6.0
        # effective distance = 157.5
        # risk = 75.0 -> raw qty = 75 / 157.5 = 0.47619...
        # floored to lot 0.01 -> 0.47
        res = size_order(
            symbol="ETH/USDT",
            action=Action.OPEN_SHORT,
            side=Side.SHORT,
            entry_price=3_000.0,
            invalidation_price=3_150.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=10_000.0,
            config=cfg,
            lot_size=0.01,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.quantity == 0.47
        assert res.side == Side.SHORT


class TestSizerCaps:
    """Tests for the individual sizing caps and constraints."""

    def test_single_position_cap(self, cfg: TradingConfig):
        """Order is capped by max_single_position_pct (25% default)."""
        # equity = 10,000 -> max single position = 2,000 USDT (20% default)
        # entry = 100, inv = 99.9 (tiny stop -> raw risk qty would be huge)
        # Capped to 2,000 / 100 = 20 qty
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=99.9,
            conviction=ConvictionTier.A,
            confidence=0.9,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=10_000.0,
            config=cfg,
            lot_size=0.1,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.notional <= 2_000.0 + 1e-6
        assert res.quantity == pytest.approx(20.0)

    def test_exposure_cap(self, cfg: TradingConfig):
        """Order is capped by remaining total exposure (max 70% default)."""
        # equity = 10,000, max total = 7,000. Current total exposure = 6,800.
        # Remaining room = 200 USDT.
        # entry = 100 -> max qty from exposure = 200 / 100 = 2.0
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=95.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=6_800.0,
            existing_symbol_notional=0.0,
            cash=3_200.0,
            config=cfg,
            lot_size=0.1,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.notional <= 200.0 + 1e-6
        assert res.quantity == pytest.approx(2.0)

    def test_exposure_already_at_cap(self, cfg: TradingConfig):
        """Rejects if total exposure is already at or above cap."""
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=95.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=7_000.0,  # 70% of 10,000
            existing_symbol_notional=0.0,
            cash=3_000.0,
            config=cfg,
        )
        assert isinstance(res, Rejection)
        assert res.reason == RejectionReason.BELOW_MIN_NOTIONAL
        assert "Total exposure" in res.detail

    def test_cash_cap(self, cfg: TradingConfig):
        """Order is capped by available cash after entry taker fee."""
        # Cash = 50 USDT.
        # entry = 100. taker fee = 10 bps (0.10%). fee_mult = 1.001
        # max qty = 50 / (100 * 1.001) = 0.4995...
        # floored to lot 0.1 = 0.4 qty
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=90.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=50.0,
            config=cfg,
            lot_size=0.1,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.quantity == pytest.approx(0.4)
        assert res.notional == pytest.approx(40.0)

    def test_zero_cash_rejected(self, cfg: TradingConfig):
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=90.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=0.0,
            config=cfg,
        )
        assert isinstance(res, Rejection)
        assert res.reason == RejectionReason.BELOW_MIN_NOTIONAL
        assert "No cash" in res.detail

    def test_min_notional_rejection(self, cfg: TradingConfig):
        """When calculated notional is below min_notional, order is rejected."""
        # min_notional = 100.0 USDT.
        # Cash = 30 USDT -> capped to < 30 USDT notional < min_notional.
        res = size_order(
            symbol="SOL/USDT",
            action=Action.OPEN_LONG,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=90.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=0.0,
            existing_symbol_notional=0.0,
            cash=30.0,
            config=cfg,
            lot_size=0.1,
            min_notional=100.0,
        )
        assert isinstance(res, Rejection)
        assert res.reason == RejectionReason.BELOW_MIN_NOTIONAL
        assert "min notional" in res.detail

    def test_add_action_caps_existing_notional(self, cfg: TradingConfig):
        """When adding to a position, existing symbol notional counts against single position cap."""
        # Single pos cap = 20% of 10,000 = 2,000 USDT.
        # Existing symbol notional = 1,700 USDT.
        # Room = 300 USDT.
        # entry = 100 -> max add qty = 300 / 100 = 3.0
        res = size_order(
            symbol="SOL/USDT",
            action=Action.ADD,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=90.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=1_700.0,
            existing_symbol_notional=1_700.0,
            cash=5_000.0,
            config=cfg,
            lot_size=0.1,
            min_notional=10.0,
        )
        assert isinstance(res, SizedOrder)
        assert res.quantity == pytest.approx(3.0)
        assert res.notional <= 300.0 + 1e-6

    def test_add_action_rejected_if_already_at_single_cap(self, cfg: TradingConfig):
        """Adding when existing position is at or above single position cap is rejected."""
        res = size_order(
            symbol="SOL/USDT",
            action=Action.ADD,
            side=Side.LONG,
            entry_price=100.0,
            invalidation_price=90.0,
            conviction=ConvictionTier.A,
            confidence=0.8,
            equity=10_000.0,
            total_exposure=2_000.0,
            existing_symbol_notional=2_000.0,
            cash=5_000.0,
            config=cfg,
            lot_size=0.1,
            min_notional=10.0,
        )
        assert isinstance(res, Rejection)
        assert res.reason == RejectionReason.BELOW_MIN_NOTIONAL
        assert "single-position cap" in res.detail


class TestSizerRandomizedInvariants:
    """Randomized property test: loss at invalidation never exceeds risk budget."""

    def test_loss_never_exceeds_budget_seeded_monte_carlo(self, cfg: TradingConfig):
        """Across 1000 randomized market and portfolio configurations,

        actual loss at invalidation (qty * effective_distance) NEVER exceeds
        the allocated risk budget (equity * risk_pct * tier_mult).
        """
        rng = random.Random(42)

        for _ in range(1000):
            equity = rng.uniform(100.0, 1_000_000.0)
            cash = rng.uniform(0.0, equity)
            total_exposure = rng.uniform(0.0, equity * 0.9)
            existing_notional = rng.uniform(0.0, min(total_exposure, equity * 0.25))

            entry_price = rng.uniform(0.01, 100_000.0)
            # stop distance between 0.5% and 15% of entry price
            dist_pct = rng.uniform(0.005, 0.15)
            is_long = rng.choice([True, False])
            action = Action.OPEN_LONG if is_long else Action.OPEN_SHORT
            side = Side.LONG if is_long else Side.SHORT

            if is_long:
                inv_price = entry_price * (1.0 - dist_pct)
            else:
                inv_price = entry_price * (1.0 + dist_pct)

            conviction = rng.choice([ConvictionTier.A, ConvictionTier.B])
            confidence = rng.uniform(0.1, 1.0)
            lot_size = rng.choice([0.0001, 0.001, 0.01, 0.1, 1.0])
            min_notional = rng.choice([5.0, 10.0, 20.0])

            result = size_order(
                symbol="TEST/USDT",
                action=action,
                side=side,
                entry_price=entry_price,
                invalidation_price=inv_price,
                conviction=conviction,
                confidence=confidence,
                equity=equity,
                total_exposure=total_exposure,
                existing_symbol_notional=existing_notional,
                cash=cash,
                config=cfg,
                lot_size=lot_size,
                min_notional=min_notional,
            )

            if isinstance(result, SizedOrder):
                tier_mult = cfg.conviction_multipliers[conviction]
                max_allowed_risk = equity * (cfg.risk_per_trade_pct / 100.0) * tier_mult

                # Effective distance including slippage and fees
                raw_dist = abs(entry_price - inv_price)
                slip = entry_price * (cfg.slippage_bps / 10_000.0)
                fees = entry_price * (cfg.fees_taker_bps / 10_000.0) * 2
                effective_dist = raw_dist + slip + fees
                actual_loss_at_invalidation = result.quantity * effective_dist

                # Invariant 1: Loss at invalidation <= risk budget
                assert actual_loss_at_invalidation <= max_allowed_risk + 1e-4

                # Invariant 2: Result notional + existing <= single position cap
                max_single = equity * (cfg.max_single_position_pct / 100.0)
                assert (result.notional + existing_notional) <= max_single + 1e-4

                # Invariant 3: Result notional + total exposure <= total exposure cap
                max_total = equity * (cfg.max_total_exposure_pct / 100.0)
                assert (result.notional + total_exposure) <= max_total + 1e-4

                # Invariant 4: Notional is at least min_notional
                assert result.notional >= min_notional - 1e-4

                # Invariant 5: Quantity is a valid multiple of lot_size
                if lot_size >= 1.0:
                    assert result.quantity.is_integer()
                else:
                    decimals = len(str(lot_size).split(".")[1])
                    rounded_check = round(result.quantity / lot_size) * lot_size
                    assert math.isclose(result.quantity, rounded_check, abs_tol=10 ** (-decimals))
