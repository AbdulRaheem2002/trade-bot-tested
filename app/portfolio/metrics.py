"""Performance metrics and trade journal export – Rule / Phase 3.

Computes:
- Total Return
- Win Rate
- Profit Factor
- Average R
- Expectancy (R and PnL)
- Max Drawdown
- Sharpe & Sortino (annualized from equity snapshots)
- Exposure Time
- Fees Paid
- Number of Trades
Exports trade journal to CSV.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.common.models import ClosedTrade
from app.portfolio.db import EquitySnapshotModel, TradeModel


@dataclass
class PerformanceMetrics:
    """Summary of trading performance."""

    total_return_pct: float
    win_rate_pct: float
    profit_factor: float
    average_r: float
    expectancy_r: float
    expectancy_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    exposure_time_pct: float
    total_fees_paid: float
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    trades: Sequence[TradeModel | ClosedTrade | Any],
    snapshots: Sequence[EquitySnapshotModel | Any],
    periods_per_year: int = 2190,  # 6 4h candles/day * 365 days
) -> PerformanceMetrics:
    """Compute all performance metrics from lists of trades and equity snapshots."""
    # ── 1. Trade-level statistics ────────────────────────────────────────────
    n_trades = len(trades)
    winning = [t for t in trades if t.realized_pnl > 0]
    losing = [t for t in trades if t.realized_pnl < 0]

    n_win = len(winning)
    n_loss = len(losing)
    win_rate_pct = (n_win / n_trades * 100.0) if n_trades > 0 else 0.0

    gross_profit = sum(t.realized_pnl for t in winning)
    gross_loss = abs(sum(t.realized_pnl for t in losing))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_r = (sum(t.r_multiple for t in trades) / n_trades) if n_trades > 0 else 0.0

    # Expectancy calculation
    if n_trades > 0:
        win_rate = n_win / n_trades
        loss_rate = n_loss / n_trades
        avg_win_r = (sum(t.r_multiple for t in winning) / n_win) if n_win > 0 else 0.0
        avg_loss_r = (abs(sum(t.r_multiple for t in losing)) / n_loss) if n_loss > 0 else 0.0
        expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

        avg_win_pnl = (gross_profit / n_win) if n_win > 0 else 0.0
        avg_loss_pnl = (gross_loss / n_loss) if n_loss > 0 else 0.0
        expectancy_pnl = (win_rate * avg_win_pnl) - (loss_rate * avg_loss_pnl)
    else:
        expectancy_r = 0.0
        expectancy_pnl = 0.0

    total_fees_paid = sum(t.fees_total for t in trades)

    # ── 2. Snapshot-level statistics ─────────────────────────────────────────
    equities = [float(s.equity) for s in snapshots]
    if len(equities) >= 1:
        initial_eq = equities[0]
        final_eq = equities[-1]
        total_return_pct = ((final_eq - initial_eq) / initial_eq) * 100.0 if initial_eq > 0 else 0.0
    else:
        total_return_pct = 0.0

    # Max Drawdown
    max_dd = 0.0
    if equities:
        peak = equities[0]
        for eq in equities:
            if eq > peak:
                peak = eq
            elif peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
    max_drawdown_pct = max_dd * 100.0

    # Sharpe and Sortino ratios
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]

    sharpe = 0.0
    sortino = 0.0

    if len(returns) >= 2:
        mean_ret = sum(returns) / len(returns)
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(var_ret)

        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * math.sqrt(periods_per_year)

        downside_sq = [min(0.0, r) ** 2 for r in returns]
        downside_dev = math.sqrt(sum(downside_sq) / len(downside_sq))
        if downside_dev > 0:
            sortino = (mean_ret / downside_dev) * math.sqrt(periods_per_year)

    # Exposure Time
    if snapshots:
        in_market = sum(1 for s in snapshots if getattr(s, "exposure", 0.0) > 0)
        exposure_time_pct = (in_market / len(snapshots)) * 100.0
    else:
        exposure_time_pct = 0.0

    return PerformanceMetrics(
        total_return_pct=round(total_return_pct, 4),
        win_rate_pct=round(win_rate_pct, 2),
        profit_factor=round(profit_factor, 4),
        average_r=round(avg_r, 4),
        expectancy_r=round(expectancy_r, 4),
        expectancy_pnl=round(expectancy_pnl, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        exposure_time_pct=round(exposure_time_pct, 2),
        total_fees_paid=round(total_fees_paid, 4),
        number_of_trades=n_trades,
        winning_trades=n_win,
        losing_trades=n_loss,
        gross_profit=round(gross_profit, 4),
        gross_loss=round(gross_loss, 4),
    )


def compute_metrics_from_db(
    engine: Engine,
    periods_per_year: int = 2190,
) -> PerformanceMetrics:
    """Compute performance metrics directly from SQLite database."""
    with Session(engine) as session:
        trades = session.scalars(select(TradeModel).order_by(TradeModel.exit_time.asc())).all()
        snapshots = session.scalars(
            select(EquitySnapshotModel).order_by(EquitySnapshotModel.timestamp.asc())
        ).all()
        return compute_metrics(trades, snapshots, periods_per_year=periods_per_year)


def export_trade_journal_csv(
    trades: Sequence[TradeModel | ClosedTrade | Any],
    output_path: str | Path | None = None,
) -> str:
    """Export closed trades to a formatted CSV trade journal string or file."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")

    # Header
    writer.writerow(
        [
            "trade_id",
            "symbol",
            "side",
            "entry_price",
            "exit_price",
            "quantity",
            "entry_time",
            "exit_time",
            "realized_pnl",
            "r_multiple",
            "fees_total",
            "exit_reason",
        ]
    )

    for t in trades:
        writer.writerow(
            [
                getattr(t, "trade_id", ""),
                getattr(t, "symbol", ""),
                getattr(t.side, "value", str(getattr(t, "side", ""))),
                f"{float(getattr(t, 'entry_price', 0.0)):.4f}",
                f"{float(getattr(t, 'exit_price', 0.0)):.4f}",
                f"{float(getattr(t, 'quantity', 0.0)):.6f}",
                str(getattr(t, "entry_time", "")),
                str(getattr(t, "exit_time", "")),
                f"{float(getattr(t, 'realized_pnl', 0.0)):.4f}",
                f"{float(getattr(t, 'r_multiple', 0.0)):.4f}",
                f"{float(getattr(t, 'fees_total', 0.0)):.4f}",
                str(getattr(t, "exit_reason", "")),
            ]
        )

    csv_content = output.getvalue()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(csv_content.encode("utf-8"))

    return csv_content
