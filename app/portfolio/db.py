"""SQLAlchemy 2.0 Database models and initialization for Portfolio & System State – Rule / Phase 3."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

CURRENT_SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class SchemaVersionModel(Base):
    """Tracks applied schema migrations and database version."""

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    description: Mapped[str] = mapped_column(String(255), default="")


class OrderModel(Base):
    """Executed or pending orders."""

    __tablename__ = "orders"

    client_order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="filled")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class FillModel(Base):
    """Execution fills with price, fees, and slippage."""

    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_bps: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PositionModel(Base):
    """Current and historical position states."""

    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    side: Mapped[str] = mapped_column(String(16))
    entry_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    invalidation_price: Mapped[float] = mapped_column(Float)
    conviction: Mapped[str] = mapped_column(String(8))
    risk_amount: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN or CLOSED


class TradeModel(Base):
    """Closed round-trip trades with performance attribution."""

    __tablename__ = "trades"

    trade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    realized_pnl: Mapped[float] = mapped_column(Float)
    r_multiple: Mapped[float] = mapped_column(Float)
    fees_total: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(64), default="")


class EquitySnapshotModel(Base):
    """Point-in-time portfolio equity and exposure snapshots."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    exposure: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)


class DecisionModel(Base):
    """LLM decision logs per reflex cycle."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_response: Mapped[str] = mapped_column(Text)
    parsed_json: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)


class RejectionModel(Base):
    """Rejection audit log per reflex cycle."""

    __tablename__ = "rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)


class BrainReportModel(Base):
    """Daily Brain market review and thesis report."""

    __tablename__ = "brain_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    market_bias: Mapped[str] = mapped_column(String(32))
    theses_json: Mapped[str] = mapped_column(Text)
    report_json: Mapped[str] = mapped_column(Text)


class SystemStateModel(Base):
    """Persistent agent state across restarts."""

    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    mode: Mapped[str] = mapped_column(String(16), default="paper")
    breaker_state_json: Mapped[str] = mapped_column(Text, default="{}")
    kill_switch: Mapped[bool] = mapped_column(Boolean, default=False)
    last_cycle_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_state: Mapped[str] = mapped_column(String(32), default="RUNNING")
    cash: Mapped[float] = mapped_column(Float, default=10000.0)
    peak_equity: Mapped[float] = mapped_column(Float, default=10000.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


def init_portfolio_db(engine_or_url: Engine | str) -> Engine:
    """Initialize portfolio database tables and record schema version."""
    if isinstance(engine_or_url, str):
        connect_args: dict[str, Any] = {}
        if engine_or_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        engine = create_engine(engine_or_url, connect_args=connect_args)
    else:
        engine = engine_or_url

    Base.metadata.create_all(engine)

    # Automatic schema migration for Phase 5 additions
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(
                text(
                    "ALTER TABLE system_state ADD COLUMN agent_state VARCHAR(32) DEFAULT 'RUNNING'"
                )
            )
            conn.commit()
    except Exception:
        pass

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        existing = session.scalar(
            select(SchemaVersionModel).where(SchemaVersionModel.version == CURRENT_SCHEMA_VERSION)
        )
        if not existing:
            version_entry = SchemaVersionModel(
                version=CURRENT_SCHEMA_VERSION,
                description="Phase 3 portfolio and execution persistence schema",
                applied_at=datetime.now(UTC),
            )
            session.add(version_entry)
            session.commit()

    return engine
