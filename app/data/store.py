"""SQLite Candle Store using SQLAlchemy 2.0 – Rule / Phase 2.

Features:
- Idempotent upserts keyed by (symbol, timeframe, open_time)
- Strict exclusion of in-progress candles: only closed candles are ever stored or returned
- Gap detection and automatic repair via ExchangeClient
- Historical backfilling and incremental updates to latest closed candle
- Clean DataFrame retrieval for indicator computation
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import Float, Integer, String, create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.common.logging import get_logger

if TYPE_CHECKING:
    from app.data.exchange_client import ExchangeClient

logger = get_logger(__name__)


def timeframe_to_ms(timeframe: str) -> int:
    """Convert timeframe string like '1m', '15m', '1h', '4h', '1d' to milliseconds."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60 * 1000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600 * 1000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400 * 1000
    if tf.endswith("w"):
        return int(tf[:-1]) * 7 * 86400 * 1000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def is_candle_closed(open_time_ms: int, timeframe: str, now_ms: int | None = None) -> bool:
    """Return True if the candle starting at *open_time_ms* has fully closed."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    tf_ms = timeframe_to_ms(timeframe)
    return (open_time_ms + tf_ms) <= now_ms


# ════════════════════════════════════════════════════════════════════════
# Database Model
# ════════════════════════════════════════════════════════════════════════


class Base(DeclarativeBase):
    pass


class CandleModel(Base):
    """Candle table with composite primary key (symbol, timeframe, open_time)."""

    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(16), primary_key=True)
    open_time: Mapped[int] = mapped_column(Integer, primary_key=True)  # epoch ms UTC
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)


# ════════════════════════════════════════════════════════════════════════
# CandleStore Implementation
# ════════════════════════════════════════════════════════════════════════


class CandleStore:
    """SQLite candle storage with idempotent upsert, gap detection, and repair."""

    def __init__(self, db_url: str | Engine = "sqlite:///trading_agent.db") -> None:
        if isinstance(db_url, Engine):
            self.engine = db_url
            self.db_url = str(db_url.url)
        else:
            self.db_url = db_url
            self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def upsert_candles(
        self,
        symbol: str,
        timeframe: str,
        raw_candles: list[list[float]],
        now_ms: int | None = None,
    ) -> int:
        """Idempotently insert or update closed candles.

        *raw_candles* format: [[open_time_ms, open, high, low, close, volume], ...]
        Any in-progress candle where open_time + tf_ms > now_ms is strictly ignored.
        """
        if not raw_candles:
            return 0

        if now_ms is None:
            now_ms = int(time.time() * 1000)

        tf_ms = timeframe_to_ms(timeframe)
        records: list[dict[str, Any]] = []

        for c in raw_candles:
            open_time = int(c[0])
            # Strict rule: ONLY CLOSED CANDLES ARE STORED
            if open_time + tf_ms > now_ms:
                logger.debug(
                    "Excluding in-progress candle for %s %s open_time=%d (closes at %d > now %d)",
                    symbol,
                    timeframe,
                    open_time,
                    open_time + tf_ms,
                    now_ms,
                )
                continue

            records.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "open_time": open_time,
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )

        if not records:
            return 0

        with self.Session() as session:
            stmt = sqlite_upsert(CandleModel).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "open_time"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            session.execute(stmt)
            session.commit()

        return len(records)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve stored candles as a DataFrame sorted ascending by open_time.

        Only closed candles are returned.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        tf_ms = timeframe_to_ms(timeframe)

        # Max allowed open_time is now_ms - tf_ms
        max_open_time = now_ms - tf_ms

        stmt = (
            select(CandleModel)
            .where(CandleModel.symbol == symbol)
            .where(CandleModel.timeframe == timeframe)
            .where(CandleModel.open_time <= max_open_time)
        )

        if start_time is not None:
            stmt = stmt.where(CandleModel.open_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(CandleModel.open_time <= end_time)

        if limit is not None:
            # Order desc to get latest N, then reverse in pandas
            stmt = stmt.order_by(CandleModel.open_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(CandleModel.open_time.asc())

        with self.Session() as session:
            rows = session.scalars(stmt).all()

        if not rows:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

        data = [
            {
                "open_time": r.open_time,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]

        df = pd.DataFrame(data)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
        now_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the latest closed candle dict or None."""
        df = self.get_candles(symbol, timeframe, limit=1, now_ms=now_ms)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()

    def detect_gaps(
        self,
        symbol: str,
        timeframe: str,
        start_time: int | None = None,
        end_time: int | None = None,
        now_ms: int | None = None,
    ) -> list[tuple[int, int]]:
        """Identify missing intervals in stored candles.

        Returns list of (gap_start_ms, gap_end_ms) tuples where:
        - gap_start_ms is the first missing open_time
        - gap_end_ms is the last missing open_time
        """
        df = self.get_candles(
            symbol,
            timeframe,
            start_time=start_time,
            end_time=end_time,
            now_ms=now_ms,
        )
        tf_ms = timeframe_to_ms(timeframe)
        gaps: list[tuple[int, int]] = []

        if df.empty:
            if start_time is not None and end_time is not None and start_time <= end_time:
                return [(start_time, end_time)]
            return []

        open_times = df["open_time"].to_list()

        # Check gap between start_time and first candle
        if start_time is not None and open_times[0] > start_time:
            first_gap_end = open_times[0] - tf_ms
            if start_time <= first_gap_end:
                gaps.append((start_time, first_gap_end))

        # Check internal consecutive gaps
        for i in range(len(open_times) - 1):
            curr_t = open_times[i]
            next_t = open_times[i + 1]
            diff = next_t - curr_t
            if diff > tf_ms:
                gap_start = curr_t + tf_ms
                gap_end = next_t - tf_ms
                gaps.append((gap_start, gap_end))

        # Check gap between last candle and end_time
        if end_time is not None and open_times[-1] < end_time:
            last_gap_start = open_times[-1] + tf_ms
            if last_gap_start <= end_time:
                gaps.append((last_gap_start, end_time))

        return gaps

    async def backfill(
        self,
        symbol: str,
        timeframe: str,
        days: int,
        client: ExchangeClient,
        now_ms: int | None = None,
    ) -> int:
        """Backfill *days* of closed candles from *client*.

        Returns total number of candles upserted.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        tf_ms = timeframe_to_ms(timeframe)
        start_ms = now_ms - (days * 86_400 * 1000)

        logger.info(
            "Backfilling %s %s for %d days (since %s)",
            symbol,
            timeframe,
            days,
            datetime.fromtimestamp(start_ms / 1000, tz=UTC).isoformat(),
        )

        total_upserted = 0
        since = start_ms

        while since < now_ms:
            batch = await client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,
            )
            if not batch:
                break

            upserted = self.upsert_candles(symbol, timeframe, batch, now_ms=now_ms)
            total_upserted += upserted

            last_open = int(batch[-1][0])
            next_since = last_open + tf_ms

            if next_since <= since:
                break
            since = next_since

            # If exchange returned fewer than 1000 or reached now, we may be done
            if len(batch) < 1000:
                break

        logger.info(
            "Backfill completed for %s %s: %d candles saved",
            symbol,
            timeframe,
            total_upserted,
        )
        return total_upserted

    async def update_to_latest(
        self,
        symbol: str,
        timeframe: str,
        client: ExchangeClient,
        now_ms: int | None = None,
    ) -> int:
        """Update candles up to the latest closed candle.

        Fetches starting from the candle immediately following the latest stored one.
        If no candles are stored, backfills 30 days.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        tf_ms = timeframe_to_ms(timeframe)
        latest = self.get_latest_candle(symbol, timeframe, now_ms=now_ms)

        if latest is None:
            logger.info(
                "No existing candles for %s %s; defaulting to 30 days backfill", symbol, timeframe
            )
            return await self.backfill(symbol, timeframe, days=30, client=client, now_ms=now_ms)

        since = int(latest["open_time"]) + tf_ms
        if since + tf_ms > now_ms:
            logger.debug("%s %s is already up to date", symbol, timeframe)
            return 0

        total_upserted = 0
        while since + tf_ms <= now_ms:
            batch = await client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,
            )
            if not batch:
                break

            upserted = self.upsert_candles(symbol, timeframe, batch, now_ms=now_ms)
            total_upserted += upserted

            last_open = int(batch[-1][0])
            next_since = last_open + tf_ms
            if next_since <= since:
                break
            since = next_since

            if len(batch) < 1000:
                break

        logger.info("Updated %s %s: %d new candles saved", symbol, timeframe, total_upserted)
        return total_upserted

    async def repair_gaps(
        self,
        symbol: str,
        timeframe: str,
        gaps: list[tuple[int, int]],
        client: ExchangeClient,
        now_ms: int | None = None,
    ) -> int:
        """Fetch missing candles for all identified gaps and store them.

        Returns total candles repaired.
        """
        if not gaps:
            return 0

        tf_ms = timeframe_to_ms(timeframe)
        total_repaired = 0

        for gap_start, gap_end in gaps:
            logger.info(
                "Repairing gap for %s %s from %d to %d",
                symbol,
                timeframe,
                gap_start,
                gap_end,
            )
            since = gap_start
            while since <= gap_end:
                batch = await client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000,
                )
                if not batch:
                    break

                upserted = self.upsert_candles(symbol, timeframe, batch, now_ms=now_ms)
                total_repaired += upserted

                last_open = int(batch[-1][0])
                next_since = last_open + tf_ms
                if next_since <= since:
                    break
                since = next_since

                if len(batch) < 1000:
                    break

        logger.info(
            "Gap repair complete for %s %s: %d candles filled", symbol, timeframe, total_repaired
        )
        return total_repaired


# ════════════════════════════════════════════════════════════════════════
# Convenience module-level functions
# ════════════════════════════════════════════════════════════════════════

_default_store: CandleStore | None = None


def get_default_store() -> CandleStore:
    """Return singleton default CandleStore."""
    global _default_store
    if _default_store is None:
        _default_store = CandleStore()
    return _default_store


async def backfill(
    symbol: str,
    timeframe: str,
    days: int,
    client: ExchangeClient,
    store: CandleStore | None = None,
) -> int:
    """Module-level backfill function."""
    s = store or get_default_store()
    return await s.backfill(symbol, timeframe, days, client)


async def update_to_latest(
    symbol: str,
    timeframe: str,
    client: ExchangeClient,
    store: CandleStore | None = None,
) -> int:
    """Module-level update_to_latest function."""
    s = store or get_default_store()
    return await s.update_to_latest(symbol, timeframe, client)


def detect_gaps(
    symbol: str,
    timeframe: str,
    start_time: int | None = None,
    end_time: int | None = None,
    store: CandleStore | None = None,
) -> list[tuple[int, int]]:
    """Module-level detect_gaps function."""
    s = store or get_default_store()
    return s.detect_gaps(symbol, timeframe, start_time, end_time)


async def repair_gaps(
    symbol: str,
    timeframe: str,
    gaps: list[tuple[int, int]],
    client: ExchangeClient,
    store: CandleStore | None = None,
) -> int:
    """Module-level repair_gaps function."""
    s = store or get_default_store()
    return await s.repair_gaps(symbol, timeframe, gaps, client)
