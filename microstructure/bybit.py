"""
Bybit v5 public websocket -> Candle assembly.

Pure parsing logic with NO dependencies, so it is fully testable offline and
can replay saved JSONL as easily as it consumes a live socket. Transport lives
in collector.py.

Schemas (verified against the v5 docs):

  publicTrade.{symbol}
    https://bybit-exchange.github.io/docs/v5/websocket/public/trade
      T    number   fill timestamp, MILLISECONDS
      S    string   side of TAKER: "Buy" | "Sell"   <- the aggressor
      v    string   trade size
      p    string   trade price
      L    string   tickDirection. "Unique field for Perps & futures"
      BT   boolean  block trade
      RPI  boolean  retail price improvement trade
    data is an array; a single message may carry up to 1024 trades.

  orderbook.1.{symbol}
    https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
      b/a  array    [price, size] as strings
      u    integer  update id
      cts  number   matching-engine timestamp, correlates with trade T
    Level 1 for linear/inverse/spot is SNAPSHOT ONLY -- no deltas to apply.

Three data-quality hazards this handles, all of which would silently corrupt
the absorption metrics:

  1. BLOCK TRADES (BT=true) are negotiated off-book and never interact with
     the visible orderbook. A large block prints enormous size at an unchanged
     price -- the exact signature of perfect absorption -- while proving
     nothing about the bid. Filtered by default.

  2. THE 3-SECOND REPUBLISH. Level 1 re-sends an unchanged snapshot with the
     SAME u every 3s. Counting those inflates bbo_samples during quiet periods
     and drags the time-average toward whatever the stale state happened to
     be. Deduplicated on u.

  3. FEED GAPS vs QUIET MINUTES. Both produce a candle with zero trades, but
     they mean opposite things: a quiet minute is real information (maximal
     seller exhaustion), a disconnect is missing data. Distinguished by book
     sample count, since orderbook.1 pushes at least every 3 seconds -- so a
     minute with no book samples means we were not connected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .features import BookSample, Candle, TickDirection, Trade, build_candle

MS_PER_MIN = 60_000
# orderbook.1 pushes on change and at least every 3s, so a healthy minute
# carries at least this many samples. Well below 20 to stay tolerant.
MIN_BOOK_SAMPLES_HEALTHY = 5

_SIDE_TO_IS_BUY = {"Buy": True, "Sell": False}


@dataclass
class Emitted:
    """A completed candle plus the provenance needed to trust it."""

    candle: Candle
    minute_start_ms: int
    feed_healthy: bool
    synthetic: bool = False  # reconstructed for a minute with no messages
    dropped_block_trades: int = 0
    dropped_duplicate_books: int = 0

    @property
    def usable(self) -> bool:
        """A zero-trade candle is usable ONLY if the feed was alive; otherwise
        it is a hole masquerading as quiet."""
        return self.feed_healthy


@dataclass
class BybitAssembler:
    """Feed it decoded websocket messages; it yields completed candles.

    Candles close on the first message belonging to a later minute, so the
    final in-progress candle is only emitted by flush().
    """

    tick_size: float
    agg_k: float = 25.0
    drop_block_trades: bool = True
    drop_rpi_trades: bool = False

    _minute: Optional[int] = field(default=None, init=False)
    _trades: List[Trade] = field(default_factory=list, init=False)
    _book: List[BookSample] = field(default_factory=list, init=False)
    _seen_u: set = field(default_factory=set, init=False)
    _prev_close: Optional[float] = field(default=None, init=False)
    _prev_tick_up: Optional[bool] = field(default=None, init=False)
    _dropped_blocks: int = field(default=0, init=False)
    _dropped_dupes: int = field(default=0, init=False)
    # Smallest non-zero price increment actually observed on the tape. Used to
    # catch a misconfigured tick_size, which silently scales every tick-based
    # metric -- a TRXUSDT run was collected at 0.00001 when the real tick is
    # 0.0001, inflating ticks_up/ticks_down and impact by 10x.
    _min_increment: Optional[float] = field(default=None, init=False)
    _last_seen_price: Optional[float] = field(default=None, init=False)

    def observed_tick_size(self) -> Optional[float]:
        return self._min_increment

    def tick_size_warning(self, tol: float = 1.5) -> Optional[str]:
        """Non-None when the configured tick_size disagrees with the tape.

        Only a heuristic: the true tick is the GCD of observed increments, and
        the smallest observed increment is an upper bound on it. So this can
        miss a too-large configured value early in a run, but it reliably
        catches an order-of-magnitude mistake.
        """
        obs = self._min_increment
        if obs is None or self.tick_size <= 0:
            return None
        ratio = obs / self.tick_size
        if ratio > tol:
            return (f"configured tick_size={self.tick_size:g} but the smallest "
                    f"increment seen is {obs:g} ({ratio:.0f}x larger). "
                    f"Tick-based metrics are inflated by that factor.")
        if ratio < 1 / tol:
            return (f"configured tick_size={self.tick_size:g} is LARGER than the "
                    f"smallest observed increment {obs:g}; prices move in "
                    f"finer steps than configured.")
        return None

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    def on_message(self, msg: Dict) -> List[Emitted]:
        """Route by topic. Accepts the raw decoded message dict."""
        topic = msg.get("topic", "")
        if topic.startswith("publicTrade"):
            return self.on_trades(msg)
        if topic.startswith("orderbook"):
            return self.on_orderbook(msg)
        return []  # subscription acks, pongs, anything else

    def on_trades(self, msg: Dict) -> List[Emitted]:
        rows = msg.get("data") or []
        if not rows:
            return []

        out = self._advance(int(rows[0]["T"]))

        for r in rows:
            if self.drop_block_trades and r.get("BT"):
                self._dropped_blocks += 1
                continue
            if self.drop_rpi_trades and r.get("RPI"):
                continue

            is_buy = _SIDE_TO_IS_BUY.get(r["S"])
            if is_buy is None:  # unknown side, refuse to guess
                continue

            # L is absent on spot. Leave it None and let build_candle derive
            # tick direction locally with carry-forward.
            raw_dir = r.get("L")
            try:
                td = TickDirection(raw_dir) if raw_dir else None
            except ValueError:
                td = None

            price = float(r["p"])
            if self._last_seen_price is not None:
                gap = abs(price - self._last_seen_price)
                # guard against float noise on small prices
                if gap > 1e-12 and (self._min_increment is None
                                    or gap < self._min_increment):
                    self._min_increment = gap
            self._last_seen_price = price

            self._trades.append(
                Trade(
                    ts=int(r["T"]) / 1000.0,  # ms -> seconds
                    price=price,
                    size=float(r["v"]),
                    is_buy=is_buy,
                    tick_direction=td,
                )
            )
        return out

    def on_orderbook(self, msg: Dict) -> List[Emitted]:
        data = msg.get("data") or {}
        bids, asks = data.get("b") or [], data.get("a") or []
        if not bids or not asks:
            return []

        # cts is the matching-engine timestamp and is the field the docs say
        # correlates with trade T; fall back to ts.
        ts_ms = int(msg.get("cts") or msg.get("ts") or 0)
        if not ts_ms:
            return []

        out = self._advance(ts_ms)

        # The 3-second republish carries an identical u. Counting it would
        # bias the time-averaged bbo toward stale quiet-period state.
        u = data.get("u")
        if u is not None:
            if u in self._seen_u:
                self._dropped_dupes += 1
                return out
            self._seen_u.add(u)

        self._book.append(
            BookSample(
                ts=ts_ms / 1000.0,
                bid_sz=float(bids[0][1]),
                ask_sz=float(asks[0][1]),
            )
        )
        return out

    # ------------------------------------------------------------------
    # candle boundaries
    # ------------------------------------------------------------------

    def _advance(self, ts_ms: int) -> List[Emitted]:
        minute = (ts_ms // MS_PER_MIN) * MS_PER_MIN

        if self._minute is None:
            self._minute = minute
            return []
        if minute <= self._minute:
            return []  # still inside the current candle (or a late arrival)

        out = [self._close_current()]

        # Minutes with no messages at all. Emit them so the series has no
        # holes, flagged unhealthy because silence on orderbook.1 means we
        # were disconnected rather than that the market was quiet.
        gap = self._minute + MS_PER_MIN
        while gap < minute:
            out.append(self._synthetic_gap(gap))
            gap += MS_PER_MIN

        self._minute = minute
        return out

    def _close_current(self) -> Emitted:
        assert self._minute is not None
        start = self._minute
        candle = build_candle(
            ts_open=start / 1000.0,
            ts_close=(start + MS_PER_MIN) / 1000.0,
            tick_size=self.tick_size,
            trades=self._trades,
            book=self._book,
            prev_close=self._prev_close,
            prev_tick_up=self._prev_tick_up,
            agg_k=self.agg_k,
        )

        healthy = len(self._book) >= MIN_BOOK_SAMPLES_HEALTHY
        emitted = Emitted(
            candle=candle,
            minute_start_ms=start,
            feed_healthy=healthy,
            dropped_block_trades=self._dropped_blocks,
            dropped_duplicate_books=self._dropped_dupes,
        )

        if self._trades:
            self._prev_close = candle.close
            self._prev_tick_up = candle.last_tick_up
        self._trades, self._book = [], []
        self._seen_u.clear()
        self._dropped_blocks = self._dropped_dupes = 0
        return emitted

    def _synthetic_gap(self, start: int) -> Emitted:
        candle = build_candle(
            ts_open=start / 1000.0,
            ts_close=(start + MS_PER_MIN) / 1000.0,
            tick_size=self.tick_size,
            trades=[],
            book=[],
            prev_close=self._prev_close,
            prev_tick_up=self._prev_tick_up,
            agg_k=self.agg_k,
        )
        return Emitted(
            candle=candle,
            minute_start_ms=start,
            feed_healthy=False,
            synthetic=True,
        )

    def flush(self) -> List[Emitted]:
        """Close the in-progress candle. Call on shutdown."""
        if self._minute is None:
            return []
        out = [self._close_current()]
        self._minute = None
        return out
