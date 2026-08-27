"""
Event-clock flow measurement, paired with a time-clock decision grid.

The problem this solves: a 1-minute bar contains a random number of trades
(median 5 on POPCAT, 3 on TRX), so the variance of any flow ratio computed on
it varies by orders of magnitude between bars. agg=0.9 from 3 trades and from
300 trades are not the same measurement.

Two constructions, used together:

  ImbalanceBar  closes when cumulative signed flow crosses a threshold. The
                bar boundary is itself informative: one-sided flow closes bars
                faster. This is the primitive for "N aggressive buys surpassed
                the sells".

  RollingFlow   a trailing window of exactly N trades, readable at any instant.
                Sampled on the 1-minute grid it gives a flow reading with
                CONSTANT n, so thresholds mean the same thing in every bar.

WARNING on RollingFlow: at 5 trades/min a 50-trade window spans ~10 minutes, so
consecutive 1-minute readings overlap ~90% and are heavily autocorrelated. That
is fine for reading current state and fatal for statistical testing -- any
significance test must use non-overlapping samples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .features import Trade

EPS = 1e-12


# ---------------------------------------------------------------------------
# Event clock: bars that close on flow imbalance
# ---------------------------------------------------------------------------

@dataclass
class Imbalance:
    """One completed imbalance bar."""

    ts_open: float
    ts_close: float
    open: float
    high: float
    low: float
    close: float
    buy_vol: float
    sell_vol: float
    buy_trades: int
    sell_trades: int
    side: str            # "BUY" or "SELL" -- which side broke the threshold
    threshold: float

    @property
    def span_s(self) -> float:
        return self.ts_close - self.ts_open

    @property
    def volume(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def n_trades(self) -> int:
        return self.buy_trades + self.sell_trades

    @property
    def delta(self) -> float:
        return self.buy_vol - self.sell_vol

    @property
    def ret_pct(self) -> float:
        return (self.close - self.open) / self.open * 100 if self.open else 0.0

    @property
    def intensity(self) -> Optional[float]:
        """Threshold reached per second. High = the imbalance built fast, which
        is the part a fixed-size bar cannot express."""
        return self.threshold / self.span_s if self.span_s > EPS else None


@dataclass
class ImbalanceBarrier:
    """Emits an Imbalance bar whenever |cumulative signed flow| >= threshold.

    use_volume=True  -> signed VOLUME (size-weighted; one whale can close a bar)
    use_volume=False -> signed TRADE COUNT (each order counts once, which is
                        closer to "N aggressive buys surpass the sells")
    """

    threshold: float
    use_volume: bool = True

    _cum: float = field(default=0.0, init=False)
    _buy_vol: float = field(default=0.0, init=False)
    _sell_vol: float = field(default=0.0, init=False)
    _buy_n: int = field(default=0, init=False)
    _sell_n: int = field(default=0, init=False)
    _open: Optional[float] = field(default=None, init=False)
    _hi: float = field(default=0.0, init=False)
    _lo: float = field(default=0.0, init=False)
    _ts_open: Optional[float] = field(default=None, init=False)

    def add(self, t: Trade) -> Optional[Imbalance]:
        if self._open is None:
            self._open = self._hi = self._lo = t.price
            self._ts_open = t.ts
        self._hi = max(self._hi, t.price)
        self._lo = min(self._lo, t.price)

        step = t.size if self.use_volume else 1.0
        if t.is_buy:
            self._cum += step
            self._buy_vol += t.size
            self._buy_n += 1
        else:
            self._cum -= step
            self._sell_vol += t.size
            self._sell_n += 1

        if abs(self._cum) < self.threshold:
            return None

        bar = Imbalance(
            ts_open=self._ts_open, ts_close=t.ts,
            open=self._open, high=self._hi, low=self._lo, close=t.price,
            buy_vol=self._buy_vol, sell_vol=self._sell_vol,
            buy_trades=self._buy_n, sell_trades=self._sell_n,
            side="BUY" if self._cum > 0 else "SELL",
            threshold=self.threshold,
        )
        self._reset()
        return bar

    def _reset(self) -> None:
        self._cum = self._buy_vol = self._sell_vol = 0.0
        self._buy_n = self._sell_n = 0
        self._open = self._ts_open = None
        self._hi = self._lo = 0.0


# ---------------------------------------------------------------------------
# Event clock: trailing fixed-N window, readable on any grid
# ---------------------------------------------------------------------------

@dataclass
class RollingFlow:
    """Trailing window of exactly N trades. Read it whenever you like."""

    n: int = 50
    _buf: Deque[Trade] = field(default_factory=deque, init=False)

    def add(self, t: Trade) -> None:
        self._buf.append(t)
        while len(self._buf) > self.n:
            self._buf.popleft()

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.n

    @property
    def span_s(self) -> Optional[float]:
        if len(self._buf) < 2:
            return None
        return self._buf[-1].ts - self._buf[0].ts

    @property
    def buy_vol(self) -> float:
        return sum(t.size for t in self._buf if t.is_buy)

    @property
    def sell_vol(self) -> float:
        return sum(t.size for t in self._buf if not t.is_buy)

    @property
    def volume(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> float:
        return self.buy_vol - self.sell_vol

    @property
    def agg(self) -> Optional[float]:
        """Constant-n flow ratio. Needs no shrinkage prior: n is fixed by
        construction, so every reading carries the same confidence."""
        v = self.volume
        return None if v <= EPS else self.buy_vol / v

    @property
    def agg_count(self) -> Optional[float]:
        """Order-weighted rather than size-weighted."""
        if not self._buf:
            return None
        return sum(1 for t in self._buf if t.is_buy) / len(self._buf)

    @property
    def trade_rate(self) -> Optional[float]:
        """Trades per second across the window -- an activity measure that a
        fixed-N window gets for free."""
        s = self.span_s
        return None if not s or s <= EPS else self.n / s


# ---------------------------------------------------------------------------
# The pairing
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """An event-clock trigger evaluated against time-clock context."""

    minute_ms: int
    bar: Imbalance
    bbo_avg: Optional[float]
    bbo_rank: Optional[float]      # percentile of bbo within this instrument
    rolling_agg: Optional[float]
    fired: bool
    reasons: List[str] = field(default_factory=list)


def evaluate(
    bar: Imbalance,
    minute_ms: int,
    bbo_avg: Optional[float],
    bbo_rank: Optional[float],
    rolling_agg: Optional[float],
    require_side: str = "SELL",
    bbo_min_rank: float = 0.80,
) -> Signal:
    """Pair an imbalance-bar close with the time-bar's book state.

    Default encodes the ONLY direction the data supports: aggressive selling
    exhausting itself (SELL-side imbalance bar completes) while the book is
    bid-heavy (bbo in its top quintile). bbo_avg is the single feature that
    survived testing -- rho +0.168 at h=5, z +6.32, positive in 8/8 tapes.

    A BUY-side trigger is the opposite of every measured sign, so
    require_side="BUY" is available but expected to be unprofitable.
    """
    reasons = []
    ok_side = bar.side == require_side
    reasons.append(f"imbalance closed {bar.side}-side"
                   f"{'' if ok_side else f' (wanted {require_side})'}")

    ok_book = bbo_rank is not None and bbo_rank >= bbo_min_rank
    if bbo_rank is None:
        reasons.append("no bbo percentile yet (needs history)")
    else:
        reasons.append(f"bbo {bbo_avg:.3f} at rank {bbo_rank:.0%}"
                       f"{'' if ok_book else f' (need >={bbo_min_rank:.0%})'}")

    reasons.append(f"bar spanned {bar.span_s:.0f}s over {bar.n_trades} trades")

    return Signal(
        minute_ms=minute_ms, bar=bar, bbo_avg=bbo_avg, bbo_rank=bbo_rank,
        rolling_agg=rolling_agg, fired=ok_side and ok_book, reasons=reasons,
    )
