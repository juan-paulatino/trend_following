"""
Intra-candle depth dynamics, measured on the BOOK-UPDATE clock.

Motivating event (TRXUSDT chart, updates ~255-270): best ask fell from ~26,000
to ~2,500 while best bid rose from ~6,000 to ~11,000, with almost no trades in
between. Then price pumped.

Nothing in the existing feature set can see that:

  bbo_avg              a time-average over the candle; averaging smears the
                       collapse into a middling number
  bid_sz_avg/ask_sz_avg same
  ask_rel/bid_rel      level against a 60-CANDLE median, far too slow
  d_ask/d_bid          candle-over-candle, so a within-minute collapse and
                       partial recovery cancels out

Three design decisions, each with a reason:

LOG, not percent. Depth moves multiplicatively. 26,000 -> 2,500 is -90% while
the reverse is +940%, so no single percentage threshold treats collapses and
rebuilds alike. In logs both are +/-2.34: symmetric, additive, and roughly
normal, which is what makes z-scoring meaningful.

THE UPDATE CLOCK, not time and not volume. The event carried almost no trades,
so a volume or trade bar would not have advanced at all -- it is indexed by the
wrong event type. Book messages outnumber trades about 14:1 in these tapes, so
update-space also has far more resolution.

BOTH CHANNELS, plus their difference. log(bid/ask) is exactly logit(bbo), so the
difference of the two log-changes recovers the imbalance signal while the
separate channels keep the "which side moved" information that bbo discards.
bbo also compresses at the extremes -- 0.95 to 0.98 is 19:1 to 49:1 in odds --
and the logit un-compresses precisely the region these events live in.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

EPS = 1e-12


def _safe_log(x: float) -> Optional[float]:
    return math.log(x) if x > EPS else None


@dataclass
class DepthSnapshot:
    """One BOOK CHANGE. Heartbeat republishes (unchanged u) are not changes and
    should not advance the update clock."""

    ts: float
    bid_sz: float
    ask_sz: float

    @property
    def logit_bbo(self) -> Optional[float]:
        """log(bid/ask) -- the unbounded form of bbo."""
        lb, la = _safe_log(self.bid_sz), _safe_log(self.ask_sz)
        return None if lb is None or la is None else lb - la


@dataclass
class DepthVelocity:
    """Rolling log-change of best bid and ask depth over the last `window`
    book changes, with a z-score against a longer trailing history.

    Read it after every update; it is a continuously available statistic rather
    than a bar.
    """

    window: int = 20        # book changes, roughly the span of the chart event
    history: int = 500      # trailing sample for normalisation
    _buf: Deque[DepthSnapshot] = field(default_factory=deque, init=False)
    _hist_div: Deque[float] = field(default_factory=deque, init=False)
    _hist_bid: Deque[float] = field(default_factory=deque, init=False)
    _hist_ask: Deque[float] = field(default_factory=deque, init=False)

    def update(self, snap: DepthSnapshot) -> None:
        self._buf.append(snap)
        while len(self._buf) > self.window + 1:
            self._buf.popleft()
        d = self.divergence
        if d is not None:
            self._hist_div.append(d)
            while len(self._hist_div) > self.history:
                self._hist_div.popleft()
        for val, hist in ((self.d_log_bid, self._hist_bid),
                          (self.d_log_ask, self._hist_ask)):
            if val is not None:
                hist.append(val)
                while len(hist) > self.history:
                    hist.popleft()

    # ---- raw velocities ----

    @property
    def ready(self) -> bool:
        return len(self._buf) > self.window

    def _ends(self):
        if not self.ready:
            return None, None
        return self._buf[0], self._buf[-1]

    @property
    def d_log_bid(self) -> Optional[float]:
        """Log change in best-bid size across the window. Positive = building."""
        a, b = self._ends()
        if a is None:
            return None
        la, lb = _safe_log(a.bid_sz), _safe_log(b.bid_sz)
        return None if la is None or lb is None else lb - la

    @property
    def d_log_ask(self) -> Optional[float]:
        """Log change in best-ask size. NEGATIVE = supply withdrawing, which is
        the leading half of the chart event."""
        a, b = self._ends()
        if a is None:
            return None
        la, lb = _safe_log(a.ask_sz), _safe_log(b.ask_sz)
        return None if la is None or lb is None else lb - la

    @property
    def divergence(self) -> Optional[float]:
        """d_log_bid - d_log_ask, which equals the change in logit(bbo).

        Large positive = bid building while ask withdraws. That single number is
        the chart event, but it cannot distinguish "bid doubled" from "ask
        halved" -- use opposition() for that.
        """
        b, a = self.d_log_bid, self.d_log_ask
        return None if b is None or a is None else b - a

    @property
    def opposition(self) -> Optional[float]:
        """How much BOTH sides moved, in opposite directions.

        min(|d_log_bid|, |d_log_ask|) when the signs oppose, else 0. This is what
        makes the chart event distinctive: not one side moving, but the bid
        building AND the ask withdrawing at the same time.
        """
        b, a = self.d_log_bid, self.d_log_ask
        if b is None or a is None:
            return None
        if b > 0 > a:
            return min(abs(b), abs(a))
        if a > 0 > b:
            return -min(abs(b), abs(a))
        return 0.0

    # ---- normalised ----

    @staticmethod
    def _z(value: Optional[float], hist: Deque[float]) -> Optional[float]:
        if value is None or len(hist) < 30:
            return None
        mu = sum(hist) / len(hist)
        var = sum((x - mu) ** 2 for x in hist) / (len(hist) - 1)
        sd = var ** 0.5
        return None if sd <= EPS else (value - mu) / sd

    @property
    def divergence_z(self) -> Optional[float]:
        """Divergence in units of its own trailing standard deviation.

        The answer to "is 26,000 -> 2,500 a lot?", which requires the
        instrument's own distribution rather than an absolute threshold.
        """
        return self._z(self.divergence, self._hist_div)

    @property
    def d_log_bid_z(self) -> Optional[float]:
        return self._z(self.d_log_bid, self._hist_bid)

    @property
    def d_log_ask_z(self) -> Optional[float]:
        return self._z(self.d_log_ask, self._hist_ask)

    @property
    def span_s(self) -> Optional[float]:
        a, b = self._ends()
        return None if a is None else b.ts - a.ts


@dataclass
class DepthCusum:
    """One-sided CUSUM change detector on logit(bbo).

    A rolling window answers "how much has it moved". CUSUM answers "has a
    persistent shift begun, and how quickly can I say so" -- accumulating small
    consistent deviations rather than waiting for one large one. It is the
    standard tool for this, and it is not a bar.

    drift is the slack per observation, in logit units. Raising it ignores
    small wobbles; lowering it fires sooner and more often.
    """

    drift: float = 0.05
    threshold: float = 1.0
    _ref: Optional[float] = field(default=None, init=False)
    _pos: float = field(default=0.0, init=False)
    _neg: float = field(default=0.0, init=False)

    def update(self, snap: DepthSnapshot) -> Optional[str]:
        x = snap.logit_bbo
        if x is None:
            return None
        if self._ref is None:
            self._ref = x
            return None
        dev = x - self._ref
        self._pos = max(0.0, self._pos + dev - self.drift)
        self._neg = min(0.0, self._neg + dev + self.drift)
        if self._pos > self.threshold:
            self._reset(x)
            return "BID_BUILDING"
        if self._neg < -self.threshold:
            self._reset(x)
            return "ASK_BUILDING"
        # slow reference drift so a new regime becomes the new baseline
        self._ref += 0.01 * dev
        return None

    def _reset(self, x: float) -> None:
        self._ref = x
        self._pos = self._neg = 0.0

    @property
    def state(self) -> tuple:
        return (self._pos, self._neg)
