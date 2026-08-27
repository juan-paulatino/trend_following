"""
Intra-candle microstructure feature set for absorption / exhaustion detection.

DESIGN PRINCIPLE
----------------
Store PRIMITIVES and PATH data; DERIVE everything else.

Primitives are the raw sums (buy_vol, sell_vol, tick counts, book sums).
Path data (cumulative-delta extremes, quartile buckets, tick runs) cannot be
reconstructed after the candle closes, so it must be captured live.

Everything else -- delta, delta/volume, agg, impact ratios -- is a pure
function of those and is exposed as a derived property. This matters because
several "different" metrics are algebraically identical:

    delta / volume  ==  2 * agg_raw - 1     (exactly, r = 1.0)
    delta           ==  volume * (2 * agg_raw - 1)

Storing them all as independent columns creates perfect collinearity.
The trade-flow content of a candle has exactly two degrees of freedom:
SCALE (volume) and BALANCE (agg).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

EPS = 1e-12
N_BUCKETS = 4  # 15s buckets within a 1-minute candle


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

class TickDirection(str, Enum):
    """Bybit v5 `tickDirection`. The zero-variants carry forward the last
    DIRECTIONAL move, which is what makes them informative: a ZeroMinusTick
    means trades are executing while the market is under downward pressure
    and price is NOT moving -- the bid is refilling as fast as it is hit.

    https://bybit-exchange.github.io/docs/v5/enum#tickdirection
    """

    PLUS = "PlusTick"              # price rose vs previous trade
    ZERO_PLUS = "ZeroPlusTick"     # same price; last directional move was up
    MINUS = "MinusTick"            # price fell vs previous trade
    ZERO_MINUS = "ZeroMinusTick"   # same price; last directional move was down

    @property
    def is_up_state(self) -> bool:
        """Directional regime, treating zero-ticks as continuations."""
        return self in (TickDirection.PLUS, TickDirection.ZERO_PLUS)

    @property
    def is_pinned(self) -> bool:
        """Trade executed without moving price. The absorption primitive."""
        return self in (TickDirection.ZERO_PLUS, TickDirection.ZERO_MINUS)


@dataclass(frozen=True)
class Trade:
    ts: float
    price: float
    size: float
    is_buy: bool  # True = aggressive buy (lifted the ask)
    # Prefer the exchange-supplied value: it carries zero-tick state across
    # candle boundaries and across gaps in our own stream. Derived locally
    # when absent.
    tick_direction: Optional[TickDirection] = None


@dataclass(frozen=True)
class BookSample:
    ts: float
    bid_sz: float
    ask_sz: float


# --------------------------------------------------------------------------
# Sub-candle bucket (path data -- captures trajectory, not just totals)
# --------------------------------------------------------------------------

@dataclass
class Bucket:
    """One sub-window of the candle. Fixed count avoids the sample-size bias
    that contaminates min/max order statistics on sparse data."""

    seconds: float = 15.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    buy_trades: int = 0
    sell_trades: int = 0
    bbo_sum: float = 0.0
    bbo_n: int = 0

    @property
    def volume(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def n_trades(self) -> int:
        return self.buy_trades + self.sell_trades

    @property
    def bbo_avg(self) -> Optional[float]:
        return self.bbo_sum / self.bbo_n if self.bbo_n else None

    @property
    def sell_rate(self) -> float:
        """Aggressive sell trades per second."""
        return self.sell_trades / max(self.seconds, EPS)

    @property
    def buy_rate(self) -> float:
        return self.buy_trades / max(self.seconds, EPS)


# --------------------------------------------------------------------------
# The candle
# --------------------------------------------------------------------------

@dataclass
class Candle:
    """
    Fields above the divider are PERSISTED primitives / path data.
    Everything below is DERIVED and must not be stored as a model feature
    alongside its parents.
    """

    ts_open: float
    ts_close: float
    tick_size: float

    # ---- price ----
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    pv_sum: float = 0.0  # sum(price * size), for VWAP

    # ---- aggressive flow totals ----
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    buy_trades: int = 0
    sell_trades: int = 0

    # ---- cumulative-delta path (NOT reconstructible post-hoc) ----
    # Convention: cumulative delta starts at 0.0 at candle open and the origin
    # is included, so delta_max >= 0 >= delta_min always.
    delta_max: float = 0.0
    delta_min: float = 0.0
    # Default to ts_open: an extreme "at the origin" means the excursion never
    # went that way at all (e.g. delta_max == 0 in a pure-sell candle).
    ts_delta_max: Optional[float] = None
    ts_delta_min: Optional[float] = None

    # Volume-weighted time sums, in seconds since candle open. These give the
    # CENTROID of each side's activity, which is far more robust than the
    # argmin/argmax of the path -- it uses all the volume instead of one point.
    buy_tw_sum: float = 0.0
    sell_tw_sum: float = 0.0

    # ---- tick structure (4-state, per Bybit tickDirection) ----
    plus_ticks: int = 0
    zero_plus_ticks: int = 0
    minus_ticks: int = 0
    zero_minus_ticks: int = 0
    undetermined_ticks: int = 0  # no prior direction to carry forward

    # Cross-tab of aggressor side against whether the aggression MOVED price.
    # This is the sharpest decomposition available: it answers "did the
    # pressing side achieve anything" on a per-trade basis.
    sell_minus: int = 0       # sell hit bid, price fell   -> supply winning
    sell_zero_minus: int = 0  # sell hit bid, price pinned -> ABSORPTION
    buy_plus: int = 0         # buy lifted ask, price rose -> demand winning
    buy_zero_plus: int = 0    # buy lifted ask, pinned     -> DISTRIBUTION

    # Depth INFERRED from executions rather than from the displayed book.
    #
    # Each trade is a measurement. A sell of size q that leaves price pinned
    # proves bid depth > q; one that ticks price down proves depth <= q. So:
    #
    #     absorbed_max_sell < true bid depth <= broke_min_sell
    #
    # This bound is unspoofable -- displayed orders can be cancelled, but a
    # fill cannot be taken back.
    absorbed_max_sell: float = 0.0            # largest sell that FAILED to move price
    broke_min_sell: float = float("inf")      # smallest sell that DID move price
    absorbed_sell_vol: float = 0.0            # total sell volume that moved nothing
    absorbed_max_buy: float = 0.0             # mirror: largest buy the ask absorbed
    broke_min_buy: float = float("inf")

    # Runs measured on the directional STATE, so zero-ticks extend a run
    # rather than breaking it -- a MinusTick/ZeroMinus/ZeroMinus sequence is
    # one continuous episode of downward pressure.
    max_run_up_state: int = 0
    max_run_down_state: int = 0

    # ---- orderbook ----
    bbo_sum: float = 0.0
    bbo_samples: int = 0
    bid_sz_sum: float = 0.0
    ask_sz_sum: float = 0.0
    bid_sz_min: float = float("inf")  # wall-pull / spoof detection
    bid_sz_max: float = 0.0

    # ---- trajectory ----
    buckets: List[Bucket] = field(default_factory=list)

    # ---- shrinkage prior strength (calibrated externally) ----
    agg_k: float = 25.0

    # ---- zero-tick carry-forward state, to seed the next candle ----
    last_tick_up: Optional[bool] = None

    # ======================================================================
    # DERIVED: scale and balance
    # ======================================================================

    @property
    def volume(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def n_trades(self) -> int:
        return self.buy_trades + self.sell_trades

    @property
    def no_trades(self) -> bool:
        """A zero-trade candle is not missing data -- it is maximal seller
        exhaustion. Flag it explicitly rather than masking it as agg=0.5."""
        return self.n_trades == 0

    @property
    def delta(self) -> float:
        """Signed net aggressive flow. NOTE: == volume * (2*agg_raw - 1)."""
        return self.buy_vol - self.sell_vol

    @property
    def agg_raw(self) -> Optional[float]:
        """Unshrunk ratio. None (never 0.5) when there is no data, because 0.5
        is a legal observable value and must not double as a sentinel."""
        if self.volume <= EPS:
            return None
        return self.buy_vol / self.volume

    @property
    def agg(self) -> Optional[float]:
        """Shrunk toward 0.5 by a finite prior of strength agg_k.

        This is what separates the three cases that collapse to 0.5 under the
        raw ratio:
            0 vs 0      -> None   (no information)
            1 vs 1      -> ~0.50  but agg_n == 2, so visibly unreliable
            800 vs 800  -> 0.500  with agg_n == 1600, a real reading
        """
        if self.volume <= EPS:
            return None
        return (self.buy_vol + self.agg_k * 0.5) / (self.volume + self.agg_k)

    @property
    def agg_n(self) -> float:
        """Confidence companion for agg. ALWAYS carry this alongside agg --
        the information is two-dimensional and no scalar can hold both."""
        return self.volume

    @property
    def delta_pct(self) -> Optional[float]:
        """Provided for interpretability ONLY. Algebraically identical to
        2*agg_raw - 1. Never use as a model feature together with agg."""
        a = self.agg_raw
        return None if a is None else 2.0 * a - 1.0

    @property
    def vwap(self) -> Optional[float]:
        return self.pv_sum / self.volume if self.volume > EPS else None

    @property
    def close_vs_vwap(self) -> Optional[float]:
        """In ticks. Positive with sell-dominated flow means sellers pressed
        and got a worse average than the close -- they pressed and failed."""
        v = self.vwap
        if v is None:
            return None
        return (self.close - v) / self.tick_size

    # ======================================================================
    # DERIVED: cumulative-delta path
    # ======================================================================

    @property
    def delta_recovery(self) -> float:
        """Net buying that arrived after the worst point. Absorption amplitude
        in volume units -- the robust replacement for 'agg finished above
        where it started'."""
        return self.delta - self.delta_min

    @property
    def delta_giveback(self) -> float:
        """Buy-side progress surrendered before the close. Upside rejection."""
        return self.delta_max - self.delta

    @property
    def delta_range(self) -> float:
        return self.delta_max - self.delta_min

    @property
    def t_sell_centroid(self) -> Optional[float]:
        """Volume-weighted mean time of aggressive selling, seconds into the
        candle."""
        if self.sell_vol <= EPS:
            return None
        return self.sell_tw_sum / self.sell_vol

    @property
    def t_buy_centroid(self) -> Optional[float]:
        if self.buy_vol <= EPS:
            return None
        return self.buy_tw_sum / self.buy_vol

    @property
    def flow_lag(self) -> Optional[float]:
        """t_buy_centroid - t_sell_centroid, in seconds. Positive means the
        selling happened FIRST and buying followed."""
        ts, tb = self.t_sell_centroid, self.t_buy_centroid
        if ts is None or tb is None:
            return None
        return tb - ts

    @property
    def capitulation_then_arrival(self) -> Optional[bool]:
        """True when selling was concentrated EARLIER in the candle than
        buying -- the sequence the model requires. False means buyers were
        leaning in during the decline and their demand is partly spent.

        Uses volume-weighted centroids rather than path extremes: a single
        outlier trade cannot flip the answer, and it stays defined when one
        side's cumulative-delta extreme sits at the origin.
        """
        lag = self.flow_lag
        return None if lag is None else lag > 0.0

    # ======================================================================
    # DERIVED: price impact (separates absorption from vacuum drift)
    # ======================================================================

    @property
    def ticks_up(self) -> float:
        return max(self.high - self.open, 0.0) / self.tick_size

    @property
    def ticks_down(self) -> float:
        return max(self.open - self.low, 0.0) / self.tick_size

    @property
    def impact_up(self) -> Optional[float]:
        """Ticks gained per unit of aggressive BUY volume. High on low volume
        = vacuum drift (thin ask). Low = buyers eating real supply."""
        if self.buy_vol <= EPS:
            return None
        return self.ticks_up / self.buy_vol

    @property
    def impact_down(self) -> Optional[float]:
        """Ticks lost per unit of aggressive SELL volume. LOW is the absorption
        signature: heavy selling, no price damage."""
        if self.sell_vol <= EPS:
            return None
        return self.ticks_down / self.sell_vol

    # ---- tick-direction derived: replenishment without an orderbook ----

    @property
    def upticks(self) -> int:
        """Strict price rises. Alias kept for readability."""
        return self.plus_ticks

    @property
    def downticks(self) -> int:
        return self.minus_ticks

    @property
    def absorption_tick_ratio(self) -> Optional[float]:
        """Of all ticks under DOWNWARD pressure, the share that left price
        pinned. High = the bid refills as fast as it is hit.

        This is a tick-resolution replenishment proxy computed from TRADES
        ALONE -- it does not suffer the blind spot of sampled book snapshots,
        where a wall hit and refilled between two samples looks untouched.
        """
        denom = self.minus_ticks + self.zero_minus_ticks
        return self.zero_minus_ticks / denom if denom else None

    @property
    def distribution_tick_ratio(self) -> Optional[float]:
        """Mirror image: buyers lifting the ask without moving price means
        supply is capping the advance."""
        denom = self.plus_ticks + self.zero_plus_ticks
        return self.zero_plus_ticks / denom if denom else None

    @property
    def sell_efficiency(self) -> Optional[float]:
        """Fraction of aggressive sells that actually moved price down.
        LOW = sellers pressing with nothing to show for it = absorption.

        A bounded proportion, so it degrades far more gracefully on sparse
        data than volume ratios do -- three trades still give a real reading.
        """
        return self.sell_minus / self.sell_trades if self.sell_trades else None

    @property
    def buy_efficiency(self) -> Optional[float]:
        return self.buy_plus / self.buy_trades if self.buy_trades else None

    @property
    def inferred_bid_depth(self) -> Optional[tuple]:
        """(lower, upper) bound on real bid depth, from executions alone.

        The lower bound is HARD: a sell of that size was absorbed without
        moving price, so at least that much was standing there.

        The upper bound is only valid if it EXCEEDS the lower bound. The two
        observations come from different moments in the candle, and on real
        data they invert about 90% of the time -- a 478-unit sell gets absorbed
        while a 153-unit sell breaks the level, because depth changed in
        between. That is not a contradiction, it just means there is no single
        interval to report. Returns upper=None in that case; see
        depth_unstable for the signal it carries.
        """
        if self.absorbed_max_sell <= 0 and self.broke_min_sell == float("inf"):
            return None
        upper = None if self.broke_min_sell == float("inf") else self.broke_min_sell
        if upper is not None and upper <= self.absorbed_max_sell:
            upper = None
        return (self.absorbed_max_sell, upper)

    @property
    def depth_unstable(self) -> bool:
        """A smaller sell broke the level than one that was absorbed.

        Depth was not merely deep or thin, it was FLICKERING within the
        candle -- present for one order and gone for a smaller one. Common on
        thin instruments, and it means a single depth number cannot describe
        the minute at all.
        """
        return (
            self.absorbed_max_sell > 0
            and self.broke_min_sell != float("inf")
            and self.broke_min_sell <= self.absorbed_max_sell
        )

    @property
    def wall_tested(self) -> Optional[float]:
        """Largest absorbed sell as a fraction of DISPLAYED bid size.

        An untested wall is not evidence. A displayed bid of 900k that only
        ever absorbed 50-unit sells proves nothing about whether it is real --
        nobody challenged it. Near 0 means bbo is unverified; near or above 1
        means the displayed depth was genuinely consumed and held.
        """
        b = self.bid_sz_avg
        if b is None or b <= EPS:
            return None
        return self.absorbed_max_sell / b

    @property
    def hidden_liquidity(self) -> bool:
        """Price refuses to move while the displayed bid is NOT large.

        Iceberg / reserve orders: a buyer who does not want to be seen. The
        highest-conviction absorption state, because displayed depth can be
        cancelled but a fill cannot be revoked -- and here there is nothing
        displayed to spoof with.
        """
        atr = self.absorption_tick_ratio
        b, a = self.bid_sz_avg, self.ask_sz_avg
        if atr is None or b is None or a is None:
            return False
        return atr > 0.6 and self.absorbed_sell_vol > 0 and b <= a

    @property
    def pinned_share(self) -> Optional[float]:
        """Overall share of trades that executed without moving price. High =
        liquid/absorbing book; low = thin book where every trade drags price."""
        determined = (
            self.plus_ticks + self.zero_plus_ticks
            + self.minus_ticks + self.zero_minus_ticks
        )
        if not determined:
            return None
        return (self.zero_plus_ticks + self.zero_minus_ticks) / determined

    @property
    def impact_asymmetry(self) -> Optional[float]:
        """impact_up / impact_down. >1 means the ask is thinner than the bid --
        a book-asymmetry proxy computable from TRADES ALONE, no depth feed."""
        iu, idn = self.impact_up, self.impact_down
        if iu is None or idn is None or idn <= EPS:
            return None
        return iu / idn

    @property
    def absorption_per_tick(self) -> float:
        """|delta_min| of NET supply swallowed per tick of downside given up.
        Add-one smoothed to stay finite when price never ticked down.

        Sharper than impact_down because it nets off offsetting buys: it is
        the true peak inventory the bid had to absorb.
        """
        return abs(self.delta_min) / (self.ticks_down + 1.0)

    # ======================================================================
    # DERIVED: exhaustion vs arrival (decomposes the agg ratio's two legs)
    # ======================================================================

    @property
    def sell_decay(self) -> Optional[float]:
        """Late sell rate / early sell rate. < 0.5 means sell participation is
        collapsing -> EXHAUSTION (agg rises via denominator shrink)."""
        if len(self.buckets) < 2:
            return None
        first, last = self.buckets[0], self.buckets[-1]
        if first.sell_rate <= EPS:
            return None
        return last.sell_rate / first.sell_rate

    @property
    def buy_growth(self) -> Optional[float]:
        """Late buy volume / early buy volume. > 1.5 means real demand is
        ARRIVING (agg rises via numerator growth). Distinguishing this from
        sell_decay is the whole exhaustion-vs-arrival question."""
        if len(self.buckets) < 2:
            return None
        first, last = self.buckets[0], self.buckets[-1]
        if first.buy_vol <= EPS:
            return None if last.buy_vol <= EPS else float("inf")
        return last.buy_vol / first.buy_vol

    @property
    def agg_trajectory(self) -> List[Optional[float]]:
        """Per-bucket shrunk agg. Replaces agg_min/agg_max, which are extreme
        order statistics whose spread scales with trade count rather than with
        market structure."""
        out: List[Optional[float]] = []
        for b in self.buckets:
            if b.volume <= EPS:
                out.append(None)
            else:
                out.append((b.buy_vol + self.agg_k * 0.5) / (b.volume + self.agg_k))
        return out

    # ======================================================================
    # DERIVED: orderbook
    # ======================================================================

    @property
    def bbo_avg(self) -> Optional[float]:
        return self.bbo_sum / self.bbo_samples if self.bbo_samples else None

    @property
    def bid_sz_avg(self) -> Optional[float]:
        return self.bid_sz_sum / self.bbo_samples if self.bbo_samples else None

    @property
    def ask_sz_avg(self) -> Optional[float]:
        return self.ask_sz_sum / self.bbo_samples if self.bbo_samples else None

    @property
    def bbo_is_wall_not_vacuum(self) -> Optional[bool]:
        """bbo is also a ratio and is blind the same way agg is: it rises
        because the bid GREW (real commitment) or because the ask EVAPORATED
        (fragile). Needs absolute sizes to disambiguate."""
        b, a = self.bid_sz_avg, self.ask_sz_avg
        if b is None or a is None:
            return None
        return b > a

    @property
    def bid_replenishment(self) -> Optional[float]:
        """Sell volume absorbed per unit of bid depth permanently consumed.
        High = the wall refills as fast as it is hit (real). A bid that never
        trades and vanishes on approach is a spoof, not absorption."""
        if self.bid_sz_max <= EPS or self.bid_sz_min == float("inf"):
            return None
        consumed = self.bid_sz_max - self.bid_sz_min
        if consumed <= EPS:
            return float("inf")  # never dented
        return self.sell_vol / consumed


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------

def _merge_fills(trades: Sequence[Trade]) -> List[Trade]:
    """Collapse consecutive prints that share (ts, price, is_buy) into one.

    These are fills of a single aggressive order against multiple makers at the
    same level. Volume is summed; the tick_direction of the FIRST fill is kept,
    because that is the one describing what the order did to the price -- the
    trailing fills are zero-ticks by construction and carry no extra evidence.
    """
    if not trades:
        return []
    out: List[Trade] = []
    for t in trades:
        p = out[-1] if out else None
        if (
            p is not None
            and p.ts == t.ts
            and p.price == t.price
            and p.is_buy == t.is_buy
        ):
            out[-1] = Trade(
                ts=p.ts,
                price=p.price,
                size=p.size + t.size,
                is_buy=p.is_buy,
                tick_direction=p.tick_direction,
            )
        else:
            out.append(t)
    return out


def build_candle(
    ts_open: float,
    ts_close: float,
    tick_size: float,
    trades: Sequence[Trade],
    book: Sequence[BookSample] = (),
    prev_close: Optional[float] = None,
    prev_tick_up: Optional[bool] = None,
    agg_k: float = 25.0,
    n_buckets: int = N_BUCKETS,
) -> Candle:
    """Aggregate raw trades + book samples into one candle with full path data.

    prev_tick_up carries zero-tick state across the candle boundary, so the
    first trade of a candle at an unchanged price is still classifiable. Feed
    it the previous candle's `last_tick_up`.
    """

    span = max(ts_close - ts_open, EPS)
    bucket_secs = span / n_buckets

    # Every path metric below -- cumulative delta, derived tick direction,
    # quartile buckets, time centroids -- depends on trades being in TIME
    # order. Websocket messages can arrive out of order, and a single message
    # may carry up to 1024 trades, so arrival order is not guaranteed to match
    # match-engine order. Sort defensively; it is cheap next to the cost of a
    # silently corrupted path.
    trades = sorted(trades, key=lambda t: t.ts)

    # Collapse fills belonging to the SAME logical order.
    #
    # One market order swept across N makers resting at one price is reported
    # as N separate prints sharing a timestamp, price and side. Counted as N
    # trades, that single order yields 1 directional tick and N-1 zero-ticks,
    # so absorption_tick_ratio reads ~95% when nothing was absorbed beyond one
    # order hitting one level. Bybit's own tickDirection field has the same
    # property, so this is not a spot-only artifact and using L would not fix
    # it -- the defect is in treating per-fill ticks as per-order evidence.
    trades = _merge_fills(trades)
    c = Candle(
        ts_open=ts_open,
        ts_close=ts_close,
        tick_size=tick_size,
        agg_k=agg_k,
        ts_delta_max=ts_open,
        ts_delta_min=ts_open,
        buckets=[Bucket(seconds=bucket_secs) for _ in range(n_buckets)],
    )

    def bucket_of(ts: float) -> Bucket:
        idx = int((ts - ts_open) / bucket_secs)
        return c.buckets[min(max(idx, 0), n_buckets - 1)]

    # ---- book samples ----
    for s in book:
        tot = s.bid_sz + s.ask_sz
        imba = s.bid_sz / tot if tot > EPS else 0.5
        c.bbo_sum += imba
        c.bbo_samples += 1
        c.bid_sz_sum += s.bid_sz
        c.ask_sz_sum += s.ask_sz
        c.bid_sz_min = min(c.bid_sz_min, s.bid_sz)
        c.bid_sz_max = max(c.bid_sz_max, s.bid_sz)
        b = bucket_of(s.ts)
        b.bbo_sum += imba
        b.bbo_n += 1

    # ---- no trades: book is still valid, price carries over ----
    if not trades:
        if prev_close is not None:
            c.open = c.high = c.low = c.close = prev_close
        return c

    c.open = trades[0].price
    c.high = max(t.price for t in trades)
    c.low = min(t.price for t in trades)
    c.close = trades[-1].price

    cum = 0.0
    run_up = run_down = 0
    last_price = prev_close if prev_close is not None else trades[0].price
    last_up: Optional[bool] = prev_tick_up  # carried across candle boundary

    for t in trades:
        c.pv_sum += t.price * t.size
        bkt = bucket_of(t.ts)

        offset = t.ts - ts_open

        if t.is_buy:
            c.buy_vol += t.size
            c.buy_trades += 1
            c.buy_tw_sum += t.size * offset
            bkt.buy_vol += t.size
            bkt.buy_trades += 1
            cum += t.size
        else:
            c.sell_vol += t.size
            c.sell_trades += 1
            c.sell_tw_sum += t.size * offset
            bkt.sell_vol += t.size
            bkt.sell_trades += 1
            cum -= t.size

        if cum > c.delta_max:
            c.delta_max, c.ts_delta_max = cum, t.ts
        if cum < c.delta_min:
            c.delta_min, c.ts_delta_min = cum, t.ts

        # ---- tick direction: did the aggression actually move price? ----
        td = t.tick_direction
        if td is None:
            # Local fallback with zero-tick carry-forward.
            if t.price > last_price:
                td = TickDirection.PLUS
            elif t.price < last_price:
                td = TickDirection.MINUS
            elif last_up is True:
                td = TickDirection.ZERO_PLUS
            elif last_up is False:
                td = TickDirection.ZERO_MINUS
            else:
                td = None  # unchanged price with no prior direction known

        if td is TickDirection.PLUS:
            c.plus_ticks += 1
            if t.is_buy:
                c.buy_plus += 1
                c.broke_min_buy = min(c.broke_min_buy, t.size)
        elif td is TickDirection.ZERO_PLUS:
            c.zero_plus_ticks += 1
            if t.is_buy:
                c.buy_zero_plus += 1
                c.absorbed_max_buy = max(c.absorbed_max_buy, t.size)
        elif td is TickDirection.MINUS:
            c.minus_ticks += 1
            if not t.is_buy:
                c.sell_minus += 1
                c.broke_min_sell = min(c.broke_min_sell, t.size)
        elif td is TickDirection.ZERO_MINUS:
            c.zero_minus_ticks += 1
            if not t.is_buy:
                c.sell_zero_minus += 1
                c.absorbed_max_sell = max(c.absorbed_max_sell, t.size)
                c.absorbed_sell_vol += t.size
        else:
            c.undetermined_ticks += 1

        if td is not None:
            if td.is_up_state:
                run_up += 1
                run_down = 0
            else:
                run_down += 1
                run_up = 0
            c.max_run_up_state = max(c.max_run_up_state, run_up)
            c.max_run_down_state = max(c.max_run_down_state, run_down)
            last_up = td.is_up_state

        last_price = t.price

    c.last_tick_up = last_up
    return c
