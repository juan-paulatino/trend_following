"""
Absorption -> exhaustion -> arrival state machine.

Every threshold in Thresholds is a GUESS pending calibration against a full
population of events (including the failures). They are gathered in one place
so calibration is a single edit, not a hunt.

Core distinctions the machine encodes:

  1. agg rising has two causes. Sell-side decay (EXHAUSTION) and buy-side
     growth (ARRIVAL) look identical in the ratio and mean different things.
     sell_decay and buy_growth separate them.

  2. Price rising has two causes. Thin ask (VACUUM, fragile) and buyers eating
     real supply (MARKUP, confirmed). impact_up plus volume separate them.

  3. bbo rising has two causes. Bid grew (wall, real) or ask evaporated
     (vacuum, fragile). Absolute sizes separate them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional

from .features import Candle

# Floor for the shrinkage prior, in VOLUME units.
EPS_K = 1.0
# Epsilon for volatility baselines, in PERCENT. Must be far smaller than any
# real move -- a 0.05% baseline is entirely plausible on a quiet instrument.
EPS_PCT = 1e-9


class Phase(str, Enum):
    NEUTRAL = "NEUTRAL"
    ABSORPTION = "ABSORPTION"      # sellers pressing into a holding bid
    EXHAUSTION = "EXHAUSTION"      # sell flow collapsing, no buyers yet
    ARRIVAL = "ARRIVAL"            # aggressive buyers actually showing up
    MARKUP = "MARKUP"              # confirmed advance on real volume
    VACUUM = "VACUUM"              # advance on thin ask, unconfirmed
    DISTRIBUTION = "DISTRIBUTION"  # buyers pressing into a holding ask
    INVALIDATED = "INVALIDATED"    # setup broke -- bid pulled or supply won
    STALE = "STALE"                # no trades; agg undefined


@dataclass
class Thresholds:
    agg_sell_press: float = 0.30
    agg_buy_press: float = 0.70
    bbo_bid_heavy: float = 0.60
    bbo_hold: float = 0.55
    bbo_broken: float = 0.45
    sell_decay_exhausted: float = 0.50
    buy_growth_arrival: float = 1.50
    absorption_pctile: float = 0.70   # rank of absorption_per_tick vs history
    volume_pctile_thin: float = 0.40  # below this = "thin", vacuum candidate
    impact_asym_vacuum: float = 2.00
    min_trades: int = 5               # below this, flow metrics are noise
    max_setup_age: int = 6            # candles before an unresolved setup dies
    # tick-direction gates
    pinned_absorbing: float = 0.60    # ZeroMinusTick share of down-pressure
    sell_eff_absorbed: float = 0.35   # sells that failed to move price
    sell_eff_breaking: float = 0.60   # sellers walking price down freely


@dataclass
class Episode:
    """Absorption accumulated ACROSS candles.

    A single candle cannot express the magnitude of an absorption event -- the
    real reading is total supply swallowed against total price given up over
    the whole window. In the reference episode that was 4 candles for -0.22%.
    """

    start_close: float
    lowest: float
    candles: int = 0
    sell_vol: float = 0.0
    buy_vol: float = 0.0
    cum_delta: float = 0.0
    cum_delta_min: float = 0.0        # deepest net-supply excursion of the episode
    absorbed_sell_vol: float = 0.0    # sell volume that moved price not at all
    absorbed_max_sell: float = 0.0    # biggest single absorbed sell -> depth floor
    sell_minus: int = 0
    sell_zero_minus: int = 0

    @property
    def price_effect_pct(self) -> float:
        """Net price given up, in percent. Negative during absorption."""
        if self.start_close <= 0:
            return 0.0
        return (self.lowest - self.start_close) / self.start_close * 100.0

    @property
    def absorption_tick_ratio(self) -> Optional[float]:
        d = self.sell_minus + self.sell_zero_minus
        return self.sell_zero_minus / d if d else None

    def supply_per_pct(self, vol_baseline_pct: float) -> Optional[float]:
        """Net supply absorbed per unit of VOLATILITY-NORMALISED price damage.

        Dividing by raw percent is not enough: -0.22% is trivial on a lively
        instrument and enormous on a quiet one. vol_baseline_pct should be the
        instrument's typical absolute move over a window of the same length.
        """
        if vol_baseline_pct <= EPS_PCT:
            return None
        z = abs(self.price_effect_pct) / vol_baseline_pct
        return abs(self.cum_delta_min) / (z + 1.0)

    def price_effect_z(self, vol_baseline_pct: float) -> Optional[float]:
        """Price effect in units of the instrument's own typical move.

        This is the correction that makes -0.22% interpretable: z near 0 means
        price barely moved for this instrument (absorption); z beyond -1 means
        it moved a normal amount or more, and 'barely moved' was an illusion
        created by looking at the raw percentage.
        """
        if vol_baseline_pct <= EPS_PCT:
            return None
        return self.price_effect_pct / vol_baseline_pct


@dataclass
class Classified:
    phase: Phase
    confident: bool
    reasons: List[str]
    episode: Optional[Episode] = None


class PhaseMachine:
    """Stateful across candles. Keeps rolling history for percentile
    normalisation, because absorption_per_tick is only meaningful relative to
    what is normal for this instrument."""

    def __init__(self, thresholds: Optional[Thresholds] = None, history: int = 240):
        self.t = thresholds or Thresholds()
        self._absorption: Deque[float] = deque(maxlen=history)
        self._volume: Deque[float] = deque(maxlen=history)
        self._range_pct: Deque[float] = deque(maxlen=history)
        self._tick_pct: float = 0.0
        self.phase: Phase = Phase.NEUTRAL
        self.setup_age: int = 0
        self.episode: Optional[Episode] = None

    def vol_baseline_pct(self, n_candles: int = 1) -> float:
        """Typical price movement over an n-candle window, in percent.

        Uses the median intrabar RANGE (high-low), not |close-open|. On a thin
        sub-penny instrument a large share of minutes close exactly where they
        opened, so the median of |close-open| collapses to zero and silently
        disables normalisation altogether -- which is precisely what happened
        the first time this ran.

        Floored at one tick: below the price grid, normalisation is
        meaningless, since price physically cannot move less than one tick.

        Scaled by sqrt(n) for the window length (random-walk scaling).
        """
        if len(self._range_pct) < 20:
            return 0.0
        ordered = sorted(self._range_pct)
        med = ordered[len(ordered) // 2]
        return max(med, self._tick_pct) * (n_candles ** 0.5)

    def _open_episode(self, c: Candle) -> None:
        self.episode = Episode(start_close=c.open, lowest=c.low)

    def _accumulate(self, c: Candle) -> None:
        e = self.episode
        if e is None:
            return
        e.candles += 1
        e.lowest = min(e.lowest, c.low) if c.low > 0 else e.lowest
        e.sell_vol += c.sell_vol
        e.buy_vol += c.buy_vol
        # The episode trough is the minimum of the running cumulative-delta
        # PATH. Within this candle the path dipped to (prev + c.delta_min), so
        # the previous total must be read BEFORE adding c.delta -- adding first
        # and then applying delta_min double-counts the candle's own selling.
        prev = e.cum_delta
        e.cum_delta += c.delta
        e.cum_delta_min = min(e.cum_delta_min, prev + c.delta_min, e.cum_delta)
        e.absorbed_sell_vol += c.absorbed_sell_vol
        e.absorbed_max_sell = max(e.absorbed_max_sell, c.absorbed_max_sell)
        e.sell_minus += c.sell_minus
        e.sell_zero_minus += c.sell_zero_minus

    # ---- rolling percentile helper ----
    @staticmethod
    def _pctile(window: Deque[float], value: float) -> Optional[float]:
        if len(window) < 20:
            return None
        return sum(1 for w in window if w <= value) / len(window)

    def calibrate_k(self) -> float:
        """Shrinkage strength ~ a quarter of typical candle volume."""
        if not self._volume:
            return 25.0
        ordered = sorted(self._volume)
        median = ordered[len(ordered) // 2]
        return max(median * 0.25, EPS_K)

    def update(self, c: Candle) -> Classified:
        reasons: List[str] = []

        abs_pct = self._pctile(self._absorption, c.absorption_per_tick)
        vol_pct = self._pctile(self._volume, c.volume)
        self._absorption.append(c.absorption_per_tick)
        self._volume.append(c.volume)
        if c.open > 0 and not c.no_trades:
            self._range_pct.append((c.high - c.low) / c.open * 100.0)
            self._tick_pct = c.tick_size / c.open * 100.0

        # ---- zero-trade candle: not missing data, it is maximal exhaustion ----
        if c.no_trades:
            bbo = c.bbo_avg
            if bbo is not None and bbo > self.t.bbo_hold and self.phase in (
                Phase.ABSORPTION, Phase.EXHAUSTION
            ):
                reasons.append("zero trades with bid still posted: peak exhaustion")
                return self._settle(Phase.EXHAUSTION, False, reasons, c)
            reasons.append("zero trades; agg undefined")
            return self._settle(Phase.STALE, False, reasons)

        agg, bbo = c.agg, c.bbo_avg
        thin_flow = c.n_trades < self.t.min_trades
        if thin_flow:
            reasons.append(f"only {c.n_trades} trades: flow metrics unreliable")

        # ---- invalidation first ----
        if bbo is not None and bbo < self.t.bbo_broken and self.phase in (
            Phase.ABSORPTION, Phase.EXHAUSTION
        ):
            reasons.append(f"bbo {bbo:.2f} collapsed: bid pulled, not absorbed")
            return self._settle(Phase.INVALIDATED, True, reasons)

        if self.phase in (Phase.ABSORPTION, Phase.EXHAUSTION):
            self.setup_age += 1
            if self.setup_age > self.t.max_setup_age:
                reasons.append("setup expired unresolved")
                return self._settle(Phase.NEUTRAL, True, reasons)

        # Tick-level invalidation: if sellers are freely walking price down,
        # the bid is NOT holding, whatever the bbo ratio claims. This catches
        # a large-but-useless bid (parked away from the touch, or spoofed)
        # that the ratio alone would happily report as support.
        eff = c.sell_efficiency
        if (
            eff is not None
            and eff > self.t.sell_eff_breaking
            and self.phase in (Phase.ABSORPTION, Phase.EXHAUSTION)
        ):
            reasons.append(
                f"{eff:.0%} of sells moved price down: bid not holding at the touch"
            )
            return self._settle(Phase.INVALIDATED, True, reasons)

        rep = c.bid_replenishment
        if rep is not None and rep < 1.0 and c.sell_vol > 0 and self.phase == Phase.ABSORPTION:
            reasons.append(f"replenishment {rep:.2f} < 1: depth consumed faster than refilled")

        # ---- ABSORPTION: heavy net selling, little price damage, bid holds ----
        #
        # Two admissible entry paths, because the book and the tape are
        # different classes of evidence:
        #
        #   (a) DISPLAYED  -- bbo bid-heavy, i.e. the promise is visible
        #   (b) EXECUTED   -- hidden liquidity: nothing displayed, but price
        #                     refuses to move anyway (iceberg / reserve)
        #
        # Path (b) is the stronger of the two and the original bbo-only gate
        # would have rejected it outright.
        atr = c.absorption_tick_ratio
        displayed_wall = bbo is not None and bbo > self.t.bbo_bid_heavy
        executed_wall = c.hidden_liquidity

        if (
            agg is not None
            and agg < self.t.agg_sell_press
            and (displayed_wall or executed_wall)
            and (abs_pct is None or abs_pct > self.t.absorption_pctile or executed_wall)
            and c.delta < 0
        ):
            reasons.append(
                f"agg {agg:.2f} sell-pressed, "
                f"{abs(c.delta_min):.0f} net supply per {c.ticks_down:.0f} ticks down"
            )
            if executed_wall:
                reasons.append(
                    "HIDDEN liquidity: price pinned while displayed bid is not "
                    "large -- iceberg absorbing, nothing to spoof with"
                )
            elif displayed_wall:
                reasons.append(f"bbo {bbo:.2f} bid-heavy")
                if c.bbo_is_wall_not_vacuum:
                    reasons.append("bid larger than ask in absolute size: real wall")

            # Tick evidence is the strongest confirmation available, and unlike
            # bbo it needs no orderbook feed and survives sparse candles.
            tick_confirms = atr is not None and atr > self.t.pinned_absorbing
            if tick_confirms:
                reasons.append(
                    f"{atr:.0%} of down-pressure ticks left price pinned "
                    f"({c.sell_zero_minus} sells absorbed vs {c.sell_minus} that moved it): "
                    f"bid refilling at tick resolution"
                )
            if eff is not None and eff < self.t.sell_eff_absorbed:
                reasons.append(f"only {eff:.0%} of sells achieved a downtick")

            # An untested wall is not evidence. Displayed depth that nobody
            # challenged proves nothing about whether it is real.
            wt = c.wall_tested
            if wt is not None and displayed_wall:
                if wt < 0.05:
                    reasons.append(
                        f"CAUTION: largest absorbed sell is only {wt:.1%} of "
                        f"displayed bid -- the wall was never tested"
                    )
                else:
                    reasons.append(
                        f"wall tested: absorbed a sell worth {wt:.0%} of displayed bid"
                    )
            lo, hi = c.inferred_bid_depth or (0.0, None)
            if lo > 0:
                bound = f">{lo:.0f}" + (f" and <={hi:.0f}" if hi else "")
                reasons.append(f"bid depth inferred from fills alone: {bound} units")

            # Sparse candles can still be trusted when the tick evidence is
            # unambiguous: absorption_tick_ratio is a bounded proportion.
            return self._settle(
                Phase.ABSORPTION, (not thin_flow) or tick_confirms, reasons, c
            )

        # ---- EXHAUSTION vs ARRIVAL: decompose why agg rose ----
        decay, growth = c.sell_decay, c.buy_growth
        if self.phase in (Phase.ABSORPTION, Phase.EXHAUSTION):
            exhausting = decay is not None and decay < self.t.sell_decay_exhausted
            arriving = growth is not None and growth > self.t.buy_growth_arrival

            if arriving and c.delta_recovery > 0:
                reasons.append(
                    f"buy volume x{growth:.1f} late: demand ARRIVING (numerator), "
                    f"delta recovered {c.delta_recovery:.0f} off the low"
                )
                if c.capitulation_then_arrival:
                    reasons.append("selling trough preceded buying peak: demand unspent")
                else:
                    reasons.append("buyers led the decline: demand partly spent")
                return self._settle(Phase.ARRIVAL, not thin_flow, reasons, c)

            if exhausting:
                reasons.append(
                    f"sell rate decayed to {decay:.2f} with flat buying: "
                    f"EXHAUSTION (denominator), no demand confirmation"
                )
                return self._settle(Phase.EXHAUSTION, not thin_flow, reasons, c)

        # ---- advance: MARKUP vs VACUUM ----
        if c.close > c.open:
            asym = c.impact_asymmetry
            thin = vol_pct is not None and vol_pct < self.t.volume_pctile_thin
            vacuum = thin or (asym is not None and asym > self.t.impact_asym_vacuum)
            if vacuum:
                reasons.append(
                    "price up on thin participation"
                    + (f", impact asymmetry {asym:.1f}x" if asym else "")
                    + ": VACUUM drift, no inventory transfer"
                )
                return self._settle(Phase.VACUUM, False, reasons)
            reasons.append("price up on real volume with contained impact: MARKUP")
            return self._settle(Phase.MARKUP, not thin_flow, reasons)

        # ---- mirror image: buyers pressing into a holding ask ----
        if (
            agg is not None
            and bbo is not None
            and agg > self.t.agg_buy_press
            and bbo < (1 - self.t.bbo_bid_heavy)
            and c.delta > 0
            and c.close <= c.open
        ):
            reasons.append(f"agg {agg:.2f} buy-pressed but price failed: DISTRIBUTION")
            dtr = c.distribution_tick_ratio
            if dtr is not None and dtr > self.t.pinned_absorbing:
                reasons.append(
                    f"{dtr:.0%} of up-pressure ticks pinned: ask refilling, supply capping"
                )
            return self._settle(Phase.DISTRIBUTION, not thin_flow, reasons)

        return self._settle(Phase.NEUTRAL, not thin_flow, reasons)

    def _settle(
        self,
        phase: Phase,
        confident: bool,
        reasons: List[str],
        c: Optional[Candle] = None,
    ) -> Classified:
        # ---- episode lifecycle ----
        entering = phase == Phase.ABSORPTION and self.phase != Phase.ABSORPTION
        if entering and c is not None:
            self._open_episode(c)
            self.setup_age = 0
        if c is not None and self.episode is not None and phase in (
            Phase.ABSORPTION, Phase.EXHAUSTION, Phase.ARRIVAL
        ):
            self._accumulate(c)

        if phase in (Phase.NEUTRAL, Phase.MARKUP, Phase.INVALIDATED, Phase.STALE):
            self.setup_age = 0

        ep = self.episode
        if ep is not None and phase in (Phase.ABSORPTION, Phase.EXHAUSTION, Phase.ARRIVAL):
            base = self.vol_baseline_pct(max(ep.candles, 1))
            z = ep.price_effect_z(base)
            if z is not None:
                reasons.append(
                    f"episode: {ep.candles} candles, {abs(ep.cum_delta_min):.0f} net "
                    f"supply absorbed for {ep.price_effect_pct:+.2f}% "
                    f"({z:+.2f} vol-normalised)"
                )
            else:
                reasons.append(
                    f"episode: {ep.candles} candles, {abs(ep.cum_delta_min):.0f} net "
                    f"supply absorbed for {ep.price_effect_pct:+.2f}% "
                    f"(no vol baseline yet)"
                )
        if phase in (Phase.NEUTRAL, Phase.MARKUP, Phase.INVALIDATED):
            ep = None
            self.episode = None

        self.phase = phase
        return Classified(phase=phase, confident=confident, reasons=reasons, episode=ep)
