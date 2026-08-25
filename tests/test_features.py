"""
Verification of the properties that matter, especially the edge cases that the
original agg-defaults-to-0.5 formulation got wrong.

Run:  python test_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microstructure import (  # noqa: E402
    BookSample,
    Phase,
    PhaseMachine,
    Trade,
    build_candle,
)
from microstructure import TickDirection as TD  # noqa: E402

TICK = 0.00001
T0, T1 = 0.0, 60.0
ok, fail = 0, 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


def mk(trades, book=(), prev=None, k=25.0):
    return build_candle(T0, T1, TICK, trades, book, prev_close=prev, agg_k=k)


# ==========================================================================
print("\n1. delta/volume is algebraically identical to 2*agg-1")
# ==========================================================================
for b, s in [(700, 300), (10, 990), (1, 1), (450, 450)]:
    trades = (
        [Trade(1.0, 0.049, b, True)] if b else []
    ) + ([Trade(2.0, 0.049, s, False)] if s else [])
    c = mk(trades)
    lhs = c.delta_pct
    rhs = 2 * c.agg_raw - 1
    check(
        f"b={b} s={s}: delta/vol={lhs:+.6f} == 2*agg-1={rhs:+.6f}",
        abs(lhs - rhs) < 1e-12,
    )
print("  -> they are ONE degree of freedom; never use both as model features")


# ==========================================================================
print("\n2. The three states that collapsed to 0.5 are now distinguishable")
# ==========================================================================
empty = mk([])
sparse = mk([Trade(1.0, 0.049, 1, True), Trade(2.0, 0.049, 1, False)])
heavy = mk([Trade(1.0, 0.049, 800, True), Trade(2.0, 0.049, 800, False)])

check("no trades -> agg is None, not 0.5", empty.agg is None, f"got {empty.agg}")
check("no trades -> no_trades flag set", empty.no_trades)
check("sparse 1v1 -> raw agg is exactly 0.5", abs(sparse.agg_raw - 0.5) < 1e-12)
check("heavy 800v800 -> raw agg is also exactly 0.5", abs(heavy.agg_raw - 0.5) < 1e-12)
check(
    f"but agg_n separates them: {sparse.agg_n:.0f} vs {heavy.agg_n:.0f}",
    heavy.agg_n > sparse.agg_n * 100,
)
print(f"  sparse: agg={sparse.agg:.4f} n={sparse.agg_n:.0f}")
print(f"  heavy : agg={heavy.agg:.4f} n={heavy.agg_n:.0f}")


# ==========================================================================
print("\n3. Shrinkage kills the sparse-extreme artifact")
# ==========================================================================
# The log showed agg=0.077 and agg=0.936 in adjacent minutes. On 3 trades that
# is noise, not a 90/10 flow imbalance.
noise = mk([Trade(i + 1.0, 0.049, 1, False) for i in range(3)], k=25.0)
real = mk([Trade(1.0, 0.049, 900, False), Trade(2.0, 0.049, 60, True)], k=25.0)
check(
    f"3 tiny sells: raw={noise.agg_raw:.3f} -> shrunk={noise.agg:.3f} pulled to neutral",
    noise.agg > 0.40,
)
check(
    f"960 units of real selling: raw={real.agg_raw:.3f} -> shrunk={real.agg:.3f} survives",
    real.agg < 0.10,
)
print("  -> same raw reading, opposite reliability, now visible")


# ==========================================================================
print("\n4. Cumulative-delta path captures what totals cannot")
# ==========================================================================
# Identical b/s totals, opposite ORDER -> identical agg, different path.
sell_then_buy = mk(
    [Trade(1.0, 0.0490, 500, False), Trade(30.0, 0.0489, 300, False),
     Trade(50.0, 0.0490, 800, True)]
)
buy_then_sell = mk(
    [Trade(1.0, 0.0490, 800, True), Trade(30.0, 0.0491, 500, False),
     Trade(50.0, 0.0490, 300, False)]
)
check(
    f"same agg both ways ({sell_then_buy.agg_raw:.3f})",
    abs(sell_then_buy.agg_raw - buy_then_sell.agg_raw) < 1e-9,
)
check(
    f"but delta_min differs: {sell_then_buy.delta_min:.0f} vs {buy_then_sell.delta_min:.0f}",
    sell_then_buy.delta_min < buy_then_sell.delta_min,
)
check(
    f"recovery differs: {sell_then_buy.delta_recovery:.0f} vs {buy_then_sell.delta_recovery:.0f}",
    sell_then_buy.delta_recovery > buy_then_sell.delta_recovery,
)
check(
    f"capitulation-then-arrival: flow_lag={sell_then_buy.flow_lag:+.1f}s",
    sell_then_buy.capitulation_then_arrival is True,
)
check(
    f"buyers-led-decline: flow_lag={buy_then_sell.flow_lag:+.1f}s",
    buy_then_sell.capitulation_then_arrival is False,
)
print(
    f"  sell centroid {sell_then_buy.t_sell_centroid:.1f}s vs buy "
    f"{sell_then_buy.t_buy_centroid:.1f}s  (selling first)"
)
print(
    f"  sell centroid {buy_then_sell.t_sell_centroid:.1f}s vs buy "
    f"{buy_then_sell.t_buy_centroid:.1f}s  (buying first, demand spent)"
)


# ==========================================================================
print("\n5. Absorption vs vacuum from trades alone")
# ==========================================================================
# Absorption: 1500 units of selling, price gives up 1 tick.
absorb = mk(
    [Trade(1.0, 0.04902, 500, False), Trade(15.0, 0.04901, 500, False),
     Trade(30.0, 0.04901, 500, False), Trade(50.0, 0.04902, 120, True)],
    book=[BookSample(t, 900_000, 300_000) for t in (5, 20, 35, 55)],
)
# Vacuum: 40 units of buying walks price up 5 ticks.
vac = mk(
    [Trade(1.0, 0.04900, 10, True), Trade(20.0, 0.04902, 10, True),
     Trade(40.0, 0.04905, 20, True)],
    book=[BookSample(t, 200_000, 40_000) for t in (5, 20, 35, 55)],
)
check(
    f"absorption: impact_down tiny ({absorb.impact_down:.2e} ticks/unit)",
    absorb.impact_down < 1e-3,
)
check(
    f"vacuum: impact_up large ({vac.impact_up:.2e} ticks/unit)",
    vac.impact_up > absorb.impact_down * 100,
)
check(
    f"absorption_per_tick = {absorb.absorption_per_tick:.0f} units/tick",
    absorb.absorption_per_tick > 500,
)
check(f"bbo bid-heavy on the wall ({absorb.bbo_avg:.3f})", absorb.bbo_avg > 0.7)
check("absolute sizes confirm wall, not evaporated ask", absorb.bbo_is_wall_not_vacuum is True)


# ==========================================================================
print("\n6. Exhaustion (denominator) vs arrival (numerator)")
# ==========================================================================
# Sell rate collapses, nobody buys.
exhaust = mk(
    [Trade(1.0, 0.049, 300, False), Trade(3.0, 0.049, 300, False),
     Trade(5.0, 0.049, 200, False), Trade(7.0, 0.049, 200, False),
     Trade(50.0, 0.049, 20, False)]
)
# Sell rate steady, buyers pile in late.
arrive = mk(
    [Trade(1.0, 0.049, 200, False), Trade(3.0, 0.049, 100, True),
     Trade(50.0, 0.0491, 600, True), Trade(55.0, 0.0492, 700, True),
     Trade(58.0, 0.049, 200, False)]
)
check(f"exhaustion: sell_decay={exhaust.sell_decay:.2f} < 0.5", exhaust.sell_decay < 0.5)
check(f"arrival: buy_growth={arrive.buy_growth:.1f} > 1.5", arrive.buy_growth > 1.5)
print("  -> both would raise agg; only these two split the cause")


# ==========================================================================
print("\n7. Zero-trade candle is a feature, not a hole")
# ==========================================================================
m = PhaseMachine()
m.phase = Phase.ABSORPTION
quiet = mk([], book=[BookSample(t, 800_000, 200_000) for t in (5, 25, 45)], prev=0.04894)
res = m.update(quiet)
check("zero trades + posted bid -> EXHAUSTION, not neutral", res.phase == Phase.EXHAUSTION)
check(f"bbo still valid without any trades ({quiet.bbo_avg:.3f})", quiet.bbo_avg > 0.75)
check("price carried forward from prev_close", quiet.close == 0.04894)
print(f"  reason: {res.reasons[0]}")


# ==========================================================================
print("\n8. Invalidation: bid pulled rather than absorbed")
# ==========================================================================
m2 = PhaseMachine()
m2.phase = Phase.ABSORPTION
pulled = mk(
    [Trade(1.0, 0.0489, 400, False), Trade(40.0, 0.0488, 400, False)],
    book=[BookSample(t, 50_000, 900_000) for t in (5, 25, 45)],
)
r2 = m2.update(pulled)
check("bbo collapse while agg still low -> INVALIDATED", r2.phase == Phase.INVALIDATED)
print(f"  reason: {r2.reasons[0]}")


# ==========================================================================
print("\n9. Tick direction: ZeroMinusTick IS the absorption signal")
# ==========================================================================
# Same sell volume, same agg, same price path endpoints -- but one grinds price
# down tick by tick, the other gets pinned against a refilling bid.
walked_down = mk(
    [
        Trade(1.0, 0.04901, 300, False, TD.MINUS),
        Trade(10.0, 0.04900, 300, False, TD.MINUS),
        Trade(20.0, 0.04899, 300, False, TD.MINUS),
        Trade(30.0, 0.04898, 300, False, TD.MINUS),
    ]
)
pinned = mk(
    [
        Trade(1.0, 0.04901, 300, False, TD.MINUS),
        Trade(10.0, 0.04901, 300, False, TD.ZERO_MINUS),
        Trade(20.0, 0.04901, 300, False, TD.ZERO_MINUS),
        Trade(30.0, 0.04901, 300, False, TD.ZERO_MINUS),
    ]
)
check(
    f"identical agg ({walked_down.agg_raw:.3f} vs {pinned.agg_raw:.3f})",
    abs(walked_down.agg_raw - pinned.agg_raw) < 1e-12,
)
check(
    f"identical delta ({walked_down.delta:.0f})",
    abs(walked_down.delta - pinned.delta) < 1e-9,
)
check(
    f"walked down: absorption_tick_ratio={walked_down.absorption_tick_ratio:.2f}",
    walked_down.absorption_tick_ratio == 0.0,
)
check(
    f"pinned: absorption_tick_ratio={pinned.absorption_tick_ratio:.2f}",
    pinned.absorption_tick_ratio == 0.75,
)
check(
    f"sell_efficiency {walked_down.sell_efficiency:.0%} vs {pinned.sell_efficiency:.0%}",
    walked_down.sell_efficiency > pinned.sell_efficiency,
)
print("  -> agg and delta cannot tell these apart; tick direction can")

# ==========================================================================
print("\n10. Zero-tick carry-forward across the candle boundary")
# ==========================================================================
# First trade at an unchanged price is unclassifiable without prior state.
orphan = mk([Trade(1.0, 0.04900, 100, False)], prev=0.04900)
check("no prior direction -> undetermined, not silently mislabelled",
      orphan.undetermined_ticks == 1)

seeded = build_candle(
    T0, T1, TICK,
    [Trade(1.0, 0.04900, 100, False)],
    prev_close=0.04900,
    prev_tick_up=False,  # previous candle ended under downward pressure
)
check("seeded with prev_tick_up -> classified as ZeroMinusTick",
      seeded.zero_minus_ticks == 1)
check("last_tick_up exported to seed the next candle",
      seeded.last_tick_up is False)

# ==========================================================================
print("\n11. Runs measured on direction STATE, not strict price change")
# ==========================================================================
mixed = mk(
    [
        Trade(1.0, 0.04901, 100, False, TD.MINUS),
        Trade(5.0, 0.04901, 100, False, TD.ZERO_MINUS),
        Trade(9.0, 0.04901, 100, False, TD.ZERO_MINUS),
        Trade(13.0, 0.04900, 100, False, TD.MINUS),
        Trade(50.0, 0.04902, 100, True, TD.PLUS),
    ]
)
check(f"one continuous down-pressure episode of 4 (got {mixed.max_run_down_state})",
      mixed.max_run_down_state == 4)
print("  -> zero-ticks EXTEND the episode; a strict price-change rule would")
print("     have reported three separate runs of 1")

# ==========================================================================
print("\n12. Tick evidence rescues a sparse absorption candle")
# ==========================================================================
m3 = PhaseMachine()
for _ in range(25):  # seed percentile history
    m3.update(mk([Trade(1.0, 0.049, 50, True), Trade(30.0, 0.049, 50, False)]))
sparse_absorb = mk(
    [
        Trade(1.0, 0.04901, 900, False, TD.MINUS),
        Trade(20.0, 0.04901, 900, False, TD.ZERO_MINUS),
        Trade(40.0, 0.04901, 900, False, TD.ZERO_MINUS),
    ],
    book=[BookSample(t, 900_000, 200_000) for t in (5, 25, 45)],
)
r3 = m3.update(sparse_absorb)
check(f"3 trades but classified {r3.phase.value}", r3.phase == Phase.ABSORPTION)
check("confident despite being under min_trades, on tick evidence", r3.confident)
for rr in r3.reasons:
    print(f"    - {rr}")

# ==========================================================================
print("\n13. Big bid + sellers walking price down = NOT absorption")
# ==========================================================================
m4 = PhaseMachine()
m4.phase = Phase.ABSORPTION
fake_wall = mk(
    [
        Trade(1.0, 0.04900, 400, False, TD.MINUS),
        Trade(15.0, 0.04899, 400, False, TD.MINUS),
        Trade(30.0, 0.04898, 400, False, TD.MINUS),
        Trade(45.0, 0.04897, 400, False, TD.MINUS),
    ],
    # bbo LOOKS supportive -- large bid, but parked away from the touch
    book=[BookSample(t, 950_000, 150_000) for t in (5, 25, 45)],
)
r4 = m4.update(fake_wall)
check(f"bbo={fake_wall.bbo_avg:.2f} bid-heavy but every sell ticked price down",
      r4.phase == Phase.INVALIDATED)
for rr in r4.reasons:
    if "moved price down" in rr:
        print(f"    - {rr}")
print(f"    absorption_tick_ratio={fake_wall.absorption_tick_ratio:.2f} "
      f"sell_efficiency={fake_wall.sell_efficiency:.0%}")
print("  -> the bbo ratio alone would have called this support")

# ==========================================================================
print("\n14. Inferred bid depth from fills alone (no orderbook)")
# ==========================================================================
# A 900-unit sell is absorbed; a 1500-unit sell breaks the level.
# => 900 < true bid depth <= 1500, proven by execution, unspoofable.
# Coherent case: 900 absorbed, 1500 broke the level -> a real interval.
probe = mk(
    [
        Trade(1.0, 0.04901, 900, False, TD.ZERO_MINUS),
        Trade(20.0, 0.04900, 1500, False, TD.MINUS),
    ],
    book=[BookSample(t, 1000, 400) for t in (5, 15, 25)],
)
lo, hi = probe.inferred_bid_depth
check(f"lower bound from absorbed fill: >{lo:.0f}", lo == 900)
check(f"upper bound from broken level: <={hi:.0f}", hi == 1500)
check("bounds are coherent, so not flagged unstable", probe.depth_unstable is False)
check(f"wall_tested = {probe.wall_tested:.0%} of displayed bid", probe.wall_tested == 0.9)

# Inverted case, taken verbatim from the real POPCATUSDT run: a 478-unit sell
# was absorbed while a SMALLER 153-unit sell broke the level. On live data this
# happened in 89% of candles that reported bounds -- so it is the normal case,
# not an anomaly, and reporting it as ">478 and <=153" was meaningless.
flick = mk(
    [
        Trade(1.0, 0.05737, 478, False, TD.ZERO_MINUS),
        Trade(20.0, 0.05736, 153, False, TD.MINUS),
    ],
    book=[BookSample(t, 1000, 400) for t in (5, 15, 25)],
)
lo2, hi2 = flick.inferred_bid_depth
check(f"hard lower bound survives (>{lo2:.0f})", lo2 == 478)
check("incoherent upper bound suppressed rather than printed", hi2 is None)
check("flagged depth_unstable instead", flick.depth_unstable is True)
print("  -> depth was FLICKERING within the minute: present for one order,")
print("     gone for a smaller one. No single interval describes it.")

# ==========================================================================
print("\n15. Hidden liquidity: nothing displayed, price still refuses to move")
# ==========================================================================
iceberg = mk(
    [
        Trade(1.0, 0.04901, 800, False, TD.MINUS),
        Trade(15.0, 0.04901, 800, False, TD.ZERO_MINUS),
        Trade(30.0, 0.04901, 800, False, TD.ZERO_MINUS),
        Trade(45.0, 0.04901, 800, False, TD.ZERO_MINUS),
    ],
    # displayed bid SMALLER than ask -- bbo would read this as bearish
    book=[BookSample(t, 120_000, 400_000) for t in (5, 25, 45)],
)
check(f"bbo={iceberg.bbo_avg:.2f} is ask-heavy, would fail the bbo gate",
      iceberg.bbo_avg < 0.45)
check("hidden_liquidity detected anyway", iceberg.hidden_liquidity is True)
m5 = PhaseMachine()
r5 = m5.update(iceberg)
check(f"classified {r5.phase.value} via the executed-wall path",
      r5.phase == Phase.ABSORPTION)
for rr in r5.reasons:
    print(f"    - {rr}")

# ==========================================================================
print("\n16. Episode accumulation against the real logged data")
# ==========================================================================
# Closes from the reference absorption window, 01:15:59 -> 01:19:59.
REAL = [0.04905, 0.04902, 0.04898, 0.04897, 0.04894]
m6 = PhaseMachine()
# Seed a volatility baseline of roughly 0.04% per candle.
for i in range(30):
    o = 0.0490
    cl = o * (1 + (0.0004 if i % 2 else -0.0004))
    m6.update(mk([Trade(1.0, o, 40, True), Trade(30.0, cl, 40, False)]))
base = m6.vol_baseline_pct(4)
check(f"vol baseline over 4 candles = {base:.3f}% (engages now)", base > 0)

m6.phase = Phase.NEUTRAL
last = None
for i, px in enumerate(REAL[1:], start=1):
    prev_px = REAL[i - 1]
    c = mk(
        [
            Trade(1.0, prev_px, 3000, False, TD.MINUS),
            Trade(20.0, px, 3000, False, TD.ZERO_MINUS),
            Trade(40.0, px, 3000, False, TD.ZERO_MINUS),
            Trade(55.0, px, 600, True, TD.ZERO_MINUS),
        ],
        book=[BookSample(t, 700_000, 300_000) for t in (5, 25, 45)],
        prev=prev_px,
    )
    last = m6.update(c)

ep = last.episode
check(f"episode spans {ep.candles} candles", ep.candles == 4)
pe = ep.price_effect_pct
check(f"price_effect_pct = {pe:.2f}% matches your -0.22%", abs(pe - (-0.2243)) < 0.01)
z = ep.price_effect_z(base)
print(f"  net price effect      : {pe:+.4f}%")
print(f"  vol-normalised (z)    : {z:+.2f}  <- the correction that makes it readable")
print(f"  net supply absorbed   : {abs(ep.cum_delta_min):,.0f} units")
print(f"  absorbed without moving price: {ep.absorbed_sell_vol:,.0f} units")
print(f"  episode tick ratio    : {ep.absorption_tick_ratio:.0%} pinned")
print(f"  supply per unit damage: {ep.supply_per_pct(base):,.0f}")

# payoff asymmetry actually realised
low, exit_px = 0.04894, 0.05064
risk = abs(pe)
reward = (exit_px - low) / low * 100
check(f"realised payoff {reward:.2f}% vs {risk:.2f}% excursion = {reward/risk:.1f}x",
      reward / risk > 10)

# ==========================================================================
print("\n17. Volatility baseline survives candles that close where they opened")
# ==========================================================================
# On a thin sub-penny instrument many minutes close exactly at their open. A
# median of |close - open| collapses to zero and silently disables the whole
# normalisation -- so the baseline uses the intrabar RANGE and is floored at
# one tick.
m7 = PhaseMachine()
for _ in range(30):
    # closes exactly at open, but the bar has real intrabar range
    m7.update(
        mk([
            Trade(1.0, 0.04900, 50, True, TD.PLUS),
            Trade(20.0, 0.04903, 50, True, TD.PLUS),
            Trade(40.0, 0.04900, 50, False, TD.MINUS),
        ])
    )
base7 = m7.vol_baseline_pct(4)
check(f"baseline = {base7:.4f}% rather than collapsing to 0", base7 > 0)

# Now the pathological case: zero range on every candle. The floor must hold.
m8 = PhaseMachine()
for _ in range(30):
    m8.update(mk([Trade(1.0, 0.04900, 50, True), Trade(30.0, 0.04900, 50, False)]))
base8 = m8.vol_baseline_pct(1)
tick_pct = TICK / 0.04900 * 100
check(f"zero-range candles -> floored at 1 tick ({base8:.4f}% vs {tick_pct:.4f}%)",
      abs(base8 - tick_pct) < 1e-9)
check("so price_effect_z is still defined",
      m8.vol_baseline_pct(4) > 0)
print("  -> below the price grid there is nothing to normalise against")

# ==========================================================================
print(f"\n{'=' * 62}\n{ok} passed, {fail} failed\n{'=' * 62}")
raise SystemExit(1 if fail else 0)
