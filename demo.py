"""
Minimal runnable example. From the repo root:

    python3 demo.py

If you instead copied only features.py and phases.py into a bare folder, put
this file NEXT TO that folder, never inside it:

    your_project/
        demo.py             <- here
        microstructure/
            features.py
            phases.py

phases.py uses a relative import, so nothing in the package can be executed
from inside the package directory.
"""

from microstructure.features import BookSample, TickDirection as TD, Trade, build_candle
from microstructure.phases import PhaseMachine

TICK = 0.00001

# Closes from the reference absorption window (4 red HA candles, -0.22%).
CLOSES = [0.04905, 0.04902, 0.04898, 0.04897, 0.04894]

machine = PhaseMachine()

# Seed a volatility baseline. The percentile gates stay silent below 20 candles,
# so without this the episode line has nothing to normalise against.
for i in range(30):
    o = 0.0490
    cl = o * (1 + (0.0004 if i % 2 else -0.0004))
    machine.update(
        build_candle(0, 60, TICK, [Trade(1.0, o, 40, True), Trade(30.0, cl, 40, False)])
    )

print(f"volatility baseline over 4 candles: {machine.vol_baseline_pct(4):.3f}%\n")

for i, px in enumerate(CLOSES[1:], start=1):
    prev = CLOSES[i - 1]

    candle = build_candle(
        ts_open=0,
        ts_close=60,
        tick_size=TICK,
        trades=[
            # Heavy selling: the first print moves price, the rest are absorbed
            # at the same level -- ZeroMinusTick is the bid refilling.
            Trade(1.0, prev, 3000, False, TD.MINUS),
            Trade(20.0, px, 3000, False, TD.ZERO_MINUS),
            Trade(40.0, px, 3000, False, TD.ZERO_MINUS),
            Trade(55.0, px, 600, True, TD.ZERO_MINUS),
        ],
        book=[BookSample(t, 700_000, 300_000) for t in (5, 25, 45)],
        prev_close=prev,
    )

    result = machine.update(candle)

    print(f"candle {i}  close={px:.5f}  ->  {result.phase.value}"
          f"{'' if result.confident else '  (low confidence)'}")
    print(f"    agg={candle.agg:.3f} (n={candle.agg_n:,.0f})   "
          f"bbo={candle.bbo_avg:.3f}   "
          f"absorption_tick_ratio={candle.absorption_tick_ratio:.0%}")
    for reason in result.reasons:
        print(f"    - {reason}")
    print()

print("NOTE: all numbers above come from synthetic trades. The volatility")
print("baseline in particular is invented, so the vol-normalised figure is")
print("illustrative only -- it needs your real tape to mean anything.")
