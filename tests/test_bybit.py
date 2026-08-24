"""
Message-parsing tests. Payloads are the response examples from the Bybit v5
docs, so field names are verified against the source rather than assumed.

Run:  python3 tests/test_bybit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microstructure.bybit import BybitAssembler  # noqa: E402

TICK = 0.00001
ok, fail = 0, 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


def trade_msg(ts_ms, side, size, price, tick_dir=None, block=False, rpi=False):
    row = {
        "T": ts_ms,
        "s": "XYZUSDT",
        "S": side,
        "v": str(size),
        "p": str(price),
        "i": f"id-{ts_ms}",
        "BT": block,
        "seq": ts_ms,
    }
    if tick_dir:
        row["L"] = tick_dir
    if rpi:
        row["RPI"] = True
    return {"topic": "publicTrade.XYZUSDT", "type": "snapshot", "ts": ts_ms, "data": [row]}


def feed(assembler, messages):
    """Candles are emitted by on_message as boundaries are crossed, so they
    must be collected there -- flush() only closes the LAST candle."""
    out = []
    for m in messages:
        out.extend(assembler.on_message(m))
    out.extend(assembler.flush())
    return out


def book_msg(ts_ms, bid_sz, ask_sz, u):
    return {
        "topic": "orderbook.1.XYZUSDT",
        "type": "snapshot",
        "ts": ts_ms,
        "data": {
            "s": "XYZUSDT",
            "b": [["0.04901", str(bid_sz)]],
            "a": [["0.04902", str(ask_sz)]],
            "u": u,
            "seq": ts_ms,
        },
        "cts": ts_ms - 2,
    }


M0 = 1672304400000  # a clean minute boundary
M1 = M0 + 60_000
M2 = M1 + 60_000


# ==========================================================================
print("\n1. The doc's own publicTrade example parses")
# ==========================================================================
# Verbatim from https://bybit-exchange.github.io/docs/v5/websocket/public/trade
doc_example = {
    "topic": "publicTrade.BTCUSDT",
    "type": "snapshot",
    "ts": 1672304486868,
    "data": [
        {
            "T": 1672304486865,
            "s": "BTCUSDT",
            "S": "Buy",
            "v": "0.001",
            "p": "16578.50",
            "L": "PlusTick",
            "i": "20f43950-d8dd-5b31-9112-a178eb6023af",
            "BT": False,
            "seq": 1783284617,
        }
    ],
}
a = BybitAssembler(tick_size=0.5)
a.on_message(doc_example)
[em] = a.flush()
c = em.candle
check(f"taker Buy -> aggressive buy (buy_vol={c.buy_vol})", c.buy_vol == 0.001)
check(f"price parsed from string ({c.close})", c.close == 16578.50)
check(f"L='PlusTick' -> plus_ticks={c.plus_ticks}", c.plus_ticks == 1)

# ==========================================================================
print("\n2. Block trades are filtered (they never touched the book)")
# ==========================================================================
a = BybitAssembler(tick_size=TICK)
a.on_message(trade_msg(M0 + 1000, "Sell", 500, 0.04901, "MinusTick"))
# A 5,000,000-unit block at an unchanged price: enormous size, zero impact.
# Left in, this is a picture-perfect false absorption signal.
a.on_message(trade_msg(M0 + 2000, "Sell", 5_000_000, 0.04901, "ZeroMinusTick", block=True))
[em] = a.flush()
check(f"block excluded from volume (sell_vol={em.candle.sell_vol:,.0f})",
      em.candle.sell_vol == 500)
check(f"counted as dropped ({em.dropped_block_trades})", em.dropped_block_trades == 1)
check("block did NOT inflate absorbed volume",
      em.candle.absorbed_sell_vol == 0)

kept = BybitAssembler(tick_size=TICK, drop_block_trades=False)
kept.on_message(trade_msg(M0 + 1000, "Sell", 500, 0.04901, "MinusTick"))
kept.on_message(trade_msg(M0 + 2000, "Sell", 5_000_000, 0.04901, "ZeroMinusTick", block=True))
[em2] = kept.flush()
print(f"  if NOT filtered: absorbed_sell_vol jumps to {em2.candle.absorbed_sell_vol:,.0f}")
print(f"  and absorption_tick_ratio reads {em2.candle.absorption_tick_ratio:.0%}"
      " -- entirely fabricated")

# ==========================================================================
print("\n3. The 3-second republish is deduplicated on u")
# ==========================================================================
a = BybitAssembler(tick_size=TICK)
a.on_message(book_msg(M0 + 1000, 900_000, 100_000, u=500))
a.on_message(book_msg(M0 + 4000, 900_000, 100_000, u=500))  # unchanged republish
a.on_message(book_msg(M0 + 7000, 900_000, 100_000, u=500))  # and again
a.on_message(book_msg(M0 + 9000, 100_000, 900_000, u=501))  # real change
a.on_message(trade_msg(M0 + 10_000, "Sell", 10, 0.04901, "MinusTick"))
[em] = a.flush()
check(f"4 messages -> {em.candle.bbo_samples} samples", em.candle.bbo_samples == 2)
check(f"2 duplicates dropped ({em.dropped_duplicate_books})",
      em.dropped_duplicate_books == 2)
check(f"bbo_avg = {em.candle.bbo_avg:.3f} (unbiased mean of the 2 real states)",
      abs(em.candle.bbo_avg - 0.5) < 1e-9)
print("  without dedup the stale 0.90 state would have carried 3x the weight,")
print(f"  pulling bbo_avg to {(0.9*3 + 0.1)/4:.3f}")

# ==========================================================================
print("\n4. Spot has no L field -> derived locally with carry-forward")
# ==========================================================================
a = BybitAssembler(tick_size=TICK)
a.on_message(trade_msg(M0 + 1000, "Sell", 100, 0.04902))  # no L
a.on_message(trade_msg(M0 + 2000, "Sell", 100, 0.04901))  # lower -> MinusTick
a.on_message(trade_msg(M0 + 3000, "Sell", 100, 0.04901))  # equal -> ZeroMinusTick
a.on_message(trade_msg(M0 + 4000, "Sell", 100, 0.04901))  # equal -> ZeroMinusTick
[em] = a.flush()
c = em.candle
check(f"derived {c.minus_ticks} MinusTick", c.minus_ticks == 1)
check(f"derived {c.zero_minus_ticks} ZeroMinusTick", c.zero_minus_ticks == 2)
check(f"absorption_tick_ratio = {c.absorption_tick_ratio:.0%} without any L field",
      abs(c.absorption_tick_ratio - 2 / 3) < 1e-9)

# ==========================================================================
print("\n5. Quiet minute vs feed gap -- opposite meanings, same zero trades")
# ==========================================================================
msgs = []
# minute 0: normal
for i in range(8):
    msgs.append(book_msg(M0 + i * 3000, 500_000, 500_000, u=1000 + i))
msgs.append(trade_msg(M0 + 1000, "Sell", 100, 0.04901, "MinusTick"))
# minute 1: book still arriving, nobody trades -> genuinely QUIET
for i in range(8):
    msgs.append(book_msg(M1 + i * 3000, 800_000, 200_000, u=2000 + i))
# minute 2 message forces minute 1 to close
msgs.append(trade_msg(M2 + 1000, "Sell", 100, 0.04901, "MinusTick"))
ems = feed(BybitAssembler(tick_size=TICK), msgs)

quiet = next(e for e in ems if e.minute_start_ms == M1)
check(f"quiet minute has 0 trades ({quiet.candle.n_trades})", quiet.candle.n_trades == 0)
check(f"but {quiet.candle.bbo_samples} book samples -> feed_healthy",
      quiet.feed_healthy is True)
check("so it is USABLE: real information, maximal seller exhaustion",
      quiet.usable is True)
# Tolerance, not equality: 0.8 is not exactly representable, so summing seven
# copies and dividing lands a few ulps away.
check(f"bbo still valid ({quiet.candle.bbo_avg:.4f})",
      abs(quiet.candle.bbo_avg - 0.8) < 1e-9)
# 8 book messages were sent for this minute but 7 landed in it. That is
# CORRECT: attribution follows cts (matching-engine time), not ts (system
# generation time), and the docs say cts is the field that correlates with
# trade T. Near a boundary the two can disagree by a few ms.
check("boundary attribution follows cts, not ts", quiet.candle.bbo_samples == 7)

# now a true disconnect: no messages at all for two minutes
msgs = [book_msg(M0 + i * 3000, 500_000, 500_000, u=3000 + i) for i in range(8)]
msgs.append(trade_msg(M0 + 1000, "Sell", 100, 0.04901, "MinusTick"))
msgs.append(trade_msg(M0 + 3 * 60_000 + 1000, "Sell", 100, 0.04901, "MinusTick"))
ems = feed(BybitAssembler(tick_size=TICK), msgs)
gaps = [e for e in ems if e.synthetic]
check(f"2 skipped minutes reconstructed ({len(gaps)})", len(gaps) == 2)
check("flagged unhealthy", all(not g.feed_healthy for g in gaps))
check("and NOT usable -- a hole, not a quiet market",
      all(not g.usable for g in gaps))
print("  -> this is what swallowed 12 minutes and +1.5% of your logged move")

# ==========================================================================
print("\n6. Candles close on minute rollover, prev state carries forward")
# ==========================================================================
a = BybitAssembler(tick_size=TICK)
a.on_message(trade_msg(M0 + 30_000, "Sell", 100, 0.04902, "MinusTick"))
emitted = a.on_message(trade_msg(M1 + 1000, "Sell", 100, 0.04902))
check(f"crossing the boundary emits exactly 1 candle ({len(emitted)})",
      len(emitted) == 1)
check(f"closed candle is minute M0 ({emitted[0].minute_start_ms == M0})",
      emitted[0].minute_start_ms == M0)
[nxt] = a.flush()
check("unchanged price in the NEW candle inherits prior direction -> ZeroMinusTick",
      nxt.candle.zero_minus_ticks == 1)
check("not left undetermined", nxt.candle.undetermined_ticks == 0)

print(f"\n{'=' * 62}\n{ok} passed, {fail} failed\n{'=' * 62}")
raise SystemExit(1 if fail else 0)
