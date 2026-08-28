#!/usr/bin/env python3
"""
Is the HA entry late, and was absorption detectable at the low?

Walks each run log sequentially, tracking candle index, so entries and exits are
located exactly in the price series. For every trade it measures:

  entry_position  how much of the low->peak move was already gone at entry
  captured        how much of that move the trade actually kept
  bbo/agg at low  what the metrics looked like at the swing low, which is the
                  only place an "early absorption" entry could have happened

The last one is the test that matters: if bbo was already elevated at the low,
an earlier entry is reachable with data the tracker already computes. If it was
not, no amount of tuning finds the low.
"""

import glob
import re
import statistics as st
from pathlib import Path

from analyze_run import strip_rtf

CANDLE = re.compile(
    r"\[([\d\-: ,]+)\] close=([\d.]+) \| HA\(O/C\)=([\d.]+)/([\d.]+) \((\w+)\)"
    r" \| agg=([\d.]+) \| bbo=([\d.]+)"
)
ENTER = re.compile(r"Entering 'Ride the Wave' mode")
EXIT = re.compile(r"EXIT LONG at [\d\- :,]+\s*\|\s*exit_price=([\d.]+)\s*\|\s*"
                  r"entry_price=([\d.]+)")

LOOKBACK = 20   # candles to search back for the swing low
FORWARD = 30    # candles after entry to search for the peak


def parse(path: Path):
    txt = strip_rtf(path.read_text(errors="replace"))
    candles, events = [], []
    for line in txt.splitlines():
        m = CANDLE.search(line)
        if m:
            candles.append({
                "close": float(m.group(2)),
                "ha": m.group(5),
                "agg": float(m.group(6)),
                "bbo": float(m.group(7)),
            })
            continue
        if ENTER.search(line):
            events.append(("ENTER", len(candles) - 1, None))
            continue
        e = EXIT.search(line)
        if e:
            events.append(("EXIT", len(candles) - 1, float(e.group(1))))
    return candles, events


def main():
    rows = []
    for f in sorted(Path(p) for p in glob.glob("frac34*.rtf")):
        candles, events = parse(f)
        if len(candles) < 50:
            continue
        pending = None
        for kind, idx, px in events:
            if kind == "ENTER":
                pending = idx
            elif kind == "EXIT" and pending is not None:
                i0, i1 = pending, idx
                pending = None
                if i0 < 1 or i1 <= i0 or i1 >= len(candles):
                    continue
                entry = candles[i0]["close"]
                exit_px = px if px else candles[i1]["close"]

                back = candles[max(0, i0 - LOOKBACK):i0 + 1]
                low_i = min(range(len(back)), key=lambda k: back[k]["close"])
                low = back[low_i]["close"]
                low_abs = max(0, i0 - LOOKBACK) + low_i

                fwd = candles[i0:min(len(candles), i1 + FORWARD)]
                peak = max(c["close"] for c in fwd)

                move = peak - low
                if move <= 0 or low <= 0:
                    continue
                rows.append({
                    "run": f.name,
                    "entry_pos": (entry - low) / move,
                    "captured": (exit_px - entry) / move,
                    "move_pct": move / low * 100,
                    "ret_pct": (exit_px - entry) / entry * 100,
                    "bars_from_low": i0 - low_abs,
                    "bbo_at_low": candles[low_abs]["bbo"],
                    "agg_at_low": candles[low_abs]["agg"],
                    "bbo_at_entry": candles[i0]["bbo"],
                    "agg_at_entry": candles[i0]["agg"],
                    "bbo_series": [c["bbo"] for c in candles],
                })

    if not rows:
        print("no trades located")
        return

    n = len(rows)
    print(f"located {n} trades in the candle series\n")

    print("=== how late is the entry? ===")
    ep = [r["entry_pos"] for r in rows]
    print(f"  fraction of the low->peak move already gone at entry")
    print(f"    median {st.median(ep):.1%}   mean {st.mean(ep):.1%}")
    q = sorted(ep)
    print(f"    p25 {q[n//4]:.1%}   p75 {q[3*n//4]:.1%}")
    print(f"  entries above 50% of the move: "
          f"{sum(1 for x in ep if x > 0.5)/n:.0%}")
    print(f"  entries above 80% of the move: "
          f"{sum(1 for x in ep if x > 0.8)/n:.0%}")
    bl = [r["bars_from_low"] for r in rows]
    print(f"  candles between the low and the entry: median {st.median(bl):.0f}, "
          f"mean {st.mean(bl):.1f}")

    print("\n=== how much of the move did the trade keep? ===")
    cap = [r["captured"] for r in rows]
    print(f"  median {st.median(cap):+.1%}   mean {st.mean(cap):+.1%}")
    print(f"  median available move size: "
          f"{st.median([r['move_pct'] for r in rows]):.2f}%")

    print("\n=== was absorption visible AT THE LOW? ===")
    print("  (the only place an early entry could have happened)")
    for label, key in (("bbo", "bbo_at_low"), ("agg", "agg_at_low")):
        v = [r[key] for r in rows]
        s = sorted(v)
        print(f"  {label} at low: median {st.median(v):.3f}  "
              f"p25 {s[n//4]:.3f}  p75 {s[3*n//4]:.3f}")
    for label, k_low, k_ent in (("bbo", "bbo_at_low", "bbo_at_entry"),
                                ("agg", "agg_at_low", "agg_at_entry")):
        lo = st.median([r[k_low] for r in rows])
        en = st.median([r[k_ent] for r in rows])
        print(f"  {label}: {lo:.3f} at the low -> {en:.3f} at entry "
              f"({en - lo:+.3f})")

    # Would a bbo percentile gate have fired at the low?
    print("\n=== could a bbo gate have caught the low? ===")
    allb = [b for r in rows for b in r["bbo_series"][:1]]  # placeholder guard
    pool = sorted({b for r in rows for b in r["bbo_series"]})
    def pct(v):
        lo, hi = 0, len(pool)
        while lo < hi:
            mid = (lo + hi) // 2
            if pool[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(pool)
    rank_low = [pct(r["bbo_at_low"]) for r in rows]
    rank_ent = [pct(r["bbo_at_entry"]) for r in rows]
    print(f"  bbo percentile at the low:   median {st.median(rank_low):.0%}")
    print(f"  bbo percentile at the entry: median {st.median(rank_ent):.0%}")
    for thr in (0.6, 0.7, 0.8):
        hit = sum(1 for x in rank_low if x >= thr) / n
        print(f"  lows where bbo was already in the top "
              f"{1-thr:.0%}: {hit:.0%}")


if __name__ == "__main__":
    main()
