#!/usr/bin/env python3
"""
Replay every raw tape through the CURRENT code and emit one row per candle.

The console logs from earlier runs cannot be recomputed, but the tapes can, so
this regenerates every feature with:

  - the fill-fragmentation fix (PR #7)
  - time-ordered trade paths (PR #7)
  - the CORRECT tick size per instrument (TRX runs were collected at 0.00001
    when the real tick is 0.0001)

    python3 replay_tapes.py                 # write candles.csv
    python3 replay_tapes.py --frag-compare  # quantify the PR #7 fix
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

from microstructure import features as F
from microstructure.bybit import BybitAssembler
from microstructure.phases import PhaseMachine

# Verified from the tapes: smallest observed price increment per symbol.
TICK = {"POPCATUSDT": 0.00001, "TRXUSDT": 0.0001}

FIELDS = [
    "tape", "symbol", "minute_ms", "phase", "confident", "usable",
    "open", "high", "low", "close", "vwap",
    "n_trades", "volume", "buy_vol", "sell_vol", "delta",
    "agg", "agg_raw", "agg_n",
    "delta_min", "delta_max", "delta_recovery", "delta_giveback", "flow_lag",
    "plus", "zero_plus", "minus", "zero_minus",
    "absorption_tick_ratio", "sell_efficiency", "buy_efficiency", "pinned_share",
    "impact_up", "impact_down", "impact_asymmetry", "absorption_per_tick",
    "bbo_avg", "bbo_samples", "bid_sz_avg", "ask_sz_avg",
    "absorbed_max_sell", "depth_unstable", "wall_tested",
    "sell_decay", "buy_growth", "dup_books",
]


def symbol_of(name: str) -> str:
    return "TRXUSDT" if "TRX" in name.upper() else "POPCATUSDT"


def replay(path: Path, tick: float):
    """Yield (Emitted, Classified) for one tape."""
    asm = BybitAssembler(tick_size=tick)
    machine = PhaseMachine()
    out = []

    def handle(em):
        res = machine.update(em.candle) if em.usable else None
        out.append((em, res))

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            for em in asm.on_message(msg):
                handle(em)
    for em in asm.flush():
        handle(em)
    return out, asm


def row_of(tape, symbol, em, res):
    c = em.candle
    g = lambda v: "" if v is None else v
    return {
        "tape": tape, "symbol": symbol, "minute_ms": em.minute_start_ms,
        "phase": res.phase.value if res else "UNUSABLE",
        "confident": int(res.confident) if res else 0,
        "usable": int(em.usable),
        "open": c.open, "high": c.high, "low": c.low, "close": c.close,
        "vwap": g(c.vwap),
        "n_trades": c.n_trades, "volume": c.volume,
        "buy_vol": c.buy_vol, "sell_vol": c.sell_vol, "delta": c.delta,
        "agg": g(c.agg), "agg_raw": g(c.agg_raw), "agg_n": c.agg_n,
        "delta_min": c.delta_min, "delta_max": c.delta_max,
        "delta_recovery": c.delta_recovery, "delta_giveback": c.delta_giveback,
        "flow_lag": g(c.flow_lag),
        "plus": c.plus_ticks, "zero_plus": c.zero_plus_ticks,
        "minus": c.minus_ticks, "zero_minus": c.zero_minus_ticks,
        "absorption_tick_ratio": g(c.absorption_tick_ratio),
        "sell_efficiency": g(c.sell_efficiency),
        "buy_efficiency": g(c.buy_efficiency),
        "pinned_share": g(c.pinned_share),
        "impact_up": g(c.impact_up), "impact_down": g(c.impact_down),
        "impact_asymmetry": g(c.impact_asymmetry),
        "absorption_per_tick": c.absorption_per_tick,
        "bbo_avg": g(c.bbo_avg), "bbo_samples": c.bbo_samples,
        "bid_sz_avg": g(c.bid_sz_avg), "ask_sz_avg": g(c.ask_sz_avg),
        "absorbed_max_sell": c.absorbed_max_sell,
        "depth_unstable": int(c.depth_unstable),
        "wall_tested": g(c.wall_tested),
        "sell_decay": g(c.sell_decay), "buy_growth": g(c.buy_growth),
        "dup_books": em.dropped_duplicate_books,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="candles.csv")
    ap.add_argument("--frag-compare", action="store_true",
                    help="replay twice, with and without the fill-merge fix")
    args = ap.parse_args()

    tapes = sorted(Path(p) for p in glob.glob("*.jsonl"))
    if not tapes:
        sys.exit("no *.jsonl tapes found")

    if args.frag_compare:
        original = F._merge_fills
        results = {}
        for label, fn in (("with fix", original), ("WITHOUT fix", lambda t: list(t))):
            F._merge_fills = fn
            pinned, trades, orders = [], 0, 0
            per_symbol = {}
            for t in tapes:
                sym = symbol_of(t.name)
                rows, _ = replay(t, TICK[sym])
                for em, res in rows:
                    c = em.candle
                    trades += c.n_trades
                    if c.absorption_tick_ratio is not None:
                        pinned.append(c.absorption_tick_ratio)
                        per_symbol.setdefault(sym, []).append(c.absorption_tick_ratio)
            results[label] = (pinned, trades, per_symbol)
        F._merge_fills = original

        print("=== impact of the fill-fragmentation fix (PR #7) ===\n")
        print(f"  {'':<13} {'prints/orders':>14} {'median pinned':>15} "
              f"{'mean pinned':>13} {'n candles':>10}")
        for label in ("WITHOUT fix", "with fix"):
            pinned, trades, _ = results[label]
            print(f"  {label:<13} {trades:>14,} {st.median(pinned):>14.1%} "
                  f"{st.mean(pinned):>12.1%} {len(pinned):>10,}")
        print()
        for sym in sorted(results["with fix"][2]):
            a = results["WITHOUT fix"][2].get(sym, [])
            b = results["with fix"][2].get(sym, [])
            if a and b:
                print(f"  {sym:<12} median pinned {st.median(a):>6.1%} "
                      f"-> {st.median(b):>6.1%}")
        wo = results["WITHOUT fix"][1]
        wf = results["with fix"][1]
        print(f"\n  {wo:,} raw prints collapsed to {wf:,} logical orders "
              f"({(1 - wf / wo):.1%} were fragments of a larger order)")
        return

    all_rows = []
    print(f"{'tape':<36} {'candles':>8} {'usable':>7} {'trades':>7} {'warn':>5}")
    for t in tapes:
        sym = symbol_of(t.name)
        rows, asm = replay(t, TICK[sym])
        usable = sum(1 for em, _ in rows if em.usable)
        ntr = sum(em.candle.n_trades for em, _ in rows)
        warn = "yes" if asm.tick_size_warning() else "-"
        print(f"{t.name:<36} {len(rows):>8} {usable:>7} {ntr:>7,} {warn:>5}")
        all_rows.extend(row_of(t.name, sym, em, res) for em, res in rows)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    usable = sum(r["usable"] for r in all_rows)
    print(f"\nwrote {len(all_rows):,} candles ({usable:,} usable) -> {args.out}")


if __name__ == "__main__":
    main()
