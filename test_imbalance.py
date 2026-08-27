#!/usr/bin/env python3
"""
Measure the imbalance-bar / time-bar pairing on the real tapes.

Reports what the event clock actually looks like at these liquidity levels, and
whether pairing a flow trigger with the one validated time-bar feature (bbo_avg)
produces anything above transaction costs.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from microstructure.bybit import BybitAssembler
from microstructure.features import Trade
from microstructure.imbalance import ImbalanceBarrier, RollingFlow, evaluate

TICK = {"POPCATUSDT": 0.00001, "TRXUSDT": 0.0001}
FEE = 0.20
MS_MIN = 60_000


def symbol_of(name: str) -> str:
    return "TRXUSDT" if "TRX" in name.upper() else "POPCATUSDT"


def load_tape(path: Path):
    """Return (trades sorted by ts, {minute_ms: (close, bbo_avg)})."""
    sym = symbol_of(path.name)
    asm = BybitAssembler(tick_size=TICK[sym])
    minutes = {}
    trades: list[Trade] = []

    def take(em):
        c = em.candle
        if em.usable and c.n_trades > 0:
            minutes[em.minute_start_ms] = (c.close, c.bbo_avg)
        trades.extend(c._merged if hasattr(c, "_merged") else [])

    raw = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("topic", "").startswith("publicTrade"):
                for r in msg.get("data") or []:
                    if r.get("BT"):
                        continue
                    is_buy = {"Buy": True, "Sell": False}.get(r["S"])
                    if is_buy is None:
                        continue
                    raw.append(Trade(ts=int(r["T"]) / 1000.0, price=float(r["p"]),
                                     size=float(r["v"]), is_buy=is_buy))
            for em in asm.on_message(msg):
                take(em)
    for em in asm.flush():
        take(em)

    raw.sort(key=lambda t: t.ts)
    return raw, minutes, sym


def bbo_ranker(minutes_by_symbol):
    """Percentile rank of a bbo value within its instrument's history."""
    sorted_vals = {}
    for sym, vals in minutes_by_symbol.items():
        sorted_vals[sym] = sorted(v for v in vals if v is not None)

    def rank(sym, v):
        arr = sorted_vals.get(sym)
        if not arr or v is None or len(arr) < 50:
            return None
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(arr)

    return rank


def main():
    tapes = sorted(Path(p) for p in glob.glob("*.jsonl"))
    data = {}
    bbo_pool = defaultdict(list)
    for t in tapes:
        raw, minutes, sym = load_tape(t)
        if len(raw) < 50:
            continue
        data[t.name] = (raw, minutes, sym)
        bbo_pool[sym].extend(b for _, b in minutes.values())
    rank = bbo_ranker(bbo_pool)

    print(f"loaded {len(data)} tapes\n")

    # ---- what does the event clock look like? ----
    print("=== imbalance bars: what the event clock costs you in time ===")
    print(f"  {'symbol':<12} {'mode':<7} {'theta':>8} {'bars':>6} "
          f"{'med span':>10} {'med trades':>11} {'buy/sell':>10}")
    chosen = {}
    for sym in sorted({s for _, _, s in data.values()}):
        rates = []
        for name, (raw, _, s) in data.items():
            if s != sym:
                continue
            span = (raw[-1].ts - raw[0].ts) / 60
            if span > 30:
                rates.append(len(raw) / span)
        med_rate = st.median(rates) if rates else 0
        for mode, use_vol in (("count", False), ("volume", True)):
            for mult in (10, 25, 50):
                if use_vol:
                    vols = [t.size for name, (raw, _, s) in data.items()
                            if s == sym for t in raw]
                    theta = st.median(vols) * mult
                else:
                    theta = float(mult)
                bars, spans, ntr, sides = [], [], [], []
                for name, (raw, _, s) in data.items():
                    if s != sym:
                        continue
                    b = ImbalanceBarrier(threshold=theta, use_volume=use_vol)
                    for tr in raw:
                        bar = b.add(tr)
                        if bar:
                            bars.append((name, bar))
                            spans.append(bar.span_s)
                            ntr.append(bar.n_trades)
                            sides.append(bar.side)
                if len(bars) < 20:
                    continue
                nb = sum(1 for x in sides if x == "BUY")
                print(f"  {sym:<12} {mode:<7} {theta:>8.1f} {len(bars):>6} "
                      f"{st.median(spans)/60:>8.1f}m {st.median(ntr):>11.0f} "
                      f"{nb}/{len(sides)-nb:>9}")
                if mode == "count" and mult == 25:
                    chosen[sym] = (theta, use_vol, bars)
        print(f"  {sym:<12} (median trade rate {med_rate:.1f}/min)")

    # ---- does the pairing produce anything? ----
    print("\n=== SELL-side imbalance + bid-heavy book, forward returns ===")
    print("  measured from the imbalance bar's close price to the minute close")
    print("  H minutes later; detrended per tape\n")
    for sym, (theta, use_vol, bars) in chosen.items():
        print(f"  {sym}  (theta={theta:.0f} trades, {len(bars)} bars)")
        for H in (5, 15, 30):
            groups = defaultdict(list)
            per_tape = defaultdict(list)
            for name, bar in bars:
                raw, minutes, s = data[name]
                m0 = int(bar.ts_close * 1000) // MS_MIN * MS_MIN
                tgt = m0 + H * MS_MIN
                if m0 not in minutes or tgt not in minutes:
                    continue
                _, bbo = minutes[m0]
                r = (minutes[tgt][0] - bar.close) / bar.close * 100
                sig = evaluate(bar, m0, bbo, rank(s, bbo), None,
                               require_side="SELL", bbo_min_rank=0.80)
                key = ("FIRED" if sig.fired else
                       f"{bar.side}-side, book rank low")
                groups[key].append(r)
                per_tape[(key, name)].append(r)
            # detrend per tape using all observations from that tape
            tape_mean = defaultdict(list)
            for (key, name), rs in per_tape.items():
                tape_mean[name].extend(rs)
            means = {n: st.mean(v) for n, v in tape_mean.items() if v}
            adj = defaultdict(list)
            for (key, name), rs in per_tape.items():
                m = means.get(name, 0.0)
                adj[key].extend(r - m for r in rs)
            for key in sorted(adj, key=lambda k: -len(adj[k])):
                rs = adj[key]
                if len(rs) < 10:
                    continue
                m = st.mean(rs)
                print(f"    H={H:<3} {key:<28} n={len(rs):<5} "
                      f"mean={m:+.4f}%  win={sum(r>0 for r in rs)/len(rs):>4.0%}"
                      f"  {'CLEARS' if abs(m) > FEE else 'below'} fee")
        print()

    # ---- constant-n rolling window, for comparison ----
    print("=== RollingFlow: what a constant-n window costs in time ===")
    for N in (20, 50, 100):
        for sym in sorted({s for _, _, s in data.values()}):
            spans = []
            for name, (raw, _, s) in data.items():
                if s != sym:
                    continue
                rf = RollingFlow(n=N)
                for tr in raw:
                    rf.add(tr)
                    if rf.ready and rf.span_s:
                        spans.append(rf.span_s)
            if spans:
                print(f"  N={N:<4} {sym:<12} median window span "
                      f"{st.median(spans)/60:>5.1f} min   "
                      f"p90 {sorted(spans)[int(len(spans)*0.9)]/60:>5.1f} min")


if __name__ == "__main__":
    main()
