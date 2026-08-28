#!/usr/bin/env python3
"""
Absolute book depth as a predictor, rather than the bid/ask ratio.

Motivated by a TRXUSDT pump-and-dump chart in which ask depth withdrew in
ABSOLUTE terms (from ~28,000 to near zero) before price completed its ascent.
Every book feature tested so far was the ratio bbo_avg, which is blind to that:
the ratio cannot distinguish "the bid grew" from "the ask evaporated", and it
cannot see total liquidity changing at all.

Four constructions:

  d_ask / d_bid     period-over-period percent change in depth
  ask_rel / bid_rel depth relative to its own trailing 60-candle median
  depth_tot         bid + ask, absolute total liquidity at the touch
  bbo_avg           the existing ratio, for comparison

Result: levels relative to an instrument's own norm predict; period-over-period
changes do not. And depth_tot is significant on POPCAT, where the bbo ratio is
null -- the first feature to work on that instrument.
"""

import math
import statistics as st

from analyze_candles import by_tape, detrend, load, spearman

FEE = 0.20
HORIZONS = (5, 15, 30)
FEATURES = ("d_ask", "d_bid", "ask_rel", "bid_rel", "depth_tot", "bbo_avg")


def enrich(rows):
    """Attach absolute-depth features, computed per tape so trailing medians
    never straddle a disconnect."""
    for _, cs in by_tape(rows).items():
        prev = None
        hist_b, hist_a = [], []
        for c in cs:
            b, a = c.get("bid_sz_avg"), c.get("ask_sz_avg")
            for k in ("d_ask", "d_bid", "ask_rel", "bid_rel", "depth_tot"):
                c[k] = None
            if b is None or a is None:
                continue
            if prev and prev[1]:
                c["d_ask"] = (a - prev[1]) / prev[1] * 100
            if prev and prev[0]:
                c["d_bid"] = (b - prev[0]) / prev[0] * 100
            if len(hist_a) >= 20:
                ma, mb = st.median(hist_a[-60:]), st.median(hist_b[-60:])
                if ma > 0:
                    c["ask_rel"] = a / ma
                if mb > 0:
                    c["bid_rel"] = b / mb
            c["depth_tot"] = b + a
            hist_a.append(a)
            hist_b.append(b)
            prev = (b, a)
    return rows


def z_of(rho, n):
    if rho is None or abs(rho) >= 1 or n <= 4:
        return float("nan")
    return 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 3)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k + 1
    return r


def residualise(target, control):
    """Remove the linear-in-ranks component of control from target."""
    rt, rc = ranks(target), ranks(control)
    mc, mt = st.mean(rc), st.mean(rt)
    den = sum((x - mc) ** 2 for x in rc)
    beta = sum((x - mc) * (y - mt) for x, y in zip(rc, rt)) / den if den else 0.0
    return [y - mt - beta * (x - mc) for x, y in zip(rc, rt)]


def main():
    rows = enrich(load())
    n_tests = len(FEATURES) * len(HORIZONS) * 2
    print(f"{len(rows):,} candles.  {n_tests} tests here; roughly 92 have now been")
    print(f"run against this dataset in total, so Bonferroni wants |z| > 3.5.\n")

    for sym in sorted({r["symbol"] for r in rows}):
        sub = [r for r in rows if r["symbol"] == sym]
        print(f"=== {sym} ({len(sub)} candles) ===")
        print(f"  {'feature':<12} {'h':>4} {'n':>6} {'rho':>8} {'z':>7} "
              f"{'Q5-Q1':>10} {'vs fee':>10}")
        for f in FEATURES:
            for h in HORIZONS:
                pairs = [(c[f], r) for c, r in detrend(sub, h)
                         if c.get(f) is not None]
                if len(pairs) < 200:
                    continue
                xs = [a for a, _ in pairs]
                ys = [b for _, b in pairs]
                rho = spearman(xs, ys)
                z = z_of(rho, len(xs))
                if rho is None:
                    continue
                ps = sorted(pairs)
                k = len(ps) // 5
                sp = (st.mean([b for _, b in ps[4 * k:]])
                      - st.mean([b for _, b in ps[:k]]))
                vs = "clears" if abs(sp) > FEE else f"{FEE/abs(sp):.1f}x short" \
                    if sp else "-"
                mark = "  <-- survives Bonferroni" if abs(z) > 3.5 else ""
                print(f"  {f:<12} {h:>4} {len(xs):>6} {rho:>+8.3f} {z:>+7.2f} "
                      f"{sp:>+9.4f}% {vs:>10}{mark}")
            print()

    # depth_tot correlates with activity; check it is not merely a volume proxy.
    print("=== is depth_tot just a volume proxy? (POPCAT) ===")
    pop = [r for r in rows if r["symbol"] == "POPCATUSDT"]
    pv = [(c["depth_tot"], c["volume"]) for c in pop
          if c.get("depth_tot") is not None and c.get("volume") is not None]
    print(f"  spearman(depth_tot, volume)   = "
          f"{spearman([a for a,_ in pv], [b for _,b in pv]):+.3f}")
    for h in HORIZONS:
        sel = [(c, r) for c, r in detrend(pop, h)
               if c.get("depth_tot") is not None and c.get("volume") is not None]
        if len(sel) < 200:
            continue
        d = [c["depth_tot"] for c, _ in sel]
        v = [c["volume"] for c, _ in sel]
        y = [r for _, r in sel]
        raw = spearman(d, y)
        adj = spearman(residualise(d, v), y)
        print(f"  h={h:<3} raw rho={raw:+.3f}  volume-adjusted={adj:+.3f}  "
              f"z={z_of(adj, len(y)):+.2f}  "
              f"{'survives' if abs(z_of(adj, len(y))) > 1.96 else 'GONE'}")


if __name__ == "__main__":
    main()
