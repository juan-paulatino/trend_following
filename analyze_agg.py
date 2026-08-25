#!/usr/bin/env python3
"""
Is aggressive flow contrarian? Pool the runs and test it.

Each run is detrended by subtracting its own mean forward return, which removes
the regime drift and isolates the CROSS-SECTIONAL relationship: within a
session, do high-agg candles do better or worse than low-agg candles?

Without detrending, any relationship is confounded by the fact that run 1 rose
11% and run 2 fell 5%.
"""

import statistics as st
import sys
from pathlib import Path

from analyze_run import CANDLE, strip_rtf


def load(path: Path):
    out = []
    for ln in strip_rtf(path.read_text(errors="replace")).splitlines():
        m = CANDLE.search(ln)
        if not m:
            continue
        _, phase, close, agg, n, bbo, pinned, _ = m.groups()
        out.append({
            "phase": phase,
            "close": float(close),
            "agg": None if agg == "None" else float(agg),
            "n": float(n.replace(",", "")),
            "bbo": None if bbo == "None" else float(bbo),
        })
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def build(runs, horizon):
    """Return detrended (agg, fwd_return) pairs pooled across runs."""
    pooled = []
    for name, cs in runs:
        rows = []
        for i, c in enumerate(cs):
            if i + horizon >= len(cs) or c["agg"] is None:
                continue
            r = (cs[i + horizon]["close"] - c["close"]) / c["close"] * 100
            rows.append((c["agg"], r, c["bbo"], c["n"]))
        if not rows:
            continue
        mean_r = st.mean(r for _, r, _, _ in rows)
        # detrend: express each return relative to this session's own drift
        pooled.extend((a, r - mean_r, b, n) for a, r, b, n in rows)
    return pooled


def quintiles(pairs, key=0, label="agg"):
    rows = sorted(pairs, key=lambda p: p[key])
    k = len(rows) // 5
    if k < 3:
        return
    print(f"    {label} quintile      n   mean bucket   detrended fwd%   win%")
    for q in range(5):
        lo = q * k
        hi = (q + 1) * k if q < 4 else len(rows)
        chunk = rows[lo:hi]
        vals = [c[key] for c in chunk]
        rets = [c[1] for c in chunk]
        win = sum(r > 0 for r in rets) / len(rets)
        print(f"      Q{q+1} {min(vals):.3f}-{max(vals):.3f} {len(chunk):>5} "
              f"{st.mean(vals):>10.3f}   {st.mean(rets):>+13.3f}   {win:>5.0%}")


def main(paths):
    runs = [(p.name, load(p)) for p in paths]
    for name, cs in runs:
        aggs = [c["agg"] for c in cs if c["agg"] is not None]
        drift = (cs[-1]["close"] / cs[0]["close"] - 1) * 100
        print(f"{name}: {len(cs)} candles, drift {drift:+.2f}%, "
              f"agg median {st.median(aggs):.3f}")

    print("\n=== agg vs DETRENDED forward return (pooled across both runs) ===")
    print("  detrending removes each session's drift, isolating the")
    print("  cross-sectional effect\n")
    for h in (1, 3, 5, 10):
        pairs = build(runs, h)
        if not pairs:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        # crude two-sided significance for rho via Fisher z
        n = len(xs)
        z = None
        if rho is not None and n > 4 and abs(rho) < 1:
            import math
            z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 3)
        print(f"  h={h:<3} n={n:<5} pearson={r:+.3f}  spearman={rho:+.3f}"
              + (f"  z={z:+.2f}" + ("  SIGNIFICANT" if abs(z) > 1.96 else "  (ns)")
                 if z is not None else ""))
        quintiles(pairs, 0, "agg")
        print()

    print("=== same test, per run, to check the sign is stable ===")
    for name, cs in runs:
        for h in (1, 5):
            pairs = build([(name, cs)], h)
            if len(pairs) < 10:
                continue
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            print(f"  {name:<14} h={h:<3} spearman(agg, fwd) = {rho:+.3f}")

    print("\n=== bbo vs detrended forward return (for comparison) ===")
    for h in (1, 5):
        pairs = [p for p in build(runs, h) if p[2] is not None]
        if len(pairs) < 10:
            continue
        rho = spearman([p[2] for p in pairs], [p[1] for p in pairs])
        print(f"  h={h:<3} n={len(pairs):<5} spearman(bbo, fwd) = {rho:+.3f}")
        quintiles(pairs, 2, "bbo")
        print()


if __name__ == "__main__":
    args = sys.argv[1:] or ["output.rtf", "output2.rtf"]
    main([Path(a) for a in args])
