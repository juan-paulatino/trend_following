#!/usr/bin/env python3
"""
Analyse candles.csv -- every tape replayed through the fixed code.

Returns are detrended PER TAPE, so each session's drift cannot masquerade as
signal, and are reported against the 0.20% round-trip taker fee.
"""

import csv
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

FEE = 0.20


def load(path="candles.csv"):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["usable"] != "1":
                continue
            for k, v in list(r.items()):
                if k in ("tape", "symbol", "phase"):
                    continue
                r[k] = float(v) if v not in ("", None) else None
            rows.append(r)
    return rows


def by_tape(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["tape"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["minute_ms"])
    return d


def detrend(rows, h):
    """Attach a detrended forward return to every candle.

    The tape mean must be computed over ALL candles in that tape and only then
    may subsets be selected. Detrending a subset centres that subset on zero by
    construction, which silently forces every group mean to 0.0000 -- the bug
    this replaced.
    """
    out = []
    for _, cs in by_tape(rows).items():
        rs = []
        for i, c in enumerate(cs):
            if i + h >= len(cs) or c["close"] <= 0:
                continue
            rs.append((c, (cs[i + h]["close"] - c["close"]) / c["close"] * 100))
        if len(rs) < 5:
            continue
        m = st.mean(r for _, r in rs)
        out.extend((c, r - m) for c, r in rs)
    return out


def fwd_pairs(rows, h, field, _cache={}):
    """(feature value, detrended fwd return), detrended against the FULL tape."""
    key = (id(rows), h)
    if key not in _cache:
        _cache[key] = detrend(rows, h)
    return [(c[field], r) for c, r in _cache[key] if c.get(field) is not None]


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def welch(a, b):
    if len(a) < 3 or len(b) < 3:
        return None, None
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    if se == 0:
        return None, None
    t = (st.mean(a) - st.mean(b)) / se
    return t, 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def main():
    rows = load()
    print(f"{len(rows):,} usable candles across {len(by_tape(rows))} tapes\n")

    for sym in sorted({r["symbol"] for r in rows}):
        sub = [r for r in rows if r["symbol"] == sym]
        tr = [r["n_trades"] for r in sub]
        print(f"  {sym:<12} {len(sub):>5} candles  "
              f"median {st.median(tr):>4.0f} trades/candle  "
              f"zero-trade {sum(1 for t in tr if t == 0)/len(tr):>5.1%}")

    print("\n=== phase distribution (corrected code) ===")
    ph = Counter(r["phase"] for r in rows)
    for p, n in ph.most_common():
        print(f"  {p:<14} {n:>5}  {n/len(rows):>6.1%}")

    print("\n=== corrected metric medians, by symbol ===")
    for f in ("agg", "bbo_avg", "absorption_tick_ratio", "sell_efficiency",
              "pinned_share", "wall_tested"):
        line = f"  {f:<22}"
        for sym in sorted({r["symbol"] for r in rows}):
            vals = [r[f] for r in rows if r["symbol"] == sym and r[f] is not None]
            line += f"  {sym[:6]}={st.median(vals):.3f} (n={len(vals)})" if vals \
                else f"  {sym[:6]}=none"
        print(line)

    print("\n=== forward returns by phase (detrended against the FULL tape) ===")
    print(f"  {'phase':<14} {'h':>3} {'n':>6} {'mean%':>9} {'median%':>9} {'win%':>6}"
          f" {'vs fee':>8}")
    for h in (1, 5, 15):
        allc = detrend(rows, h)
        for p, _ in ph.most_common():
            rs = [r for c, r in allc if c["phase"] == p]
            if len(rs) < 20:
                continue
            m = st.mean(rs)
            print(f"  {p:<14} {h:>3} {len(rs):>6} {m:>+9.4f} "
                  f"{st.median(rs):>+9.4f} {sum(r>0 for r in rs)/len(rs):>5.0%}"
                  f" {'clears' if abs(m) > FEE else 'below':>8}")
        print()

    print("\n=== does any feature predict? spearman vs detrended fwd return ===")
    print(f"  {'feature':<24} {'h':>3} {'n':>6} {'rho':>8} {'z':>7}")
    for f in ("agg", "bbo_avg", "absorption_tick_ratio", "sell_efficiency",
              "delta", "delta_recovery", "impact_asymmetry", "flow_lag",
              "absorption_per_tick", "volume"):
        for h in (1, 5, 15):
            pairs = fwd_pairs(rows, h, f)
            if len(pairs) < 100:
                continue
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            rho = spearman(xs, ys)
            if rho is None:
                continue
            n = len(xs)
            z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 3) \
                if abs(rho) < 1 else float("nan")
            flag = "  <-- |z|>1.96" if abs(z) > 1.96 else ""
            print(f"  {f:<24} {h:>3} {n:>6} {rho:>+8.3f} {z:>+7.2f}{flag}")

    # ---- robustness: does the effect hold in BOTH symbols and most tapes? ----
    print("\n=== robustness of the significant features ===")
    print("  30 tests were run above (10 features x 3 horizons). Bonferroni at")
    print("  alpha=0.05 requires |z| > 3.14, not 1.96.\n")
    for f in ("bbo_avg", "flow_lag", "agg", "delta"):
        print(f"  {f}")
        for sym in sorted({r["symbol"] for r in rows}):
            sub = [r for r in rows if r["symbol"] == sym]
            for h in (1, 5, 15):
                pairs = fwd_pairs(sub, h, f)
                if len(pairs) < 100:
                    continue
                rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
                if rho is None or abs(rho) >= 1:
                    continue
                n = len(pairs)
                z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 3)
                print(f"    {sym:<12} h={h:<3} n={n:>5} rho={rho:+.3f} z={z:+.2f}")
        # per-tape sign consistency at h=1
        signs = []
        for tape, cs in by_tape(rows).items():
            pairs = fwd_pairs(cs, 1, f)
            if len(pairs) < 50:
                continue
            rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
            if rho is not None:
                signs.append(rho)
        if signs:
            pos = sum(1 for s in signs if s > 0)
            print(f"    per-tape h=1: {pos}/{len(signs)} tapes positive, "
                  f"median rho={st.median(signs):+.3f}")
        print()

    print("\n=== the agg>0.95 hypothesis on the full corrected dataset ===")
    for h in (1, 5, 15):
        pairs = fwd_pairs(rows, h, "agg")
        hi = [r for a, r in pairs if a > 0.95]
        lo = [r for a, r in pairs if a <= 0.95]
        if len(hi) < 10:
            continue
        t, p = welch(hi, lo)
        d = st.mean(hi) - st.mean(lo)
        print(f"  h={h:<3} n_hi={len(hi):<5} diff={d:+.4f}%  t={t:+.2f}  p={p:.3f}"
              f"   {'clears' if abs(d) > FEE else 'BELOW'} {FEE}% fee")


if __name__ == "__main__":
    main()
