#!/usr/bin/env python3
"""
Pre-registered tests on the newly uploaded tapes.

Two hypotheses were stated in earlier PRs, before this data existed:

  H1 (PR #8, #12, #14, #15)  bbo_avg predicts POSITIVELY on TRXUSDT.
                             Confirmed five times; rho ~0.15-0.32.
  H2 (PR #13)                depth_tot = bid_sz + ask_sz predicts POSITIVELY on
                             POPCATUSDT, where the bbo ratio is null.
                             rho +0.112 (z +4.84) at h=15 in-sample.

Tapes recorded on 2026-08-25/26 are IN-SAMPLE. Tapes from 08-27 onward are the
holdout. Each is tested separately; pooling them would not be a holdout test.

Returns are detrended per tape. Position-in-run is controlled for, because a
feature that merely occurs early in a one-directional run looks predictive after
detrending -- the artifact that produced a spurious z of -5.45 in PR #14.
"""

from __future__ import annotations

import glob
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

from microstructure.bybit import BybitAssembler
from microstructure.phases import PhaseMachine

TICK = {"POPCATUSDT": 0.00001, "TRXUSDT": 0.0001}
IN_SAMPLE_DATES = ("20260825", "20260826")
FEE = 0.20


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
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def zof(rho, n):
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
    rt, rc = ranks(target), ranks(control)
    mc, mt = st.mean(rc), st.mean(rt)
    den = sum((x - mc) ** 2 for x in rc)
    beta = sum((x - mc) * (y - mt) for x, y in zip(rc, rt)) / den if den else 0.0
    return [y - mt - beta * (x - mc) for x, y in zip(rc, rt)]


def load_tape(path: Path):
    sym = "TRXUSDT" if "TRX" in path.name.upper() else "POPCATUSDT"
    asm = BybitAssembler(tick_size=TICK[sym])
    machine = PhaseMachine()
    rows = []

    def take(em):
        if not em.usable:
            return
        c = em.candle
        if c.n_trades == 0 or c.close <= 0:
            return
        machine.update(c)
        rows.append({
            "close": c.close,
            "bbo": c.bbo_avg,
            "depth_tot": (c.bid_sz_avg + c.ask_sz_avg)
            if (c.bid_sz_avg is not None and c.ask_sz_avg is not None) else None,
            "volume": c.volume,
        })

    import json
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
                take(em)
    for em in asm.flush():
        take(em)
    return sym, rows


def build(tapes, feature, h):
    """Detrended pairs plus position index, pooled across the given tapes."""
    out = []
    for rows in tapes:
        pairs = []
        for i, c in enumerate(rows):
            if i + h >= len(rows) or c.get(feature) is None:
                continue
            pairs.append((i, c[feature],
                          (rows[i + h]["close"] - c["close"]) / c["close"] * 100))
        if len(pairs) < 20:
            continue
        mu = st.mean(p[2] for p in pairs)
        out += [(i, v, r - mu) for i, v, r in pairs]
    return out


def report(label, tapes, feature, expect, horizons=(5, 15, 30)):
    print(f"\n  {label}")
    n_tot = sum(len(t) for t in tapes)
    det = math.tanh((1.96 + 0.84) / math.sqrt(max(n_tot - 3, 1)))
    print(f"    {n_tot} candles, resolves rho > {det:.3f} "
          f"({'POWERED' if n_tot >= 269 else 'UNDERPOWERED'})")
    print(f"    {'h':>4} {'n':>6} {'rho':>8} {'z':>7} {'pos-adj':>9} {'z':>7} "
          f"{'Q5-Q1':>10} {'verdict':>9}")
    for h in horizons:
        data = build(tapes, feature, h)
        if len(data) < 150:
            continue
        pos = [float(i) for i, _, _ in data]
        xs = [v for _, v, _ in data]
        ys = [r for _, _, r in data]
        rho = spearman(xs, ys)
        adj = spearman(residualise(xs, pos), ys)
        ps = sorted(zip(xs, ys))
        k = len(ps) // 5
        sp = (st.mean([b for _, b in ps[4 * k:]])
              - st.mean([b for _, b in ps[:k]])) if k >= 6 else float("nan")
        za, zr = zof(adj, len(ys)), zof(rho, len(ys))
        held = (adj > 0) if expect == "POSITIVE" else (adj < 0)
        sig = abs(za) > 1.96
        verdict = ("HELD" if sig else "held-ns") if held else \
                  ("FLIPPED" if sig else "flip-ns")
        print(f"    {h:>4} {len(ys):>6} {rho:>+8.3f} {zr:>+7.2f} {adj:>+9.3f} "
              f"{za:>+7.2f} {sp:>+9.4f}% {verdict:>9}")


def main():
    tapes = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(x) for x in glob.glob("*.jsonl")):
        if os.path.getsize(p) == 0:
            continue
        era = "in-sample" if any(d in p.name for d in IN_SAMPLE_DATES) else "HOLDOUT"
        sym, rows = load_tape(p)
        if len(rows) >= 20:
            tapes[sym][era].append(rows)

    for sym in tapes:
        for era in tapes[sym]:
            n = sum(len(t) for t in tapes[sym][era])
            print(f"{sym:<12} {era:<10} {len(tapes[sym][era])} tapes, {n:>5} candles")

    print("\n" + "=" * 78)
    print("H1 (PR #8/#12/#14/#15): bbo_avg predicts POSITIVELY on TRXUSDT")
    print("=" * 78)
    for era in ("in-sample", "HOLDOUT"):
        if tapes["TRXUSDT"][era]:
            report(era, tapes["TRXUSDT"][era], "bbo", "POSITIVE")

    print("\n" + "=" * 78)
    print("H2 (PR #13): depth_tot predicts POSITIVELY on POPCATUSDT")
    print("=" * 78)
    for era in ("in-sample", "HOLDOUT"):
        if tapes["POPCATUSDT"][era]:
            report(era, tapes["POPCATUSDT"][era], "depth_tot", "POSITIVE")

    print("\n  --- and bbo on POPCAT, which was established as NULL ---")
    if tapes["POPCATUSDT"]["HOLDOUT"]:
        report("HOLDOUT", tapes["POPCATUSDT"]["HOLDOUT"], "bbo", "POSITIVE")


if __name__ == "__main__":
    main()
