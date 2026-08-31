#!/usr/bin/env python3
"""
Measure depth velocity on the tapes and test whether it predicts anything.

Runs on the BOOK-UPDATE clock (unique-u changes only, heartbeats excluded), then
aggregates the extreme reading within each minute so it can be tested against
forward returns on the same footing as every other feature.
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from microstructure.depth_velocity import DepthCusum, DepthSnapshot, DepthVelocity

MS_MIN = 60_000
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


def process(path: Path):
    """Return {minute_ms: {feature: value}} plus {minute_ms: close}."""
    vel = DepthVelocity(window=20, history=500)
    cus = DepthCusum(drift=0.05, threshold=1.0)
    per_min = defaultdict(lambda: {"div_max": None, "div_min": None,
                                   "opp_max": None, "ask_drop_min": None,
                                   "bid_build_max": None, "cusum_bid": 0,
                                   "cusum_ask": 0, "updates": 0})
    closes = {}
    seen_u = set()

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic = msg.get("topic", "")
            if topic.startswith("publicTrade"):
                for r in msg.get("data") or []:
                    if r.get("BT"):
                        continue
                    t = int(r["T"])
                    closes[t // MS_MIN * MS_MIN] = float(r["p"])
                continue
            if not topic.startswith("orderbook"):
                continue
            data = msg.get("data") or {}
            b, a = data.get("b") or [], data.get("a") or []
            if not b or not a:
                continue
            u = data.get("u")
            if u is not None:
                if u in seen_u:      # heartbeat: not a book CHANGE
                    continue
                seen_u.add(u)
            ts_ms = int(msg.get("cts") or msg.get("ts") or 0)
            if not ts_ms:
                continue
            snap = DepthSnapshot(ts=ts_ms / 1000.0,
                                 bid_sz=float(b[0][1]), ask_sz=float(a[0][1]))
            vel.update(snap)
            fired = cus.update(snap)
            m = ts_ms // MS_MIN * MS_MIN
            rec = per_min[m]
            rec["updates"] += 1
            if fired == "BID_BUILDING":
                rec["cusum_bid"] += 1
            elif fired == "ASK_BUILDING":
                rec["cusum_ask"] += 1
            dz = vel.divergence_z
            if dz is not None:
                rec["div_max"] = dz if rec["div_max"] is None else max(rec["div_max"], dz)
                rec["div_min"] = dz if rec["div_min"] is None else min(rec["div_min"], dz)
            op = vel.opposition
            if op is not None:
                rec["opp_max"] = op if rec["opp_max"] is None else max(rec["opp_max"], op)
            az = vel.d_log_ask_z
            if az is not None:
                rec["ask_drop_min"] = az if rec["ask_drop_min"] is None \
                    else min(rec["ask_drop_min"], az)
            bz = vel.d_log_bid_z
            if bz is not None:
                rec["bid_build_max"] = bz if rec["bid_build_max"] is None \
                    else max(rec["bid_build_max"], bz)
    return per_min, closes


def main():
    tapes = sorted(Path(p) for p in glob.glob("*.jsonl"))
    if not tapes:
        print("no tapes found")
        return

    rows = []   # (symbol, tape, minute, features, close)
    for t in tapes:
        sym = "TRXUSDT" if "TRX" in t.name.upper() else "POPCATUSDT"
        per_min, closes = process(t)
        mins = sorted(m for m in per_min if m in closes)
        for m in mins:
            rows.append((sym, t.name, m, per_min[m], closes[m]))
        if mins:
            print(f"{t.name:<36} {len(mins):>5} minutes, "
                  f"{sum(per_min[m]['updates'] for m in mins):>7,} book changes, "
                  f"cusum fires: bid={sum(per_min[m]['cusum_bid'] for m in mins)} "
                  f"ask={sum(per_min[m]['cusum_ask'] for m in mins)}")

    feats = ("div_max", "div_min", "opp_max", "ask_drop_min", "bid_build_max")
    print(f"\n=== depth-velocity features vs detrended forward return ===")
    print("  (extreme reading within each minute; detrended per tape)\n")
    for sym in ("TRXUSDT", "POPCATUSDT"):
        sub = [r for r in rows if r[0] == sym]
        if len(sub) < 200:
            continue
        print(f"  {sym} ({len(sub)} minutes)")
        print(f"    {'feature':<16} {'h':>4} {'n':>6} {'rho':>8} {'z':>7} {'Q5-Q1':>10}")
        by_tape = defaultdict(list)
        for r in sub:
            by_tape[r[1]].append(r)
        for f in feats:
            for h in (5, 15, 30):
                pooled = []
                for tape, cs in by_tape.items():
                    cs = sorted(cs, key=lambda r: r[2])
                    pairs = []
                    for i, r in enumerate(cs):
                        if i + h >= len(cs):
                            continue
                        v = r[3][f]
                        if v is None or r[4] <= 0:
                            continue
                        pairs.append((v, (cs[i + h][4] - r[4]) / r[4] * 100))
                    if len(pairs) < 10:
                        continue
                    mu = st.mean(p[1] for p in pairs)
                    pooled += [(a, b - mu) for a, b in pairs]
                if len(pooled) < 150:
                    continue
                xs = [a for a, _ in pooled]
                ys = [b for _, b in pooled]
                rho = spearman(xs, ys)
                z = zof(rho, len(xs))
                ps = sorted(pooled)
                k = len(ps) // 5
                sp = (st.mean([b for _, b in ps[4 * k:]])
                      - st.mean([b for _, b in ps[:k]])) if k >= 6 else float("nan")
                mark = "  <--" if abs(z) > 3.5 else ""
                print(f"    {f:<16} {h:>4} {len(xs):>6} {rho:>+8.3f} {z:>+7.2f} "
                      f"{sp:>+9.4f}%{mark}")
            print()


if __name__ == "__main__":
    main()
