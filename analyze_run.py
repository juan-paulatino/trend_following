#!/usr/bin/env python3
"""Parse a collector log (plain text or RTF) and report what actually happened."""

import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path


def strip_rtf(raw: str) -> str:
    if not raw.lstrip().startswith("{\\rtf"):
        return raw
    txt = re.sub(r"\{\\\*?\\[^{}]*\}", "", raw)      # groups like {\fonttbl...}
    txt = re.sub(r"^\{\\rtf[^\n]*\n", "", txt)
    txt = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), txt)
    txt = txt.replace("\\\n", "\n")                   # line-continuation backslash
    txt = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", txt)      # control words
    txt = txt.replace("{", "").replace("}", "")
    return txt


CANDLE = re.compile(
    r"\[(\d{2}:\d{2})\]\s+(\w+)\s+close=([\d.eE+-]+)\s+"
    r"agg=\s*([\d.]+|None)\s+\(n=\s*([\d,]+)\)\s+bbo=\s*([\d.]+|None)\s+"
    r"pinned=\s*(\d+%|n/a)(.*)"
)
GAP = re.compile(r"\[(\d{2}:\d{2})\]\s+GAP")
DUPES = re.compile(r"filtered (\d+) block trades, (\d+) duplicate book snapshots")
BOUND = re.compile(r"inferred from fills alone: >([\d.]+)(?: and <=([\d.]+))?")
WALL = re.compile(r"absorbed a sell worth (\d+)% of displayed bid")
EPISODE = re.compile(r"episode: (\d+) candles, ([\d.]+) net supply absorbed for "
                     r"([+-][\d.]+)% \(([^)]*)\)")


def main(path: Path) -> None:
    text = strip_rtf(path.read_text(errors="replace"))
    lines = text.splitlines()

    candles, gaps = [], []
    blocks = dupes = 0
    bounds, walls, episodes = [], [], []
    cur = None

    for ln in lines:
        m = CANDLE.search(ln)
        if m:
            t, phase, close, agg, n, bbo, pinned, tail = m.groups()
            cur = {
                "t": t, "phase": phase, "close": float(close),
                "agg": None if agg == "None" else float(agg),
                "n": float(n.replace(",", "")),
                "bbo": None if bbo == "None" else float(bbo),
                "pinned": None if pinned == "n/a" else int(pinned.rstrip("%")) / 100,
                "low_conf": "low confidence" in tail,
                "reasons": [],
            }
            candles.append(cur)
            continue
        if GAP.search(ln):
            gaps.append(GAP.search(ln).group(1))
            continue
        d = DUPES.search(ln)
        if d:
            blocks += int(d.group(1)); dupes += int(d.group(2))
        b = BOUND.search(ln)
        if b:
            bounds.append((float(b.group(1)),
                           float(b.group(2)) if b.group(2) else None))
        w = WALL.search(ln)
        if w:
            walls.append(int(w.group(1)) / 100)
        e = EPISODE.search(ln)
        if e:
            episodes.append((int(e.group(1)), float(e.group(2)),
                             float(e.group(3)), e.group(4)))
        if cur is not None and ln.strip().startswith("- "):
            cur["reasons"].append(ln.strip()[2:])

    print(f"file: {path.name}   {len(lines):,} lines")
    print(f"candles classified: {len(candles)}   gaps: {len(gaps)}")
    print(f"filtered: {blocks} block trades, {dupes} duplicate book snapshots\n")

    print("=== phase distribution ===")
    counts = Counter(c["phase"] for c in candles)
    for ph, n in counts.most_common():
        print(f"  {ph:<14} {n:>4}  {n/len(candles):>6.1%}")

    lowc = sum(c["low_conf"] for c in candles)
    print(f"\n  low confidence: {lowc} ({lowc/max(len(candles),1):.1%})")

    def dist(name, vals, pct=False):
        vals = [v for v in vals if v is not None]
        if not vals:
            print(f"  {name}: no data")
            return
        q = sorted(vals)
        f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.3f}")
        print(f"  {name:<10} n={len(vals):<5} "
              f"min={f(q[0])} p25={f(q[len(q)//4])} med={f(q[len(q)//2])} "
              f"p75={f(q[3*len(q)//4])} max={f(q[-1])}")

    print("\n=== metric distributions, ALL candles ===")
    dist("agg", [c["agg"] for c in candles])
    dist("bbo", [c["bbo"] for c in candles])
    dist("pinned", [c["pinned"] for c in candles], pct=True)
    dist("n (vol)", [c["n"] for c in candles])

    absorp = [c for c in candles if c["phase"] == "ABSORPTION"]
    if absorp:
        print("\n=== during ABSORPTION only ===")
        dist("agg", [c["agg"] for c in absorp])
        dist("bbo", [c["bbo"] for c in absorp])
        dist("pinned", [c["pinned"] for c in absorp], pct=True)

    if bounds:
        inverted = [(lo, hi) for lo, hi in bounds if hi is not None and hi <= lo]
        print(f"\n=== inferred-depth bounds ===")
        print(f"  reported: {len(bounds)}")
        print(f"  INVERTED (lower >= upper): {len(inverted)} "
              f"({len(inverted)/len(bounds):.0%})  <-- nonsensical as an interval")
        if inverted:
            print(f"  examples: " + ", ".join(f">{lo:.0f} and <={hi:.0f}"
                                              for lo, hi in inverted[:4]))

    if walls:
        over = [w for w in walls if w > 1.0]
        print(f"\n=== wall_tested ===")
        dist("ratio", walls, pct=True)
        print(f"  absorbed MORE than the displayed bid: {len(over)}/{len(walls)} "
              f"({len(over)/len(walls):.0%})  <-- refill or hidden size")

    # ---- forward returns: does a phase predict anything? ----
    print("\n=== forward returns by phase (the only question that matters) ===")
    by_phase = defaultdict(list)
    for i, c in enumerate(candles):
        for h in (1, 3, 5, 10):
            if i + h < len(candles):
                r = (candles[i + h]["close"] - c["close"]) / c["close"] * 100
                by_phase[(c["phase"], h)].append(r)

    print(f"  {'phase':<14} {'h':>3} {'n':>5} {'mean%':>8} {'median%':>8} {'win%':>7}")
    for ph in [p for p, _ in counts.most_common()]:
        for h in (1, 3, 5, 10):
            rs = by_phase.get((ph, h), [])
            if len(rs) < 3:
                continue
            win = sum(r > 0 for r in rs) / len(rs)
            print(f"  {ph:<14} {h:>3} {len(rs):>5} {st.mean(rs):>8.3f} "
                  f"{st.median(rs):>8.3f} {win:>6.0%}")

    base = [(candles[i+1]["close"] - candles[i]["close"]) / candles[i]["close"] * 100
            for i in range(len(candles) - 1)]
    if base:
        print(f"\n  baseline (all candles, h=1): mean={st.mean(base):+.3f}% "
              f"median={st.median(base):+.3f}% "
              f"win={sum(r>0 for r in base)/len(base):.0%}")

    if episodes:
        print("\n=== episodes ===")
        no_base = sum(1 for _, _, _, z in episodes if "no vol baseline" in z)
        print(f"  episode lines: {len(episodes)}, of which "
              f"{no_base} ({no_base/len(episodes):.0%}) had NO volatility baseline")
        longest = max(episodes, key=lambda e: e[0])
        print(f"  longest: {longest[0]} candles, {longest[1]:,.0f} supply, "
              f"{longest[2]:+.2f}% ({longest[3]})")

    print(f"\n=== span ===")
    if candles:
        print(f"  {candles[0]['t']} -> {candles[-1]['t']}   "
              f"close {candles[0]['close']:.5f} -> {candles[-1]['close']:.5f}  "
              f"({(candles[-1]['close']/candles[0]['close']-1)*100:+.2f}%)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "output.rtf"))
