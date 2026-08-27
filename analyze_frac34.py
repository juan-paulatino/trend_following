#!/usr/bin/env python3
"""Extract and evaluate the trade record from a v9e_Buy_Frac34 log."""

import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from analyze_run import strip_rtf

# Bybit linear taker is ~0.055% per side; maker ~0.02%. Round trips:
TAKER_RT = 0.11
MAKER_RT = 0.04

EXIT = re.compile(
    r"EXIT LONG at ([\d\- :,]+)\s*\|\s*exit_price=([\d.]+)\s*\|\s*"
    r"entry_price=([\d.]+)\s*\|\s*pnl=(-?[\d.]+)\s*\|\s*reasons:\s*(.*)"
)


def main(path):
    txt = strip_rtf(Path(path).read_text(errors="replace"))
    lines = [l for l in txt.splitlines() if l.strip()]

    trades = []
    for l in lines:
        m = EXIT.search(l)
        if m:
            ts, xp, ep, pnl, reason = m.groups()
            xp, ep = float(xp), float(ep)
            trades.append({
                "ts": ts.strip(), "entry": ep, "exit": xp,
                "pnl_abs": float(pnl),
                "ret_pct": (xp - ep) / ep * 100,
                "reason": reason.strip()[:60],
            })

    if not trades:
        print("no completed trades found")
        return

    rets = [t["ret_pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    flats = [r for r in rets if r == 0]

    print(f"=== {Path(path).name}: {len(trades)} completed round trips ===\n")
    print(f"  gross mean return   {st.mean(rets):+.4f}%")
    print(f"  gross median        {st.median(rets):+.4f}%")
    print(f"  gross total         {sum(rets):+.4f}%")
    print(f"  win / loss / flat   {len(wins)} / {len(losses)} / {len(flats)}"
          f"   ({len(wins)/len(trades):.0%} win)")
    if wins:
        print(f"  mean win            {st.mean(wins):+.4f}%   best {max(wins):+.4f}%")
    if losses:
        print(f"  mean loss           {st.mean(losses):+.4f}%   worst {min(losses):+.4f}%")
    print(f"  best / worst        {max(rets):+.4f}% / {min(rets):+.4f}%")

    print(f"\n  --- after fees ---")
    for label, fee in (("taker both sides", TAKER_RT), ("maker both sides", MAKER_RT)):
        net = [r - fee for r in rets]
        print(f"  {label:<18} mean={st.mean(net):+.4f}%  total={sum(net):+.4f}%  "
              f"win={sum(1 for r in net if r > 0)/len(net):.0%}")

    print(f"\n  --- exit reasons ---")
    for reason, n in Counter(t["reason"].split("|")[0].strip()
                             for t in trades).most_common():
        sub = [t["ret_pct"] for t in trades
               if t["reason"].split("|")[0].strip() == reason]
        print(f"  {n:>3}x  mean {st.mean(sub):+.4f}%  {reason[:56]}")

    print(f"\n  --- every trade ---")
    cum = 0.0
    for t in trades:
        cum += t["ret_pct"]
        print(f"  {t['ts'][:19]}  {t['entry']:.5f} -> {t['exit']:.5f}  "
              f"{t['ret_pct']:+7.4f}%  cum {cum:+8.4f}%   {t['reason'][:44]}")

    # how much of the gross is one trade?
    biggest = max(rets, key=abs)
    print(f"\n  largest single trade is {biggest:+.4f}%, "
          f"{abs(biggest)/max(abs(sum(rets)), 1e-9):.0%} of the gross total")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "frac34_9.rtf")
