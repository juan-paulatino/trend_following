#!/usr/bin/env python3
"""
Aggregate the trade record across every v9e_Buy_Frac34 run log.

Ten runs is enough to ask the questions a single run cannot: does the exit-rule
mix hold up, is the profit still carried by a handful of trades, and is any of
it above fees.
"""

import glob
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from analyze_run import strip_rtf

TAKER_RT = 0.11   # Bybit linear taker ~0.055% per side
MAKER_RT = 0.04

EXIT = re.compile(
    r"EXIT LONG at ([\d\- :,]+)\s*\|\s*exit_price=([\d.]+)\s*\|\s*"
    r"entry_price=([\d.]+)\s*\|\s*pnl=(-?[\d.]+)\s*\|\s*reasons:\s*(.*)"
)
GATE = re.compile(r"BUY_FRAC confirmation gate \(([^)]+)\):\s*(ALLOW|BLOCK) LONG"
                  r"\s*\|\s*buy_frac=([\d.]+)")


def rule_of(reason: str) -> str:
    for tag, name in (("RULE 1", "RULE 1 PANIC"), ("RULE 2", "RULE 2 HOLD"),
                      ("RULE 3", "RULE 3 WEAR"), ("RULE 4", "RULE 4 G-R-R")):
        if tag in reason:
            return name
    return "other"


def load(path: Path):
    txt = strip_rtf(path.read_text(errors="replace"))
    trades, gates = [], []
    for l in txt.splitlines():
        m = EXIT.search(l)
        if m:
            ts, xp, ep, _pnl, reason = m.groups()
            xp, ep = float(xp), float(ep)
            if ep <= 0:
                continue
            trades.append({
                "run": path.name, "ts": ts.strip()[:19],
                "entry": ep, "exit": xp,
                "ret": (xp - ep) / ep * 100,
                "rule": rule_of(reason),
            })
        g = GATE.search(l)
        if g:
            gates.append((g.group(1), g.group(2), float(g.group(3))))
    return trades, gates


def main():
    files = sorted(Path(p) for p in glob.glob("frac34*.rtf"))
    if not files:
        print("no frac34*.rtf logs found")
        return

    all_tr, all_gates = [], []
    print(f"{'run':<16} {'trades':>7} {'gross%':>9} {'mean%':>9} {'win':>5}")
    for f in files:
        tr, gt = load(f)
        all_tr.extend(tr)
        all_gates.extend(gt)
        if tr:
            r = [t["ret"] for t in tr]
            print(f"{f.name:<16} {len(tr):>7} {sum(r):>+9.3f} {st.mean(r):>+9.4f} "
                  f"{sum(1 for x in r if x > 0)/len(r):>4.0%}")
        else:
            print(f"{f.name:<16} {0:>7}")

    rets = [t["ret"] for t in all_tr]
    n = len(rets)
    print(f"\n{'=' * 66}\nAGGREGATE: {n} round trips across {len(files)} runs\n{'=' * 66}")
    print(f"  gross mean        {st.mean(rets):+.4f}%")
    print(f"  gross median      {st.median(rets):+.4f}%")
    print(f"  gross total       {sum(rets):+.3f}%")
    print(f"  win rate          {sum(1 for r in rets if r > 0)/n:.1%}")
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    print(f"  mean win / loss   {st.mean(wins):+.4f}% / {st.mean(losses):+.4f}%")
    print(f"  payoff ratio      {abs(st.mean(wins)/st.mean(losses)):.2f}")
    print(f"  stdev             {st.stdev(rets):.4f}%")
    # t-stat on the gross mean being non-zero
    se = st.stdev(rets) / (n ** 0.5)
    print(f"  t-stat vs zero    {st.mean(rets)/se:+.2f}   "
          f"({'significant' if abs(st.mean(rets)/se) > 1.96 else 'not significant'})")

    print(f"\n  --- after fees, {n} round trips ---")
    for label, fee in (("taker (0.11% RT)", TAKER_RT), ("maker (0.04% RT)", MAKER_RT)):
        net = [r - fee for r in rets]
        print(f"  {label:<18} mean={st.mean(net):+.4f}%  total={sum(net):+8.2f}%  "
              f"win={sum(1 for r in net if r > 0)/n:.0%}")
    print(f"  fee drag alone     taker {n*TAKER_RT:.2f}%   maker {n*MAKER_RT:.2f}%")
    print(f"  breakeven gross needed: {TAKER_RT:.3f}% per trade "
          f"(have {st.mean(rets):.4f}%, "
          f"{TAKER_RT/max(st.mean(rets),1e-9):.0f}x short)")

    print(f"\n  --- exit rules ---")
    print(f"  {'rule':<14} {'n':>5} {'share':>7} {'mean%':>9} {'total%':>9} "
          f"{'win':>5} {'fee cost':>9}")
    for rule, cnt in Counter(t["rule"] for t in all_tr).most_common():
        sub = [t["ret"] for t in all_tr if t["rule"] == rule]
        print(f"  {rule:<14} {cnt:>5} {cnt/n:>6.0%} {st.mean(sub):>+9.4f} "
              f"{sum(sub):>+9.3f} {sum(1 for x in sub if x>0)/len(sub):>4.0%} "
              f"{cnt*TAKER_RT:>8.2f}%")

    print(f"\n  --- concentration ---")
    s = sorted(rets, reverse=True)
    tot = sum(rets)
    for k in (1, 2, 5, 10):
        if k < n:
            print(f"  top {k:<3} trades {sum(s[:k]):>+8.3f}%   "
                  f"remaining {n-k:<4} {sum(s[k:]):>+8.3f}%")
    print(f"  share of gross from best 10%: "
          f"{sum(s[:max(1,n//10)])/tot:.0%}" if tot != 0 else "")

    print(f"\n  --- entry gate ---")
    for kind, cnt in Counter(g[0] for g in all_gates).most_common():
        allow = sum(1 for g in all_gates if g[0] == kind and g[1] == "ALLOW")
        print(f"  {kind:<34} {cnt:>4} evaluated, {allow} allowed "
              f"({allow/cnt:.0%})")
    allowed = [g[2] for g in all_gates if g[1] == "ALLOW"]
    blocked = [g[2] for g in all_gates if g[1] == "BLOCK"]
    if allowed and blocked:
        print(f"  buy_frac when allowed: median {st.median(allowed):.4f}")
        print(f"  buy_frac when blocked: median {st.median(blocked):.4f}")

    # Did the gate actually select better trades? Compare across runs.
    print(f"\n  --- is a low win rate with big winners working? ---")
    print(f"  expectancy = win%*meanWin + loss%*meanLoss")
    wr = len(wins)/n
    exp = wr*st.mean(wins) + (1-wr)*st.mean(losses)
    print(f"    gross      {wr:.1%} * {st.mean(wins):+.4f} + "
          f"{1-wr:.1%} * {st.mean(losses):+.4f} = {exp:+.4f}%")
    print(f"    net taker  {exp - TAKER_RT:+.4f}%  per trade")
    need_wr = (TAKER_RT - st.mean(losses)) / (st.mean(wins) - st.mean(losses))
    print(f"  win rate needed to break even at taker fees: {need_wr:.1%} "
          f"(currently {wr:.1%})")


if __name__ == "__main__":
    main()
