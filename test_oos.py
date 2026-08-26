#!/usr/bin/env python3
"""
Out-of-sample test of the ONE hypothesis pre-registered before runs 3 and 4
existed:

    "Extreme aggressive buying (agg > 0.95) is followed by underperformance."

Stated in PR #4 from runs 1-2 only. Runs 3 (POPCAT) and 4 (TRX) are the
held-out sample. Reported whichever way it comes out.

Also reports the fraction of ZERO forward returns, because on a coarse price
grid most candles do not move at all, which makes "win rate" incomparable
across instruments and not centred on 50%.
"""

import math
import statistics as st
import sys
from pathlib import Path

from analyze_agg import load, spearman

# Runs are grouped by WHEN they became available, so that each group after the
# first is a genuine holdout for a hypothesis stated before it existed.
IN_SAMPLE = ["output.rtf", "output2.rtf"]                      # hypothesis born here
OUT_SAMPLE = ["output3_popcat.rtf", "output_trx.rtf"]          # holdout 1 (PR #5)
OUT_SAMPLE_2 = ["output4_popcat.rtf", "output2_trx.rtf"]       # holdout 2 (this PR)
CUTOFF = 0.95


def fwd(cs, i, h):
    if i + h >= len(cs):
        return None
    return (cs[i + h]["close"] - cs[i]["close"]) / cs[i]["close"] * 100


def detrended(paths, h):
    """(agg, detrended return, raw return) triples; each run centred on its own
    mean. The RAW return is carried through because detrending destroys exact
    zeros, and the zero fraction is itself diagnostic -- on a coarse price grid
    most candles do not move at all."""
    out = []
    for p in paths:
        cs = load(Path(p))
        rows = [(c["agg"], fwd(cs, i, h)) for i, c in enumerate(cs)]
        rows = [(a, r) for a, r in rows if a is not None and r is not None]
        if not rows:
            continue
        m = st.mean(r for _, r in rows)
        out.extend((a, r - m, r) for a, r in rows)
    return out


def welch(a, b):
    """Welch t-statistic and two-sided p (normal approximation)."""
    if len(a) < 3 or len(b) < 3:
        return None, None
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None, None
    t = (st.mean(a) - st.mean(b)) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def report(label, paths):
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    for h in (1, 3, 5, 10):
        pairs = detrended(paths, h)
        if len(pairs) < 20:
            print(f"  h={h}: only {len(pairs)} observations, skipping")
            continue
        # effect must clear the round-trip taker fee to be worth anything
        FEE = 0.20
        hi = [d for a, d, _ in pairs if a > CUTOFF]
        lo = [d for a, d, _ in pairs if a <= CUTOFF]
        if len(hi) < 3:
            print(f"  h={h}: only {len(hi)} candles above agg>{CUTOFF}, skipping")
            continue
        t, p = welch(hi, lo)
        # count zeros on the RAW returns, before detrending shifted them
        zeros = sum(1 for _, _, raw in pairs if abs(raw) < 1e-12)
        verdict = ""
        if p is not None:
            verdict = "  <-- p<0.05" if p < 0.05 else "  (ns)"
        print(f"  h={h:<3} agg>{CUTOFF}: n={len(hi):<4} mean={st.mean(hi):+.4f}%   "
              f"rest: n={len(lo):<4} mean={st.mean(lo):+.4f}%")
        if t is not None:
            diff = st.mean(hi) - st.mean(lo)
            cost = "BELOW round-trip fee" if abs(diff) < FEE else "clears fee"
            print(f"        difference {diff:+.4f}%  "
                  f"t={t:+.2f}  p={p:.3f}{verdict}")
            print(f"        vs {FEE:.2f}% round-trip taker fee: {cost}")
        print(f"        exactly-zero raw returns: {zeros}/{len(pairs)} "
              f"({zeros/len(pairs):.0%})")


def session_level(paths):
    """Session-level check: does a run's median agg track that run's drift?

    Reported with heavy caveats. This is the SAME between-group comparison that
    was retracted in PR #4 when it rested on two sessions, and the sessions are
    not independent -- four are POPCAT recorded hours apart, two are TRX, so the
    effective sample is closer to 2-3 regimes than to n sessions.
    """
    rows = []
    for p in paths:
        cs = load(Path(p))
        aggs = [c["agg"] for c in cs if c["agg"] is not None]
        if len(cs) < 2 or not aggs:
            continue
        drift = (cs[-1]["close"] / cs[0]["close"] - 1) * 100
        rows.append((p, st.median(aggs), drift, len(cs)))

    print(f"\n{'=' * 68}\nSESSION-LEVEL (between-run) -- interpret with care\n{'=' * 68}")
    print(f"  {'run':<22} {'n':>5} {'agg median':>11} {'drift %':>9}")
    for p, a, d, n in sorted(rows, key=lambda r: r[1]):
        print(f"  {p:<22} {n:>5} {a:>11.3f} {d:>+9.2f}")

    if len(rows) >= 4:
        rho = spearman([r[1] for r in rows], [r[2] for r in rows])
        n = len(rows)
        print(f"\n  spearman(median agg, drift) = {rho:+.3f}  over n={n} sessions")
        # exact-ish critical values for small-n two-sided Spearman at p=0.05
        crit = {4: 1.000, 5: 0.900, 6: 0.886, 7: 0.786, 8: 0.738}
        c = crit.get(n)
        if c:
            verdict = "at/beyond the p=0.05 critical value" if abs(rho) >= c \
                      else "below the p=0.05 critical value"
            print(f"  critical |rho| for n={n} at p=0.05 is {c:.3f} -> {verdict}")
        print("  CAVEAT: sessions of the same symbol recorded hours apart share a")
        print("  regime, so they are not independent draws and the nominal p is")
        print("  optimistic. A session-level pattern with no within-session")
        print("  counterpart is more likely regime coincidence than mechanism.")


def main():
    print("PRE-REGISTERED HYPOTHESIS (stated in PR #4, before runs 3-4 existed):")
    print(f'  "agg > {CUTOFF} is followed by underperformance"')
    print("  Direction predicted: NEGATIVE difference vs the rest.")

    report("IN-SAMPLE  (runs 1-2, where the hypothesis was generated)", IN_SAMPLE)
    report("HOLDOUT 1  (runs 3-4, tested in PR #5)", OUT_SAMPLE)
    report("HOLDOUT 2  (runs 5-6, NEW - never seen before this test)", OUT_SAMPLE_2)
    report("ALL HOLDOUT DATA POOLED (runs 3-6, hypothesis never fitted here)",
           OUT_SAMPLE + OUT_SAMPLE_2)
    report("EVERYTHING POOLED", IN_SAMPLE + OUT_SAMPLE + OUT_SAMPLE_2)

    session_level(IN_SAMPLE + OUT_SAMPLE + OUT_SAMPLE_2)

    print(f"\n{'=' * 68}\nper-run detail\n{'=' * 68}")
    for p in IN_SAMPLE + OUT_SAMPLE + OUT_SAMPLE_2:
        cs = load(Path(p))
        aggs = [c["agg"] for c in cs if c["agg"] is not None]
        rets = [fwd(cs, i, 1) for i in range(len(cs))]
        rets = [r for r in rets if r is not None]
        zeros = sum(1 for r in rets if abs(r) < 1e-12)
        drift = (cs[-1]["close"] / cs[0]["close"] - 1) * 100 if cs else 0
        tag = ("in-sample " if p in IN_SAMPLE
               else "holdout-1 " if p in OUT_SAMPLE else "HOLDOUT-2 ")
        print(f"  {tag} {p:<20} n={len(cs):<4} drift={drift:+6.2f}%  "
              f"agg med={st.median(aggs):.3f}  "
              f"zero-return candles={zeros/max(len(rets),1):.0%}")


if __name__ == "__main__":
    sys.exit(main())
