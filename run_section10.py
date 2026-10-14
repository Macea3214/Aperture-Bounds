"""
Section 10.1 and 10.2 checks: direct primal optimisation versus spectral dual.

Settings that reproduce the manuscript's figures (found by matching, see README):
  hbar = m = 1, R = 0.8, P_E^2 = 2, A0 = 1.25, sigma = 0.12, oscillator length 1
  Section 10.1: N in {10,14,18,22,26}, t in {0.05,0.20,0.50,0.90}
  Section 10.2: N = 24, t = 0.3, one parameter swept at a time.

Run:  python run_section10.py
Writes results_10_1.csv and results_10_2.csv.
"""
import csv
import numpy as np
from aperture_bounds import build_truncated, dual_truncated, primal_truncated

R2_BASE, PE2_BASE = 0.64, 2.0
A0_BASE, SIG_BASE = 1.25, 0.12


def compare(N, t, A0, sig, R2, PE2, n_starts=30):
    Qt, X2, P2 = build_truncated(N, t, A0, sig)
    Ld, mult = dual_truncated(Qt, X2, P2, R2, PE2)
    Lp, mx, mp = primal_truncated(Qt, X2, P2, R2, PE2, n_starts=n_starts)
    return dict(dual=Ld, primal=Lp, abs_diff=abs(Lp - Ld), rel_diff=abs(Lp - Ld) / max(abs(Ld), 1e-300),
                moment_x2=mx, moment_p2=mp, lam=mult[0], mu=mult[1])


def section_10_1():
    rows = []
    for t in (0.05, 0.20, 0.50, 0.90):
        for N in (10, 14, 18, 22, 26):
            r = compare(N, t, A0_BASE, SIG_BASE, R2_BASE, PE2_BASE)
            rows.append(dict(N=N, t=t, **r))
            print("10.1  N=%2d t=%.2f  L=%.12f  |primal-dual|=%.2e" % (N, t, r["dual"], r["abs_diff"]), flush=True)
    with open("results_10_1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def section_10_2():
    N, T = 24, 0.3
    tests = []
    for t in (0.0, 0.0005, 0.001, 0.005, 0.01, 0.05):
        tests.append(("t->0", dict(t=t)))
    for A0 in (1, 1.5, 2, 3, 5, 8):
        tests.append(("A0->inf", dict(A0=A0)))
    for A0 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        tests.append(("A0->0", dict(A0=A0)))
    for s in (0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3):
        tests.append(("sigma->0", dict(sig=s)))
    for ratio in (1.001, 1.005, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0):     # R^2 P_E^2 / (hbar^2/4)
        tests.append(("Heisenberg", dict(PE2=ratio * 0.25 / R2_BASE)))
    rows = []
    for name, kw in tests:
        p = dict(t=T, A0=A0_BASE, sig=SIG_BASE, R2=R2_BASE, PE2=PE2_BASE); p.update(kw)
        r = compare(N, p["t"], p["A0"], p["sig"], p["R2"], p["PE2"])
        rows.append(dict(regime=name, **{k: v for k, v in kw.items()}, **r))
        print("10.2  %-10s %s  L=%.10f  |primal-dual|=%.2e  lam=%.3g mu=%.3g" %
              (name, kw, r["dual"], r["abs_diff"], r["lam"], r["mu"]), flush=True)
    keys = sorted({k for r in rows for k in r.keys()}, key=lambda k: (k not in ("regime",), k))
    with open("results_10_2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    a = section_10_1()
    b = section_10_2()
    print("\nSection 10.1: %d comparisons, max abs diff %.2e, max rel diff %.2e" %
          (len(a), max(r["abs_diff"] for r in a), max(r["rel_diff"] for r in a)))
    print("Section 10.2: %d tests, max abs diff %.2e" % (len(b), max(r["abs_diff"] for r in b)))
    n26 = [r for r in a if r["N"] == 26]
    print("N=26 moment saturation: max |<X^2>-R^2| = %.2e, max |<P^2>-P_E^2| = %.2e" %
          (max(abs(r["moment_x2"] - R2_BASE) for r in n26), max(abs(r["moment_p2"] - PE2_BASE) for r in n26)))
