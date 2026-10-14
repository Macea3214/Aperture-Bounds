"""
Truncation convergence: does the harmonic oscillator truncation value approach the continuum value?

Continuum value = min over (lam, mu) of lam R^2 + mu P_E^2 - E0, with E0 the ground energy of
B P^2 + D X^2 - Q(X) computed by finite differences (Theorem 1 and Section 6 of the manuscript).
Same parameters as Section 10.1: R = 0.8, P_E^2 = 2, A0 = 1.25, sigma = 0.12.

Run:  python run_convergence.py      (writes results_convergence.csv)
"""
import csv
from aperture_bounds import build_truncated, dual_truncated, continuum_dual, click_profile

R2, PE2, A0, SIG = 0.64, 2.0, 1.25, 0.12
qfun = lambda x: 1.0 - click_profile(x, A0, SIG)
NS = [10, 14, 18, 22, 26, 30, 40, 60, 80, 100, 120]

rows = []
for t in (0.05, 0.2, 0.5, 0.9):
    Lc, mult = continuum_dual(R2, PE2, t, qfun, xmax=20.0, nx=40001)
    print("t=%.2f  continuum L_wc = %.8f   (lam=%.4f, mu=%.4f)" % (t, Lc, mult[0], mult[1]), flush=True)
    for N in NS:
        Qt, X2, P2 = build_truncated(N, t, A0, SIG)
        Ld, _ = dual_truncated(Qt, X2, P2, R2, PE2)
        rows.append(dict(t=t, N=N, truncated=Ld, continuum=Lc, difference=Ld - Lc))
        print("    N=%3d  truncated %.8f   truncated - continuum = %+.2e" % (N, Ld, Ld - Lc), flush=True)

with open("results_convergence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
