"""
Section 11 benchmark with lithium 6 scales (760 kHz preparation, mean occupation 0.46, 12.5 microseconds).

For each acceptance half width A0 this prints the Gaussian family efficiency, the generic second moment
(Chebyshev) guarantee built from the same R and P_E, and the spectral worst case efficiency
eta_wc = 1 - L_wc from the continuum dual (Theorem 1 and Section 6), for a sharp aperture.

Run:  python run_lithium.py      (writes results_lithium.csv)
"""
import csv
from aperture_bounds import lithium_benchmark

rows = []
for A0 in (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.6):
    r = lithium_benchmark(A0_um=A0, sigma_um=0.0, xmax_um=40.0, nx=20001, starts=[(0.0, 0.0), (1.0, -1.0)])
    rows.append(dict(A0_um=A0, width_um=r["width_um"], gaussian=r["gauss_eff"], chebyshev=r["chebyshev"],
                     spectral=r["eta_wc"], R_um=r["R_um"], tauPE_um=r["tauPE_um"]))
    print("A0 = %5.1f um   Gaussian family %.3f   Chebyshev %.3f   spectral worst case %.3f   (width %.3f um)" %
          (A0, r["gauss_eff"], r["chebyshev"], r["eta_wc"], r["width_um"]), flush=True)

with open("results_lithium.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
