# Aperture bounds: numerical checks

Code for the numerical results and figures in "Energy Constrained Worst Case Aperture Bounds for Freely Propagated
Continuous Variable Quantum States" (Jason Bleggi, independent researcher, 2026).

Note: this code was written with the assistance of an LLM. The author is responsible for its content.

## Requirements
Python 3.10 or newer with numpy, scipy and matplotlib (`pip install -r requirements.txt`). Tested with numpy 2.4,
scipy 1.17 and matplotlib 3.10. Every script runs on one CPU core. Random choices are seeded, so the results are
reproducible.

## Files and what they produce
| Script | Paper section | Output | Time |
|---|---|---|---|
| `run_section10.py` | 10.1 and 10.2 | `results_10_1.csv`, `results_10_2.csv` | about 2 minutes |
| `run_convergence.py` | 10.1 | `results_convergence.csv` | about 1 minute |
| `run_allocation.py` | 10.3 | `results_allocation.csv`, `results_extremal.csv` | about 3 minutes |
| `run_lithium.py` | 11 | `results_lithium.csv` | about 3 minutes |
| `make_figures.py` | 10 and 11 | `fig_*.png` (black and white) | seconds |

`aperture_bounds.py` holds the shared routines. The `log_*.txt` files are the console output of the runs that produced
the CSV files. Run the four `run_*.py` scripts first, then `make_figures.py`.

## Method
- Truncated problem: the position and momentum matrices X_N and P_N come from the harmonic oscillator ladder operators
  (oscillator length 1, hbar = m = 1). The no click effect is the matrix function Q = q(X_N), and
  Q_t = U^dagger Q U with U = exp(-i t P_N^2 / 2).
- Primal: maximize <psi|Q_t|psi> over normalized complex vectors with <X^2> <= R^2 and <P^2> <= P_E^2, using sequential
  least squares programming from 30 random starts. The dual solution is not used.
- Dual: minimize lam R^2 + mu P_E^2 + lambda_max(Q_t - lam X^2 - mu P^2) over lam, mu >= 0 with a bounded
  quasi Newton method from several starting points.
- Continuum dual: minimize lam R^2 + mu P_E^2 - E0 over lam, mu > 0, where E0 is the ground energy of
  B P^2 + D X^2 - Q(X) computed by finite differences, with B = mu + lam tau^2, D = lam mu / B, tau = t/m.
- Base parameters: R^2 = 0.64, P_E^2 = 2, a blurred aperture with A0 = 1.25 and sigma = 0.12.

## Results
Section 10.1 (N = 10, 14, 18, 22, 26 and t = 0.05, 0.20, 0.50, 0.90):
- Direct primal versus dual: 20 comparisons, largest absolute difference 5.11e-12 and largest relative difference
  3.02e-11. Both moment constraints are saturated to within 3.5e-12 at N = 26.
- Convergence to the continuum dual (values 0.2049, 0.2915, 0.4981, 0.6925 for the four times): the truncated values
  oscillate about them, with errors up to 6.5e-2 at N = 10, 8.3e-3 at N = 26 and 5.6e-5 at N = 120.

Section 10.2 (N = 24, t = 0.3, one parameter varied at a time): 33 tests, largest primal versus dual discrepancy
2.26e-9, and 28 of the 33 agree to better than 1e-11. The largest discrepancies occur where the loss is very small or the
problem is near the Heisenberg boundary.

Section 10.3 (four times t = 0.1, 0.2, 0.4, 0.8 and 29 seeded admissible states):
- Guaranteed efficiencies 0.773, 0.709, 0.567, 0.347. Guaranteed worst band variance 7.35 for the spectral allocation
  and 11.53 for equal allocation, in units of C/N_tot.
- The four extremal states attain the guarantee in their own bands to within 1e-3.
- Mean worst band RMSE excess over the state informed oracle: 13.8 percent (blurred spectral), 13.7 percent (sharp
  spectral), 8.6 percent (equal). Cost of free propagation versus an ideal adjustable apparatus: median 1.13,
  minimum 1.04, maximum 1.34.

Section 11 (lithium 6, 760 kHz, 12.5 microseconds, sharp aperture):

| A0 (micrometres) | Gaussian state | Chebyshev | Spectral guarantee |
|---|---|---|---|
| 3 | 0.725 | 0.131 | 0.609 |
| 4 | 0.854 | 0.511 | 0.780 |
| 5 | 0.931 | 0.687 | 0.861 |
| 7 | 0.989 | 0.840 | 0.926 |
| 10 | 1.000 | 0.922 | 0.959 |
| 15 | 1.000 | 0.965 | 0.980 |
| 25.6 | 1.000 | 0.988 | 0.992 |

## Limitations
The truncated values depend on N and are approximate. The continuum values use finite difference grids whose sizes are
set in the scripts. The 760 kHz and 12.5 microsecond pairing is a hypothetical benchmark built from reported scales and
is not an experimental data point.

## Citing
Please cite the Zenodo archive (DOI badge on the repository page) and the paper. `CITATION.cff` gives the citation
details.

## License
MIT License, see `LICENSE`.

## Changes
- 1.0.1: README rewritten to match the final paper; docstring and comment corrections; added `log_allocation.txt`.
  No change to any result.
- 1.0.0: first release.
