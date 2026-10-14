# Aperture bounds: numerical checks

https://doi.org/10.5281/zenodo.23381240

Code for the numerical results in "Energy Constrained Worst Case Aperture Bounds for Freely Propagated Continuous
Variable Quantum States" (Jason Bleggi, independent researcher, 2026).

Requirements: Python 3.10 or newer with numpy, scipy and matplotlib (`pip install -r requirements.txt`).
Tested with numpy 2.4, scipy 1.17 and matplotlib 3.10. The scripts use one CPU core and take a few minutes each.

Note: this code was written with the assistance of an LLM. The author is responsible for its content.

Written from the definitions in the manuscript; nothing was fitted to the
reported numbers. Three settings that the manuscript does not state were inferred by matching its figures and are
documented here: oscillator length 1 (hbar = m = omega = 1) for the truncation, N = 24 and t = 0.3 for the
Section 10.2 sweeps.

## Files
- aperture_bounds.py   core routines: truncated primal and dual, continuum dual, lithium 6 scales
- run_section10.py     Section 10.1 and 10.2: direct primal optimisation versus spectral dual (about 2 minutes)
- run_convergence.py   truncation convergence against the continuum dual (about 1 minute)
- run_lithium.py       Section 11 benchmark (about 3 minutes)
- run_allocation.py    Sections 10.3 and 10.4 (about 3 minutes)
- make_figures.py      regenerates the paper's figures (black and white) from the CSV files
- results_*.csv, log_*.txt   outputs from the runs described below

Run:  python run_section10.py ; python run_convergence.py ; python run_lithium.py ; python run_allocation.py ; python make_figures.py

## Method
- Truncated problem: X_N, P_N from the ladder operators, Q = q(X_N) as a matrix function, Q_t = U^dagger Q U with
  U = exp(-i t P_N^2 / 2). Primal: maximise <psi|Q_t|psi> over normalised complex psi with <X^2> <= R^2,
  <P^2> <= P_E^2 (SLSQP, 30 random starts, not using the dual). Dual: minimise
  lam R^2 + mu P_E^2 + lambda_max(Q_t - lam X^2 - mu P^2) over lam, mu >= 0.
- Continuum dual: minimise lam R^2 + mu P_E^2 - E0 with E0 the ground energy of B P^2 + D X^2 - Q(X),
  B = mu + lam tau^2, D = lam mu / B, tau = t/m (finite differences, Section 6 of the manuscript).

## What reproduces
| Manuscript item | Result of this code |
|---|---|
| Fig. 10.1 worst case values, N = 10..26, four times | match the plotted values to plotting accuracy (about 0.001) |
| Primal versus dual, 20 comparisons | max abs difference 5.1e-12 (manuscript: 2.9e-13); 18 of 20 below 1e-12 |
| Moment saturation at N = 26 | within 3.5e-12 |
| Section 10.2 panels (33 tests) | all five panels match; max abs difference 2.3e-9 (manuscript: 2.3e-12); 28 of 33 below 1e-11 |
| Section 11 lithium 6, A0 = 3 and 5 um | spectral 0.609 and 0.861 (match); Gaussian 0.725 and 0.931 (match) |
| Section 11 Chebyshev guarantee | 0.131 and 0.687 (manuscript: 0.135 and 0.689) |
| Section 11 A0 = 25.6 um | spectral 0.992 (above 99 percent, as stated) |

## Sections 10.3 and 10.4 (rewritten)
The original ensemble and simulation could not be reproduced because the manuscript did not specify them, and the
parameters were not tuned to recover the old numbers (5.58 percent, 16.53 percent, 18.23 percent, 36.35 percent).
run_allocation.py defines the experiment explicitly: budgets R^2 = 0.64, P_E^2 = 2, blurred aperture A0 = 1.25,
sigma = 0.12, four times 0.1, 0.2, 0.4, 0.8, and 29 seeded admissible states. Results:
- Guaranteed efficiencies 0.773, 0.709, 0.567, 0.347. Guaranteed worst band variance 7.35 (spectral) against 11.53
  (equal), in units of C/N_tot. The four extremal states attain the guarantee in their own bands (within 1e-3).
- On the 29 state ensemble the spectral rule is on average 13.8 percent above the oracle and equal allocation is
  8.6 percent above. The old claim that equal allocation is worse for typical states does not hold; the spectral rule
  is the minimax rule, not the best rule for typical states.
- Free propagation versus an ideal adjustable apparatus: the MSE ratio F(rho) is at least 1 (median 1.13, max 1.34).

make_figures.py regenerates all figures (black and white) from the CSV files.

## New finding (run_convergence.py)
The truncated values at N = 10 to 26 are not close to the continuum value. They oscillate around it
(for example t = 0.5: continuum 0.49814, truncated 0.4385, 0.5372, 0.5134, 0.4762, 0.4898 for N = 10, 14, 18, 22, 26)
and converge only slowly (N = 120: error about 5e-5). The large N truncated dual agrees with the continuum dual,
which numerically supports the reduction of Section 6.
