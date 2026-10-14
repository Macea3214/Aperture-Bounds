"""
Sections 10.3 and 10.4, written as fully specified experiments (hbar = m = 1).

Model (Section 9 of the manuscript): at propagation time t_r the variance of the estimator on band r is
    V_r = C / (N_r * eta_r(rho)),      eta_r(rho) = Tr(a_sigma(X_t) rho)  (throughput of the actual state),
where N_r is the number of trials spent on band r.  The worst band variance is max_r V_r.

Choices made here (none of them tuned; the manuscript does not state them):
  moment budgets   R^2 = 0.64, P_E^2 = 2           (the Section 10 base case)
  detector         Gaussian blurred, A0 = 1.25, sigma = 0.12
  four scales      t_r = 0.1, 0.2, 0.4, 0.8        (geometric, ratio 2)
  states           29 admissible states: 5 named, 12 random pure, 12 random mixed (seeded)
Allocation rules compared (all with the same N_tot):
  oracle            N_r proportional to 1 / eta_r(rho)   (uses the state; minimises the worst band variance)
  blurred spectral  N_r proportional to 1 / eta_wc(t_r)  for the blurred detector, state independent
  sharp spectral    N_r proportional to 1 / eta_wc(t_r)  computed for a sharp aperture of the same A0
  equal             N_r = N_tot / 4

Run:  python run_allocation.py   (writes results_allocation.csv)
"""
import csv
import numpy as np
from aperture_bounds import click_profile, continuum_dual

R2, PE2 = 0.64, 2.0
A0, SIG = 1.25, 0.12
TIMES = np.array([0.1, 0.2, 0.4, 0.8])
NX, XMAX = 8192, 12.0
x = np.linspace(-XMAX, XMAX, NX, endpoint=False)
dx = x[1] - x[0]
p = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
A_BLUR = click_profile(x, A0, SIG)


# ---------------------------------------------------------------- states
def normalise(psi):
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def gauss(center, s2, chirp=0.0, phase=0.0):
    return np.exp(-(x - center) ** 2 / (4 * s2) + 1j * (chirp * (x - center) ** 2 / 2 + phase))


def moments(psi):
    m2x = np.sum(x ** 2 * np.abs(psi) ** 2) * dx
    ph = np.fft.fft(psi)
    m2p = np.sum(p ** 2 * np.abs(ph) ** 2) / np.sum(np.abs(ph) ** 2)
    return float(m2x), float(m2p)


def hermite_functions(nmax, L):
    phi = np.zeros((nmax, NX))
    phi[0] = (np.pi * L ** 2) ** -0.25 * np.exp(-x ** 2 / (2 * L ** 2))
    if nmax > 1:
        phi[1] = np.sqrt(2.0) * (x / L) * phi[0]
    for n in range(1, nmax - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * (x / L) * phi[n] - np.sqrt(n / (n + 1.0)) * phi[n - 1]
    return phi


def admissible(psi):
    mx, mp = moments(psi)
    return mx <= R2 and mp <= PE2


def build_states(seed=2026):
    rng = np.random.default_rng(seed)
    named = {
        "gaussian": gauss(0.0, 0.25),
        "chirped": gauss(0.0, 0.25, chirp=1.8),
        "cat like": gauss(0.5, 0.16) + gauss(-0.5, 0.16),
        "multimodal": gauss(-0.55, 0.14) + gauss(0.0, 0.14, phase=np.pi / 2) + gauss(0.55, 0.14),
        "edge biased": gauss(0.7, 0.13),
    }
    pure = [(k, [(1.0, normalise(v))]) for k, v in named.items()]
    phi = hermite_functions(14, 0.8)
    pool = []
    while len(pool) < 40:
        nu = rng.uniform(0.8, 2.5)
        c = (rng.normal(size=14) + 1j * rng.normal(size=14)) * np.exp(-np.arange(14) / nu)
        psi = normalise(c @ phi)
        if admissible(psi):
            pool.append(psi)
    states = list(pure)
    for i in range(12):
        states.append(("random pure %d" % (i + 1), [(1.0, pool[i])]))
    for i in range(12):
        k = int(rng.integers(2, 5))
        idx = rng.choice(len(pool), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        states.append(("random mixed %d" % (i + 1), [(float(wi), pool[j]) for wi, j in zip(w, idx)]))
    return states


def state_moments(comps):
    mx = sum(w * moments(psi)[0] for w, psi in comps)
    mp = sum(w * moments(psi)[1] for w, psi in comps)
    return mx, mp


def throughput(comps, t, profile):
    eta = 0.0
    for w, psi in comps:
        pt = np.fft.ifft(np.exp(-1j * t * p ** 2 / 2) * np.fft.fft(psi))
        eta += w * np.sum(profile * np.abs(pt) ** 2) * dx
    return eta


# ---------------------------------------------------------------- spectral guarantees
def eta_wc(sigma):
    out = []
    for t in TIMES:
        L, _ = continuum_dual(R2, PE2, t, lambda y: 1.0 - click_profile(y, A0, sigma), xmax=12.0, nx=16001, starts=[(0.0, 0.0), (1.0, -1.0)])
        out.append(1.0 - L)
    return np.array(out)


def worst_band_variance(weights, eta):
    """worst band variance (units of C / N_tot) for allocation N_r = N_tot * weights / sum(weights)."""
    w = np.asarray(weights, float)
    return np.max(w.sum() / (w * eta))


def extremizers(N=80):
    """
    The worst case state for each band (ground state of the reduced operator, taken as the top eigenvector of
    Qt - lam X^2 - mu P^2 at the optimal multipliers) in an N dimensional oscillator truncation, and its
    throughput in every band.  Returns a matrix eta[r_state, s_band].
    """
    from scipy.linalg import eigh
    from aperture_bounds import build_truncated, dual_truncated
    Qts = [build_truncated(N, t, A0, SIG)[0] for t in TIMES]
    out = []
    for r, t in enumerate(TIMES):
        Qt, X2, P2 = build_truncated(N, t, A0, SIG)
        Ld, (lam, mu) = dual_truncated(Qt, X2, P2, R2, PE2)
        w, v = eigh(Qt - lam * X2 - mu * P2, subset_by_index=[N - 1, N - 1])
        psi = v[:, 0]
        out.append([1.0 - np.real(psi.conj() @ Q @ psi) for Q in Qts])
    return np.array(out)


if __name__ == "__main__":
    states = build_states()
    assert len(states) == 29
    eta_blur, eta_sharp = eta_wc(SIG), eta_wc(0.0)
    print("guaranteed efficiencies  blurred:", np.round(eta_blur, 4), " sharp:", np.round(eta_sharp, 4))
    rows = []
    for name, comps in states:
        mx, mp = state_moments(comps)
        assert mx <= R2 + 1e-9 and mp <= PE2 + 1e-9, (name, mx, mp)
        eta = np.array([throughput(comps, t, A_BLUR) for t in TIMES])
        assert np.all(eta >= eta_blur - 1e-6), ("guarantee violated", name, eta, eta_blur)
        v_or = worst_band_variance(1 / eta, eta)
        r = {}
        for key, w in (("blurred_spectral", 1 / eta_blur), ("sharp_spectral", 1 / eta_sharp), ("equal", np.ones(4))):
            r[key] = float(np.sqrt(worst_band_variance(w, eta) / v_or))
        # Section 10.4: ratio of free propagation MSE to ideal (eta = 1) MSE, same allocation
        w = 1 / eta_blur; Nr = w / w.sum()
        r["penalty_spectral_alloc"] = float(np.sum(1 / (Nr * eta)) / np.sum(1 / Nr))
        r["penalty_equal_alloc"] = float(np.sum(1 / (0.25 * eta)) / np.sum(1 / 0.25 * np.ones(4)))
        rows.append(dict(state=name, moment_x2=mx, moment_p2=mp, **{"eta_%d" % (i + 1): e for i, e in enumerate(eta)}, **r))
    with open("results_allocation.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for key in ("blurred_spectral", "sharp_spectral", "equal"):
        v = np.array([r[key] for r in rows])
        print("%-17s mean excess %6.2f %%   median excess %6.2f %%   worst excess %6.2f %%" %
              (key, 100 * (v.mean() - 1), 100 * (np.median(v) - 1), 100 * (v.max() - 1)))
    for key in ("penalty_spectral_alloc", "penalty_equal_alloc"):
        v = np.array([r[key] for r in rows])
        print("%-24s median %.3f  max %.3f  min %.3f" % (key, np.median(v), v.max(), v.min()))
    ex = extremizers()
    print("\nextremal states (rows: state worst for band r; columns: throughput in band s), N=80 truncation")
    print(np.round(ex, 4))
    for key, w in (("spectral", 1 / eta_blur), ("equal", np.ones(4))):
        vv = [worst_band_variance(w, e) for e in ex]
        print("%-9s worst band variance on the four extremal states (units C/N_tot):" % key, np.round(vv, 3),
              " max %.3f" % max(vv))
    print("guaranteed worst band variance  spectral %.3f   equal %.3f   (ratio %.3f)" %
          (np.sum(1 / eta_blur), 4 / eta_blur.min(), np.sum(1 / eta_blur) / (4 / eta_blur.min())))
    with open("results_extremal.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["state_worst_for_band"] + ["throughput_band_%d" % (i + 1) for i in range(4)])
        for i, e in enumerate(ex): wr.writerow([i + 1] + list(e))
    print("min throughput margin over the guarantee (all states, all bands): %.4f" %
          min(min(r["eta_%d" % (i + 1)] - e for i, e in enumerate(eta_blur)) for r in rows))
