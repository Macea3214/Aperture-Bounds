"""
Numerical checks for "Energy Constrained Worst Case Aperture Bounds".

Everything here is written from the definitions in the manuscript (hbar = m = 1 unless stated).
Nothing was fitted to the numbers reported in the paper.

Contents
  1. Harmonic oscillator truncation: direct primal optimization versus the spectral dual.
  2. Continuum dual: finite difference ground state of  B P^2 + D X^2 - Q(X)  (Theorem 1 and Section 6).
  3. Section 11 benchmark with lithium 6 scales.
"""
import numpy as np
from scipy.special import erf
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.optimize import minimize

HBAR = 1.0
MASS = 1.0


# ---------------------------------------------------------------- detector profile
def click_profile(x, A0, sigma):
    """a_sigma(x) from Section 3; sharp aperture when sigma == 0."""
    x = np.asarray(x, dtype=float)
    if sigma == 0:
        return (np.abs(x) <= A0).astype(float)
    s = np.sqrt(2.0) * sigma
    return 0.5 * (erf((A0 - x) / s) - erf((-A0 - x) / s))


# ---------------------------------------------------------------- 1. truncated problem
def ho_operators(N, L=1.0):
    """Truncated position and momentum matrices from the oscillator ladder operators (length scale L)."""
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    X = L * (a + a.T) / np.sqrt(2.0)
    P = 1j * HBAR / (L * np.sqrt(2.0)) * (a.T - a)
    return X, P


def build_truncated(N, t, A0, sigma, L=1.0, squares="matrix"):
    """
    Returns Qt, X2, P2 in the N dimensional truncation.
    Q = q(X_N) is a matrix function of the truncated position operator, Qt = U^dagger Q U with
    U = exp(-i t P_N^2 / (2 m hbar)).
    squares = "matrix": X2 = X_N @ X_N (square of the truncated operator)
    squares = "exact" : X2 = truncation of the exact operator X^2 (computed in a larger space)
    """
    X, P = ho_operators(N, L)
    w, V = eigh(X)
    Q = (V * (1.0 - click_profile(w, A0, sigma))) @ V.T
    P2m = P @ P
    wp, Vp = eigh(P2m)
    U = (Vp * np.exp(-1j * t * wp / (2 * MASS * HBAR))) @ Vp.conj().T
    Qt = U.conj().T @ Q @ U
    if squares == "matrix":
        X2, P2 = X @ X, P2m
    else:
        Xb, Pb = ho_operators(N + 2, L)
        X2, P2 = (Xb @ Xb)[:N, :N], (Pb @ Pb)[:N, :N]
    return Qt, X2.astype(complex), P2


def dual_truncated(Qt, X2, P2, R2, PE2, starts=None):
    """min over lam, mu >= 0 of lam R2 + mu PE2 + lambda_max(Qt - lam X2 - mu P2)."""
    N = Qt.shape[0]

    def f(z):
        lam, mu = z
        H = Qt - lam * X2 - mu * P2
        w, v = eigh(H, subset_by_index=[N - 1, N - 1])
        v = v[:, 0]
        val = lam * R2 + mu * PE2 + w[0]
        gl = R2 - np.real(v.conj() @ X2 @ v)
        gm = PE2 - np.real(v.conj() @ P2 @ v)
        return val, np.array([gl, gm])

    if starts is None:
        starts = [(0.5, 0.5), (5.0, 1.0), (1.0, 5.0), (20.0, 20.0), (0.05, 0.05)]
    best = None
    for s in starts:
        r = minimize(f, s, jac=True, method="L-BFGS-B", bounds=[(0, None), (0, None)],
                     options=dict(ftol=1e-15, gtol=1e-12, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x


def primal_truncated(Qt, X2, P2, R2, PE2, n_starts=30, seed=0):
    """
    Direct maximisation of <psi|Qt|psi> over normalised complex vectors with <X^2> <= R2, <P^2> <= PE2.
    Does not use the dual solution.  Restricting to pure states is justified by the extreme point
    argument (two linear constraints plus normalisation).
    """
    N = Qt.shape[0]
    rng = np.random.default_rng(seed)

    def unpack(z):
        return z[:N] + 1j * z[N:]

    def quad(A, z):
        psi = unpack(z)
        n = np.real(psi.conj() @ psi)
        Ap = A @ psi
        g = np.real(psi.conj() @ Ap) / n
        grad = np.concatenate([2 * np.real(Ap) / n - 2 * g * psi.real / n,
                               2 * np.imag(Ap) / n - 2 * g * psi.imag / n])
        return g, grad

    obj = lambda z: tuple(-u for u in quad(Qt, z))
    cons = [
        dict(type="ineq", fun=lambda z: R2 - quad(X2, z)[0], jac=lambda z: -quad(X2, z)[1]),
        dict(type="ineq", fun=lambda z: PE2 - quad(P2, z)[0], jac=lambda z: -quad(P2, z)[1]),
    ]
    best, best_z = -np.inf, None
    for k in range(n_starts):
        scale = np.exp(-0.3 * np.arange(N) * rng.uniform(0.2, 1.5))
        z0 = np.concatenate([rng.normal(size=N) * scale, rng.normal(size=N) * scale])
        r = minimize(obj, z0, jac=True, constraints=cons, method="SLSQP",
                     options=dict(ftol=1e-16, maxiter=1000))
        if not r.success and r.status not in (0, 8, 9):
            continue
        # accept only feasible points
        if quad(X2, r.x)[0] <= R2 + 1e-9 and quad(P2, r.x)[0] <= PE2 + 1e-9 and -r.fun > best:
            best, best_z = -r.fun, r.x
    mx, mp = quad(X2, best_z)[0], quad(P2, best_z)[0]
    return best, mx, mp


# ---------------------------------------------------------------- 2. continuum dual
def ground_energy(B, D, qfun, xmax, nx, hbar=1.0):
    """Lowest eigenvalue of B P^2 + D x^2 - q(x) by finite differences on [-xmax, xmax]."""
    x = np.linspace(-xmax, xmax, nx)
    dx = x[1] - x[0]
    diag = 2 * B * hbar ** 2 / dx ** 2 + D * x ** 2 - qfun(x)
    off = -B * hbar ** 2 / dx ** 2 * np.ones(nx - 1)
    w = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0), eigvals_only=True)
    return w[0]


def continuum_dual(R2, PE2, t, qfun, xmax, nx, hbar=1.0, m=1.0, starts=None):
    """
    L_wc(t) = min_{lam,mu>0} [ lam R2 + mu PE2 - E0 ],  E0 = inf spec(B P^2 + D X^2 - Q(X))
    with tau = t/m, B = mu + lam tau^2, D = lam mu / B   (Section 6).
    """
    tau = t / m

    def f(z):
        lam, mu = np.exp(z)
        B = mu + lam * tau ** 2
        D = lam * mu / B
        return lam * R2 + mu * PE2 - ground_energy(B, D, qfun, xmax, nx, hbar)

    if starts is None:
        starts = [(0.0, 0.0), (2.0, 1.0), (-1.0, 2.0), (3.0, -1.0)]
    best = None
    for s in starts:
        r = minimize(f, s, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-13, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, np.exp(best.x)


# ---------------------------------------------------------------- 3. lithium 6 scales (Section 11)
HBAR_SI = 1.054571817e-34
AMU = 1.66053906660e-27
M_LI6 = 6.0151228874 * AMU


def lithium_moments(freq_hz, nbar):
    w = 2 * np.pi * freq_hz
    R2 = (2 * nbar + 1) * HBAR_SI / (2 * M_LI6 * w)
    PE2 = (2 * nbar + 1) * M_LI6 * HBAR_SI * w / 2
    return R2, PE2


def lithium_benchmark(freq_hz=760e3, nbar=0.46, t_s=12.5e-6, A0_um=5.0, sigma_um=0.0,
                      xmax_um=40.0, nx=80001, starts=None):
    R2, PE2 = lithium_moments(freq_hz, nbar)
    ell = 1e-6                               # length unit: micrometre
    tunit = M_LI6 * ell ** 2 / HBAR_SI      # time unit so that hbar = m = 1
    R2n, PE2n, tn = R2 / ell ** 2, PE2 * ell ** 2 / HBAR_SI ** 2, t_s / tunit
    qfun = lambda x: 1.0 - click_profile(x, A0_um, sigma_um)
    Lwc, mult = continuum_dual(R2n, PE2n, tn, qfun, xmax_um, nx, starts=starts)
    width = np.sqrt(R2n + tn ** 2 * PE2n)                       # Gaussian family RMS width (um)
    gauss_eff = float(erf(A0_um / (np.sqrt(2) * width))) if sigma_um == 0 else None
    cheb = 1 - ((np.sqrt(R2n) + tn * np.sqrt(PE2n)) / A0_um) ** 2
    return dict(width_um=width, gauss_eff=gauss_eff, chebyshev=cheb, eta_wc=1 - Lwc, multipliers=mult,
                R_um=np.sqrt(R2n), tauPE_um=tn * np.sqrt(PE2n))
