"""
Regenerates the figures used in the manuscript, in black and white, from the result files written by the run_*.py
scripts.  Run those first, then:  python make_figures.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.prop_cycle": plt.cycler(color=["black"]),
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


# ------------------------------------------------------------ convergence of the truncated value
def fig_truncation():
    rows = read("results_convergence.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    marks = {0.05: "o", 0.2: "s", 0.5: "^", 0.9: "D"}
    for t, m in marks.items():
        sub = [r for r in rows if abs(float(r["t"]) - t) < 1e-9]
        N = [int(r["N"]) for r in sub]
        y = [float(r["truncated"]) for r in sub]
        ax.plot(N, y, marker=m, ms=4, lw=0.8, mfc="white", label="t = %.2f" % t)
        ax.axhline(float(sub[0]["continuum"]), ls=(0, (4, 3)), lw=0.7, color="0.35")
    ax.set_xlabel("Harmonic oscillator truncation dimension N")
    ax.set_ylabel("Worst case no click probability")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    ax.text(121, 0.17, "dashed: continuum dual", fontsize=7.5, ha="right", va="bottom", color="0.25")
    fig.tight_layout(); fig.savefig("fig_truncation.png"); plt.close(fig)


# ------------------------------------------------------------ limiting cases
def fig_limits():
    rows = read("results_10_2.csv")
    R2 = 0.64
    panels = [
        ("t->0", "t", lambda r: num(r, "t"), "Vanishing propagation time", "t"),
        ("A0->inf", "A0", lambda r: num(r, "A0"), "Increasing aperture half width", "A0"),
        ("A0->0", "A0", lambda r: num(r, "A0"), "Shrinking aperture half width", "A0"),
        ("sigma->0", "sig", lambda r: num(r, "sig"), "Vanishing detector blur", "sigma"),
        ("Heisenberg", "PE2", lambda r: num(r, "PE2") * R2 / 0.25, "Approach to the Heisenberg boundary",
         "R^2 P_E^2 / (hbar^2/4)"),
    ]
    fig = plt.figure(figsize=(6.4, 7.0))
    gs = fig.add_gridspec(3, 4)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 0:2]),
            fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[2, 1:3])]
    for ax, (reg, key, xf, title, xlabel) in zip(axes, panels):
        sub = [r for r in rows if r["regime"] == reg]
        xs = np.array([xf(r) for r in sub]); P = np.array([float(r["primal"]) for r in sub])
        D = np.array([float(r["dual"]) for r in sub])
        ax.plot(xs, D, "-", lw=0.7, color="0.4")
        ax.plot(xs, P, "o", ms=5, mfc="black", label="direct primal")
        ax.plot(xs, D, "x", ms=6, mew=0.9, color="black", label="spectral dual")
        ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel); ax.set_ylabel("Worst case no click probability")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("fig_limits.png"); plt.close(fig)


# ------------------------------------------------------------ allocation ensemble
def fig_allocation():
    rows = read("results_allocation.csv")
    data = [[float(r[k]) for r in rows] for k in ("blurred_spectral", "sharp_spectral", "equal")]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="black", lw=1.2),
                    boxprops=dict(facecolor="0.85", edgecolor="black"), whiskerprops=dict(color="black"),
                    capprops=dict(color="black"), flierprops=dict(marker="o", mfc="white", mec="black", ms=4))
    ax.set_xticklabels(["blurred\nspectral", "sharp\nspectral", "equal\nshots"])
    ax.axhline(1.0, lw=0.6, color="0.4", ls=(0, (4, 3)))
    ax.set_ylabel("Worst band RMSE divided by oracle RMSE")
    fig.tight_layout(); fig.savefig("fig_allocation.png"); plt.close(fig)


def fig_penalty():
    rows = read("results_allocation.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, r in enumerate(rows):
        name = r["state"]
        style = dict(marker="D", mfc="black") if i < 5 else (dict(marker="o", mfc="white") if "pure" in name
                                                              else dict(marker="s", mfc="0.6"))
        ax.plot(i, float(r["penalty_spectral_alloc"]), ls="", mec="black", ms=5, **style)
    ax.axhline(1.0, lw=0.6, color="0.4", ls=(0, (4, 3)))
    ax.plot([], [], "D", mfc="black", mec="black", ms=5, label="named states")
    ax.plot([], [], "o", mfc="white", mec="black", ms=5, label="random pure")
    ax.plot([], [], "s", mfc="0.6", mec="black", ms=5, label="random mixed")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
    ax.set_ylim(0.95, 1.5)
    ax.set_xlabel("State index"); ax.set_ylabel("MSE ratio, free propagation to ideal")
    fig.tight_layout(); fig.savefig("fig_penalty.png"); plt.close(fig)


# ------------------------------------------------------------ lithium 6
def fig_lithium():
    rows = read("results_lithium.csv")
    A = [float(r["A0_um"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(A, [float(r["gaussian"]) for r in rows], ":", marker="s", ms=4, mfc="white", color="black",
            label="Gaussian state efficiency")
    ax.plot(A, [float(r["spectral"]) for r in rows], "-", marker="o", ms=4, color="black", label="spectral guarantee")
    ax.plot(A, [float(r["chebyshev"]) for r in rows], "--", marker="x", ms=5, color="black",
            label="second moment Chebyshev")
    ax.set_xlabel("One dimensional acceptance half width A (micrometres)")
    ax.set_ylabel("Click efficiency"); ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig("fig_lithium.png"); plt.close(fig)


if __name__ == "__main__":
    fig_truncation(); fig_limits(); fig_allocation(); fig_lithium()
    print("figures written")
