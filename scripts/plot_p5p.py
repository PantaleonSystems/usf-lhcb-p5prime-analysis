# scripts/plot_p5p.py
"""Figures for the corrected P5' analysis.

Three things changed from the published figures.

*No illustrative stand-ins.* The previous f_geo figure was drawn from a
hard-coded ``illustrative_amplitude = 0.8`` rather than from
``geometric_factor_usf``, while the caption described the shape as "fixed by
the theory". Both curves are now plotted, and the difference is the point.

*One figure, two panels.* The spectrum and the residuals were written to
separate files while the manuscript's Figure 2 caption described upper and
lower panels of one figure -- which is why the first referee reported the
figure as missing its top half.

*A non-circular projection.* The HL-LHC figure centred its projected points on
the current data, which are what the fitted model describes, so the model was
guaranteed to win. Projections are now generated under both hypotheses.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from response import P5pResponse
from sm_predictions import (
    ANALYSIS_BINS,
    p5p_sm_binned,
    p5p_sm_legacy,
    p5p_sm_uncertainty,
)
from utils import geometric_factor_usf

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

plt.rcParams.update({"font.size": 11, "figure.autolayout": False})

SM_COLOUR = "#1f77b4"
USF_COLOUR = "#2ca02c"
DATA_COLOUR = "#d62728"
LEGACY_COLOUR = "#7f7f7f"


def _save(fig, stem: str) -> None:
    fig.savefig(RESULTS_DIR / f"{stem}.pdf")
    fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    frame = pd.read_csv(REPO_ROOT / "data" / "p5p_observables.csv")
    q2 = frame["q2_center"].values
    observed = frame["value"].values
    error = frame["error"].values
    half_width = (frame["q2_max"].values - frame["q2_min"].values) / 2

    with open(RESULTS_DIR / "fit_results.json") as fh:
        fit = json.load(fh)
    kappa = fit["posterior"]["kappa_best"]
    delta_c9 = fit["models"]["Delta C9 free"]["best_fit"][0]

    sm = p5p_sm_binned(ANALYSIS_BINS)
    sm_error = p5p_sm_uncertainty(ANALYSIS_BINS)
    response = P5pResponse()
    usf = response(kappa, -0.2 * kappa)

    print("Writing figures:")

    # ---- Figure 1: spectrum and residuals, two panels of one figure --------
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )

    top.fill_between(q2, sm - sm_error, sm + sm_error, color=SM_COLOUR,
                     alpha=0.2, label="SM theory uncertainty")
    top.plot(q2, sm, "--", color=SM_COLOUR, marker="s", markersize=4,
             label="SM (flavio, bin-integrated)")
    top.plot(q2, usf, "-", color=USF_COLOUR, marker="o", markersize=4,
             label=rf"USF ($\kappa = {kappa:.2f}$)")
    top.errorbar(q2, observed, yerr=error, xerr=half_width, fmt="o",
                 color=DATA_COLOUR, capsize=0, markersize=5,
                 label="LHCb Run-1")
    top.set_ylabel(r"$P_5'$")
    top.legend(fontsize=9, loc="upper right")
    top.grid(True, linestyle=":", alpha=0.4)
    top.set_title(r"$P_5'$ in $B^0 \to K^{*0}\mu^+\mu^-$: data, SM and USF")

    total = np.hypot(error, sm_error)
    bottom.axhspan(-1, 1, color="0.85", zorder=0)
    bottom.axhspan(-2, 2, color="0.93", zorder=-1)
    bottom.axhline(0, color="black", linewidth=0.8)
    bottom.plot(q2, (observed - sm) / total, "s", color=SM_COLOUR, label="SM")
    bottom.plot(q2, (observed - usf) / total, "o", color=USF_COLOUR, label="USF")
    bottom.set_xlabel(r"$q^2$ (GeV$^2$)")
    bottom.set_ylabel(r"pull ($\sigma$)")
    bottom.set_ylim(-4, 4)
    bottom.legend(fontsize=9, ncol=2)
    bottom.grid(True, linestyle=":", alpha=0.4)
    _save(fig, "spectrum_p5p")

    # ---- Figure 2: what the SM curve error did -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(q2, sm - sm_error, sm + sm_error, color=SM_COLOUR, alpha=0.2)
    ax.plot(q2, sm, "-", color=SM_COLOUR, marker="s", markersize=4,
            label="SM (flavio)")
    ax.plot(q2, p5p_sm_legacy(q2), ":", color=LEGACY_COLOUR, marker="^",
            markersize=4, label="hard-coded curve (published analysis)")
    ax.errorbar(q2, observed, yerr=error, xerr=half_width, fmt="o",
                color=DATA_COLOUR, capsize=0, markersize=5, label="LHCb Run-1")
    ax.set_xlabel(r"$q^2$ (GeV$^2$)")
    ax.set_ylabel(r"$P_5'$")
    ax.set_title("The published SM reference against a real calculation")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.annotate("data sit ABOVE the SM here\n(the actual anomaly)",
                xy=(5.0, -0.30), xytext=(2.2, 0.28), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.annotate("hard-coded curve is wrong here;\nsource of ~66 of the 70.4 $\\chi^2$",
                xy=(16.0, -0.12), xytext=(8.5, 0.10), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    _save(fig, "sm_curve_comparison")

    # ---- Figure 3: the geometric coupling, as predicted and as drawn -------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grid = np.linspace(0, 20, 300)
    ax.plot(grid, geometric_factor_usf(grid), "-", color=USF_COLOUR, linewidth=2,
            label="theory amplitude ($A \\approx 7\\times10^{-64}$)")
    ax.plot(grid, geometric_factor_usf(grid, amplitude=0.8), "--",
            color=LEGACY_COLOUR, linewidth=1.5,
            label="illustrative $A = 0.8$ (as published in Fig. 1)")
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$q^2$ (GeV$^2$)")
    ax.set_ylabel(r"$f_{\rm geo}(q^2)$")
    ax.set_title("Geometric coupling: predicted against illustrated")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(9.5, 1.02,
            "at the amplitude the theory predicts,\n"
            r"$f_{\rm geo} = 1$ to machine precision",
            fontsize=8, color=USF_COLOUR)
    _save(fig, "fator_geometrico")

    # ---- Figure 4: HL-LHC under both hypotheses ----------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    projected_error = error / 5

    for ax, truth, label, colour in (
        (left, sm, "if the SM is true", SM_COLOUR),
        (right, usf, "if the USF is true", USF_COLOUR),
    ):
        ax.fill_between(q2, sm - sm_error, sm + sm_error, color=SM_COLOUR, alpha=0.2)
        ax.plot(q2, sm, "--", color=SM_COLOUR, label="SM")
        ax.plot(q2, usf, "-", color=USF_COLOUR,
                label=rf"USF ($\kappa = {kappa:.2f}$)")
        ax.errorbar(q2, truth, yerr=projected_error, xerr=half_width, fmt="s",
                    color=colour, capsize=0, markersize=5,
                    label="HL-LHC projection")
        ax.set_xlabel(r"$q^2$ (GeV$^2$)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, linestyle=":", alpha=0.4)
    left.set_ylabel(r"$P_5'$")

    separation = np.sqrt(np.sum(((sm - usf) / np.hypot(projected_error, sm_error)) ** 2))
    fig.suptitle(
        "HL-LHC projection under both hypotheses "
        f"(5x smaller errors; separation {separation:.1f}$\\sigma$)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "hllhc_projection")

    # ---- Figure 5: posterior ------------------------------------------------
    import h5py

    with h5py.File(RESULTS_DIR / "mcmc_chains.h5", "r") as fh:
        samples = fh["samples"][:].ravel()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(samples, bins=60, density=True, color="0.75", edgecolor="0.4",
            linewidth=0.5, label="posterior")
    ax.axvspan(fit["posterior"]["kappa_lower"], fit["posterior"]["kappa_upper"],
               color=USF_COLOUR, alpha=0.18, label="68% credible")
    ax.axvline(fit["posterior"]["kappa_median"], color=USF_COLOUR, linewidth=2,
               label="median")
    ax.axvline(0, color=DATA_COLOUR, linestyle="--", linewidth=1.5,
               label="SM ($\\kappa = 0$)")
    ax.axvline(1.44, color=LEGACY_COLOUR, linestyle=":", linewidth=1.5,
               label="published $\\kappa = +1.44$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("probability density")
    ax.set_title(r"Posterior for $\kappa$ against a correct SM reference")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "corner_kappa")

    print(f"\nBest fit kappa = {kappa:+.3f} (free Delta C9 baseline: {delta_c9:+.3f})")
    print(f"HL-LHC SM/USF separation with 5x smaller errors: {separation:.1f} sigma")


if __name__ == "__main__":
    main()
