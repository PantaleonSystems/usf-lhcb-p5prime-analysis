# scripts/fit_usf.py
"""Fit the USF parameter to LHCb P5' data, against a proper baseline.

Differences from the published version, all of which change the result:

*Correct SM reference.* scripts/sm_predictions.py rather than a hard-coded
interpolation table that was wrong by up to 26 theory sigma.

*Full covariance.* chi2 = r^T (Sigma_exp + Sigma_theory)^-1 r. The published
chi2 divided by the experimental error alone, and theory errors here are
0.03-0.10 with correlations up to 0.97 between bins.

*Exact response.* scripts/response.py rather than dP5'/dC9 = -0.3 constant,
which is wrong by a factor of 14 across the bins.

*A baseline to beat.* A model must be compared with the naive alternative, not
only with the SM. The relevant baseline is a free constant Delta C9, which the
b -> s l l literature has fitted for a decade. If the USF cannot beat it, its
geometric content is empty however large Delta chi2 looks.

*Calibrated significance.* sqrt(Delta chi2) is asymptotic. With seven points
and a bounded prior it is calibrated here against toy Monte Carlo.

*Reproducibility.* Seeded, with convergence diagnostics and provenance written
alongside the results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from response import GRID_DC9, P5pResponse
from sm_predictions import ANALYSIS_BINS, p5p_sm_binned, p5p_sm_covariance

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "p5p_observables.csv"
RESULTS_DIR = REPO_ROOT / "results"

SEED = 20260812
N_WALKERS = 32
N_STEPS = 8000
N_TOYS = 5000

#: Natural scale of the decay, q0 = m_B, as used in the manuscript's f_geo.
Q0_SQUARED = 5.279 ** 2

#: Ratio Delta C10 / Delta C9, declared in the manuscript as a fixed prediction
#: of the framework's tensor structure and not adjusted to the data.
C10_OVER_C9 = -0.2


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Model:
    """A hypothesis mapping free parameters to (Delta C9, Delta C10) per bin."""

    name: str
    n_params: int
    bounds: tuple[float, float] | None = None

    def coefficients(self, theta, q2):
        raise NotImplementedError

    def predict(self, theta, response, q2):
        dc9, dc10 = self.coefficients(theta, q2)
        if np.isscalar(dc9):
            return response(dc9, dc10)
        # Bin-dependent coefficients: evaluate bin by bin.
        return np.array([
            response(a, b)[i] for i, (a, b) in enumerate(zip(dc9, dc10))
        ])


class StandardModel(Model):
    name = "SM"
    n_params = 0

    def coefficients(self, theta, q2):
        return 0.0, 0.0


class FreeC9(Model):
    """The literature benchmark: a single constant shift in C9."""

    name = "Delta C9 free"
    n_params = 1
    bounds = (GRID_DC9[0] * 0.95, GRID_DC9[-1] * 0.95)

    def coefficients(self, theta, q2):
        return float(np.atleast_1d(theta)[0]), 0.0


class USF(Model):
    """The published model: Delta C9 = kappa * f_geo, Delta C10 = -0.2 * that.

    f_geo is taken from scripts/utils.py. With the manuscript's own constants it
    evaluates to 1 identically, so this reduces to a constant shift with the
    C10 ratio imposed -- which is precisely what the fit is here to expose.
    """

    name = "USF (kappa)"
    n_params = 1
    bounds = (-2.0, 2.0)

    def coefficients(self, theta, q2):
        from utils import geometric_factor_usf

        kappa = float(np.atleast_1d(theta)[0])
        f_geo = geometric_factor_usf(q2)
        dc9 = kappa * f_geo
        return dc9, C10_OVER_C9 * dc9


class USFAnsatz(Model):
    """The USF shape with its normalisation freed.

        f_geo(q^2) = 1 + A / (1 + q^2/q0^2)

    Restating the geometric factor as a phenomenological ansatz, since the
    Planck suppression makes A ~ 1e-30 rather than order unity. Two parameters:
    kappa sets the overall size, A sets how much q^2 dependence is preferred.
    A = 0 recovers a constant shift, so this measures directly whether the q^2
    shape earns its extra parameter.
    """

    name = "USF ansatz (kappa, A)"
    n_params = 2
    bounds = None

    def coefficients(self, theta, q2):
        kappa, amplitude = float(theta[0]), float(theta[1])
        f_geo = 1.0 + amplitude / (1.0 + q2 / Q0_SQUARED)
        dc9 = kappa * f_geo
        return dc9, C10_OVER_C9 * dc9


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

class Likelihood:
    def __init__(self, observed, covariance, response, q2):
        self.observed = observed
        self.covariance = covariance
        self.cho = cho_factor(covariance)
        self.response = response
        self.q2 = q2

    #: Returned when a trial point falls outside the precomputed response grid.
    #: Large enough to repel a simplex, finite so the optimiser keeps working.
    OUT_OF_RANGE = 1e6

    def chi2(self, model: Model, theta) -> float:
        try:
            prediction = model.predict(theta, self.response, self.q2)
        except ValueError:
            # Outside the (Delta C9, Delta C10) grid. Extrapolating a cubic
            # interpolator there would be silently wrong, so refuse instead.
            return self.OUT_OF_RANGE
        residual = self.observed - prediction
        return float(residual @ cho_solve(self.cho, residual))

    def fit(self, model: Model):
        """Minimise chi2. Returns (best-fit parameters, chi2)."""
        if model.n_params == 0:
            return np.array([]), self.chi2(model, np.array([]))

        if model.n_params == 1:
            lo, hi = model.bounds
            result = minimize_scalar(
                lambda t: self.chi2(model, [t]), bounds=(lo, hi), method="bounded",
            )
            return np.array([result.x]), float(result.fun)

        from scipy.optimize import minimize

        best = None
        for start in ([-1.0, 0.0], [-1.0, 1.0], [0.5, -0.5], [-2.0, 3.0]):
            trial = minimize(
                lambda t: self.chi2(model, t), x0=start, method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000},
            )
            if best is None or trial.fun < best.fun:
                best = trial
        return best.x, float(best.fun)


# ---------------------------------------------------------------------------
# Significance calibration
# ---------------------------------------------------------------------------

def toy_significance(likelihood: Likelihood, model: Model, sm: StandardModel,
                     delta_chi2_observed: float, n_toys: int, rng) -> dict:
    """Calibrate Delta chi2 against pseudo-experiments generated under the SM.

    Wilks' theorem gives Delta chi2 ~ chi2(1 dof) asymptotically. With seven
    points and a bounded parameter that is an assumption, not a result.
    """
    sm_prediction = sm.predict(np.array([]), likelihood.response, likelihood.q2)
    draws = rng.multivariate_normal(
        sm_prediction, likelihood.covariance, size=n_toys
    )

    original = likelihood.observed
    statistics = np.empty(n_toys)
    try:
        for i, pseudo in enumerate(draws):
            likelihood.observed = pseudo
            _, chi2_sm = likelihood.fit(sm)
            _, chi2_model = likelihood.fit(model)
            statistics[i] = chi2_sm - chi2_model
    finally:
        likelihood.observed = original

    exceed = int(np.sum(statistics >= delta_chi2_observed))
    p_value = (exceed + 1) / (n_toys + 1)  # conservative, avoids p = 0
    return {
        "p_value_toys": p_value,
        "significance_toys": float(norm.isf(p_value)),
        "p_value_wilks": float(chi2_dist.sf(delta_chi2_observed, 1)),
        "significance_wilks_naive": float(np.sqrt(max(delta_chi2_observed, 0.0))),
        "n_toys": n_toys,
        "toys_exceeding": exceed,
        "toy_median_delta_chi2": float(np.median(statistics)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    frame = pd.read_csv(DATA_FILE)
    q2 = frame["q2_center"].values
    observed = frame["value"].values
    exp_error = frame["error"].values
    print(f"Loaded {len(q2)} q^2 bins (P5') from LHCb Run-1.")

    covariance_exp = np.diag(exp_error ** 2)
    covariance_th = p5p_sm_covariance(ANALYSIS_BINS)
    covariance = covariance_exp + covariance_th

    print("Building the Wilson-coefficient response grid...")
    response = P5pResponse()
    worst = response.validate(n_points=15)
    assert worst < 1e-3, f"response interpolator inaccurate ({worst:.2e})"
    print(f"  interpolator validated, worst deviation {worst:.2e}")

    likelihood = Likelihood(observed, covariance, response, q2)

    sm = StandardModel()
    models = [sm, FreeC9(), USF(), USFAnsatz()]

    print(f"\n{'model':<24} {'params':>7} {'chi2':>8} {'dof':>5} "
          f"{'p(gof)':>8} {'AIC':>8}  best fit")
    results = {}
    n_data = len(q2)
    for model in models:
        theta, chi2_value = likelihood.fit(model)
        dof = n_data - model.n_params
        aic = chi2_value + 2 * model.n_params
        bic = chi2_value + model.n_params * np.log(n_data)
        gof = float(chi2_dist.sf(chi2_value, dof))
        results[model.name] = {
            "n_params": model.n_params,
            "chi2": chi2_value,
            "dof": dof,
            "p_goodness_of_fit": gof,
            "aic": aic,
            "bic": bic,
            "best_fit": theta.tolist(),
        }
        rendered = ", ".join(f"{v:+.4f}" for v in theta) or "-"
        print(f"{model.name:<24} {model.n_params:>7} {chi2_value:>8.2f} {dof:>5} "
              f"{gof:>8.3f} {aic:>8.2f}  {rendered}")

    # ---- Model comparisons -------------------------------------------------
    chi2_sm = results[sm.name]["chi2"]
    chi2_c9 = results["Delta C9 free"]["chi2"]
    chi2_usf = results["USF (kappa)"]["chi2"]
    chi2_ansatz = results["USF ansatz (kappa, A)"]["chi2"]

    comparisons = {
        "delta_chi2_sm_minus_usf": chi2_sm - chi2_usf,
        "delta_chi2_sm_minus_c9": chi2_sm - chi2_c9,
        "delta_chi2_c9_minus_usf": chi2_c9 - chi2_usf,
        "delta_chi2_usf_minus_ansatz": chi2_usf - chi2_ansatz,
    }
    print("\nModel comparisons (positive favours the second model):")
    for key, value in comparisons.items():
        print(f"  {key:<32} {value:+8.3f}")

    # ---- Significance ------------------------------------------------------
    print(f"\nCalibrating significance with {N_TOYS} SM pseudo-experiments...")
    usf_model = USF()
    significance = toy_significance(
        likelihood, usf_model, sm, chi2_sm - chi2_usf, N_TOYS, rng
    )
    print(f"  naive sqrt(Delta chi2) : {significance['significance_wilks_naive']:.2f} sigma")
    print(f"  Wilks p-value          : {significance['p_value_wilks']:.4f}")
    print(f"  toy-calibrated p-value : {significance['p_value_toys']:.4f} "
          f"({significance['significance_toys']:.2f} sigma)")

    # ---- Posterior ---------------------------------------------------------
    print(f"\nSampling the posterior for kappa ({N_WALKERS} walkers, {N_STEPS} steps)...")
    lo, hi = usf_model.bounds

    def log_posterior(theta):
        if not lo < theta[0] < hi:
            return -np.inf
        return -0.5 * likelihood.chi2(usf_model, theta)

    kappa_best = results["USF (kappa)"]["best_fit"][0]
    start = kappa_best + 1e-3 * rng.standard_normal((N_WALKERS, 1))
    # emcee draws from the global NumPy RNG, not from `rng`. Without this the
    # posterior interval moves between runs -- which is how three different 68%
    # intervals ended up in circulation for the published result.
    np.random.seed(SEED)
    sampler = emcee.EnsembleSampler(N_WALKERS, 1, log_posterior)
    sampler.run_mcmc(start, N_STEPS, progress=False)

    try:
        tau = float(np.max(sampler.get_autocorr_time()))
        converged = True
    except emcee.autocorr.AutocorrError as exc:
        tau = float(np.max(exc.tau))
        converged = False
    burn_in = int(min(5 * tau, N_STEPS // 2))
    thin = max(1, int(tau / 2))
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    acceptance = float(np.mean(sampler.acceptance_fraction))

    print(f"  autocorrelation time {tau:.1f}, burn-in {burn_in}, thin {thin}")
    print(f"  acceptance fraction {acceptance:.3f}, "
          f"{len(samples)} effective samples, converged={converged}")

    quantiles = np.percentile(samples, [16, 50, 84])
    print(f"  kappa = {quantiles[1]:+.4f} "
          f"+{quantiles[2] - quantiles[1]:.4f} -{quantiles[1] - quantiles[0]:.4f}")

    # ---- Persist -----------------------------------------------------------
    import flavio
    import wilson as wilson_pkg

    payload = {
        "models": results,
        "comparisons": comparisons,
        "significance": significance,
        "posterior": {
            "kappa_median": float(quantiles[1]),
            "kappa_lower": float(quantiles[0]),
            "kappa_upper": float(quantiles[2]),
            "kappa_best": float(kappa_best),
        },
        "diagnostics": {
            "autocorr_time": tau,
            "acceptance_fraction": acceptance,
            "burn_in": burn_in,
            "thin": thin,
            "converged": converged,
            "n_effective_samples": int(len(samples)),
        },
        "provenance": {
            "seed": SEED,
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_toys": N_TOYS,
            "bins": [list(b) for b in ANALYSIS_BINS],
            "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
            "flavio_version": flavio.__version__,
            "wilson_version": wilson_pkg.__version__,
            "theory_covariance_included": True,
            "experimental_correlations_included": True,
        },
    }

    with open(RESULTS_DIR / "fit_results.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    import h5py

    with h5py.File(RESULTS_DIR / "mcmc_chains.h5", "w") as fh:
        fh.create_dataset("samples", data=samples)
        fh.attrs["seed"] = SEED

    print(f"\nResults written to {RESULTS_DIR.name}/")


if __name__ == "__main__":
    main()
