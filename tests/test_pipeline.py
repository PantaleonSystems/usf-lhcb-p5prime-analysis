# tests/test_pipeline.py
"""Reproducibility tests on the fit outputs.

The manuscript claimed a "fully reproducible" pipeline while three different
68% credibility intervals for kappa were in circulation:

    manuscript (Table 2)      [1.26,  1.64 ]
    results/fit_results.json  [1.247, 1.640]
    README.md                 [1.249, 1.629]

They disagreed because the MCMC had no seed and nothing checked the published
numbers against the generated ones.
"""
import json
import re

import pytest

from conftest import REPO_ROOT, RESULTS_DIR


@pytest.fixture(scope="module")
def fit():
    path = RESULTS_DIR / "fit_results.json"
    if not path.exists():
        pytest.skip("run scripts/fit_usf.py first")
    with open(path) as fh:
        return json.load(fh)


def test_mcmc_is_seeded():
    """The sampler must start from a reproducible state."""
    source = (REPO_ROOT / "scripts" / "fit_usf.py").read_text()
    assert re.search(r"default_rng\(|np\.random\.seed\(", source), (
        "no seed is set in fit_usf.py"
    )


def test_provenance_is_recorded(fit):
    """Results must record how they were produced.

    A results file that cannot be traced to a seed, an input hash and the
    versions of the theory codes cannot be checked by a referee or by CI.
    """
    provenance = fit["provenance"]
    required = {
        "seed", "n_walkers", "n_steps", "n_toys",
        "data_sha256", "flavio_version", "wilson_version", "bins",
    }
    assert required <= set(provenance), f"missing {sorted(required - set(provenance))}"


def test_input_hash_matches_the_data_file(fit):
    """The recorded hash must correspond to the CSV actually on disk."""
    import hashlib

    from conftest import DATA_DIR

    digest = hashlib.sha256((DATA_DIR / "p5p_observables.csv").read_bytes()).hexdigest()
    assert fit["provenance"]["data_sha256"] == digest, (
        "results were produced from a different version of the input data"
    )


def test_convergence_diagnostics_are_reported(fit):
    """Autocorrelation time and acceptance fraction must be recorded.

    The published version ran a fixed 2000 steps and discarded half as burn-in
    without checking that the chain had converged.
    """
    diagnostics = fit["diagnostics"]
    assert {"autocorr_time", "acceptance_fraction", "burn_in", "converged"} <= set(diagnostics)
    assert diagnostics["converged"], "chain did not satisfy the autocorrelation criterion"
    assert 0.15 < diagnostics["acceptance_fraction"] < 0.9
    assert diagnostics["burn_in"] >= 5 * diagnostics["autocorr_time"] * 0.99


def test_uncertainty_treatment_is_complete(fit):
    """Both theory and experimental correlations must be in the likelihood."""
    provenance = fit["provenance"]
    assert provenance["theory_covariance_included"]
    assert provenance["experimental_correlations_included"]


def test_posterior_interval_is_consistent(fit):
    """The credible interval must bracket the median and the best fit."""
    posterior = fit["posterior"]
    assert posterior["kappa_lower"] < posterior["kappa_median"] < posterior["kappa_upper"]
    assert posterior["kappa_lower"] < posterior["kappa_best"] < posterior["kappa_upper"]
