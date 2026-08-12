# tests/test_pipeline.py
"""Reproducibility tests on the fit outputs.

The manuscript claims a "fully reproducible" open-source pipeline. Three
different 68% credibility intervals for kappa are currently in circulation:

    manuscript (Table 2)      [1.26,  1.64 ]
    results/fit_results.json  [1.247, 1.640]
    README.md                 [1.249, 1.629]

They disagree because the MCMC has no seed, so every run produces a different
interval and nothing checks the published numbers against the generated ones.
"""
import json
import re

import pytest

from conftest import REPO_ROOT, RESULTS_DIR

MANUSCRIPT_INTERVAL = (1.26, 1.64)


@pytest.fixture(scope="module")
def fit_results():
    path = RESULTS_DIR / "fit_results.json"
    if not path.exists():
        pytest.skip("results/fit_results.json not generated yet")
    with open(path) as fh:
        return json.load(fh)


def test_mcmc_is_seeded():
    """The sampler must start from a reproducible state.

    DEFECT: scripts/fit_usf.py calls np.random.normal to place the walkers and
    never seeds the generator, so kappa_lower/kappa_upper change on every run.
    """
    source = (REPO_ROOT / "scripts" / "fit_usf.py").read_text()
    assert re.search(r"(np\.random\.seed|default_rng|RandomState|\bseed\s*=)", source), (
        "no seed is set in fit_usf.py; the credibility interval is not reproducible"
    )


def test_provenance_is_recorded(fit_results):
    """Results must record how they were produced.

    A results file that cannot be traced to a seed, a code revision and an
    input hash cannot be checked by a referee or by CI.
    """
    missing = {"seed", "n_steps", "n_walkers"} - set(fit_results)
    assert not missing, f"fit_results.json records no {sorted(missing)}"


def test_reported_interval_matches_the_manuscript(fit_results):
    """The published interval must match the generated one.

    DEFECT: fit_results.json gives [1.247, 1.640] against the manuscript's
    [1.26, 1.64]. Tolerance here is 0.005, i.e. the rounding implied by the
    two decimals quoted in Table 2.
    """
    lower = fit_results["kappa_lower"]
    upper = fit_results["kappa_upper"]
    assert lower == pytest.approx(MANUSCRIPT_INTERVAL[0], abs=0.005), (
        f"kappa_lower = {lower:.4f}, manuscript quotes {MANUSCRIPT_INTERVAL[0]}"
    )
    assert upper == pytest.approx(MANUSCRIPT_INTERVAL[1], abs=0.005), (
        f"kappa_upper = {upper:.4f}, manuscript quotes {MANUSCRIPT_INTERVAL[1]}"
    )


def test_readme_numbers_match_fit_results(fit_results):
    """README's worked example must match the committed results.

    DEFECT: README.md shows kappa_lower 1.2493 / kappa_upper 1.6293 while
    fit_results.json holds 1.2468 / 1.6401. The README was pasted from a
    different, unseeded run.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    for key in ("kappa_lower", "kappa_upper"):
        quoted = re.search(rf'"{key}":\s*([-\d.]+)', readme)
        assert quoted, f"{key} not quoted in README"
        assert float(quoted.group(1)) == pytest.approx(fit_results[key], abs=5e-4), (
            f"README quotes {key} = {quoted.group(1)}, "
            f"fit_results.json has {fit_results[key]:.4f}"
        )


def test_convergence_diagnostics_are_reported(fit_results):
    """Autocorrelation time and acceptance fraction must be recorded.

    DEFECT: fit_usf.py runs a fixed 2000 steps and discards half as burn-in
    without checking that the chain converged.
    """
    missing = {"autocorr_time", "acceptance_fraction"} - set(fit_results)
    assert not missing, f"fit_results.json records no {sorted(missing)}"
