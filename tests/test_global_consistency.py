# tests/test_global_consistency.py
"""Tests on the cross-check against other b -> s l l observables.

The manuscript declares Delta C10 = -0.2 * Delta C9 a fixed prediction of the
framework, "not adjusted to the data". That makes it testable for free against
observables the analysis never looked at, which is the second referee's third
point.
"""
import json

import pytest

from conftest import RESULTS_DIR


@pytest.fixture(scope="module")
def consistency():
    path = RESULTS_DIR / "global_consistency.json"
    if not path.exists():
        pytest.skip("run scripts/global_consistency.py first")
    with open(path) as fh:
        return json.load(fh)


def test_both_lepton_scenarios_are_evaluated(consistency):
    """The manuscript never says whether the effect is lepton universal.

    The two readings give opposite verdicts, so reporting only one would be
    choosing the answer.
    """
    assert {"muon-specific", "lepton-universal"} == set(consistency["scenarios"])


def test_c10_ratio_is_the_one_the_manuscript_fixes(consistency):
    """The tested point must use the declared ratio, not a fitted one."""
    assert consistency["c10_over_c9"] == -0.2
    assert consistency["delta_c10"] == pytest.approx(
        -0.2 * consistency["delta_c9"], rel=1e-9
    )


def test_muon_specific_reading_is_excluded(consistency):
    """A muon-only Delta C9 near -1.5 is ruled out by lepton-universality ratios.

    R_K would fall to 0.64 against a measured 0.949 +- 0.047.
    """
    scenario = consistency["scenarios"]["muon-specific"]
    assert abs(scenario["worst_pull"]) > 5.0
    assert scenario["sum_squared_pulls_usf"] > 10 * consistency["sum_squared_pulls_sm"]


def test_lepton_universal_reading_survives(consistency):
    """Applied to both lepton species, the fitted point is compatible.

    This is the one result in the whole review that favours the model, and it
    is recorded as such. It is not evidence *for* a geometric origin: a
    lepton-universal Delta C9 near -1 is equally what an unmodelled charm-loop
    hadronic contribution would produce.
    """
    scenario = consistency["scenarios"]["lepton-universal"]
    assert abs(scenario["worst_pull"]) < 2.0
    assert scenario["sum_squared_pulls_usf"] <= consistency["sum_squared_pulls_sm"]
