# tests/conftest.py
"""Shared fixtures for the USF validation test suite.

The suite is written to be *falsifiable*: several tests are expected to fail
against the current implementation. Each failing test names the defect it
detects in its docstring. See tests/README.md for the current status.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
YAML_DIR = DATA_DIR / "raw" / "HEPData-ins1409497-v1-yaml"
RESULTS_DIR = REPO_ROOT / "results"

# scripts/ is not a package; make its modules importable as top-level names,
# which is how fit_usf.py and plot_p5p.py already import utils.
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Physical reference values (independent of this repository's code)
# ---------------------------------------------------------------------------

#: Planck energy in GeV (PDG). The repository stores E_P in joules.
E_PLANCK_GEV = 1.22e19

#: LHCb Run-1 collision energies. The dataset fitted here is 7 and 8 TeV,
#: *not* the 14 TeV design energy used in geometric_factor_usf().
RUN1_SQRT_S_GEV = (7.0e3, 8.0e3)

#: The seven q^2 bins actually used in the analysis, as (low, high) in GeV^2.
#: These are the Table2 bins minus the photon-pole bin (0.1-0.98) and the
#: charmonium regions (8-11 and 12.5-15).
ANALYSIS_BINS = [
    (1.1, 2.5),
    (2.5, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (11.0, 12.5),
    (15.0, 17.0),
    (17.0, 19.0),
]

#: Bin centres, matching data/p5p_observables.csv.
ANALYSIS_BIN_CENTRES = np.array([np.mean(b) for b in ANALYSIS_BINS])


@pytest.fixture(scope="session")
def utils():
    """The module under test: scripts/utils.py."""
    import utils as _utils

    return _utils


@pytest.fixture(scope="session")
def sm():
    """SM predictions: scripts/sm_predictions.py.

    Owns what utils.p5p_sm used to do with a hard-coded interpolation table.
    """
    pytest.importorskip("flavio", reason="sm_predictions is a flavio wrapper")
    import sm_predictions

    return sm_predictions


@pytest.fixture(scope="session")
def observed():
    """Measured P5' values used by the fit (data/p5p_observables.csv)."""
    df = pd.read_csv(DATA_DIR / "p5p_observables.csv")
    return df.q2_center.values, df.value.values, df.error.values


@pytest.fixture(scope="session")
def flavio_sm():
    """Bin-integrated SM predictions for P5' and their theory uncertainties.

    Computed with flavio, which is the reference this repository should be
    anchored against instead of a hard-coded interpolation table.
    """
    flavio = pytest.importorskip("flavio", reason="flavio provides the SM reference")
    central, sigma = [], []
    for lo, hi in ANALYSIS_BINS:
        central.append(flavio.sm_prediction("<P5p>(B0->K*mumu)", lo, hi))
        sigma.append(flavio.sm_uncertainty("<P5p>(B0->K*mumu)", lo, hi))
    return np.array(central), np.array(sigma)


@pytest.fixture(scope="session")
def hepdata():
    """Loader for HEPData YAML tables of record ins1409497."""
    yaml = pytest.importorskip("yaml")

    def _load(table, observable):
        path = YAML_DIR / f"{table}.yaml"
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        matches = [
            dv for dv in doc["dependent_variables"]
            if dv["header"]["name"] == observable
        ]
        if not matches:
            raise KeyError(f"{observable!r} not present in {table}.yaml")
        values, errors = [], []
        for entry in matches[0]["values"]:
            var = 0.0
            for err in entry["errors"]:
                if "symerror" in err:
                    var += err["symerror"] ** 2
                else:
                    plus = err["asymerror"]["plus"]
                    minus = err["asymerror"]["minus"]
                    var += ((plus - minus) / 2) ** 2
            values.append(entry["value"])
            errors.append(np.sqrt(var))
        bins = [
            (v["low"], v["high"])
            for v in doc["independent_variables"][0]["values"]
        ]
        return bins, np.array(values), np.array(errors)

    return _load
