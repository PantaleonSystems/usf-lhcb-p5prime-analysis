# tests/test_data_extraction.py
"""Tests on the HEPData -> CSV extraction step.

The manuscript states (Sec. 3.5) that the pipeline allows "full reproduction of
the analysis" from the raw YAML files. It does not: scripts/generate_p5p_csv.py
reads F_L and S_5 from different tables with different binnings and would emit
an empty CSV. The committed data/p5p_observables.csv was produced some other
way.

The numbers in the committed CSV are nevertheless correct -- they match a
Table2-only reconstruction exactly -- so the tests separate "the values are
right" from "the script that claims to produce them works".
"""
import numpy as np
import pytest

from conftest import ANALYSIS_BINS, YAML_DIR

FL = "$F_{\\rm L}$"
S5 = "$S_{5}$"
P5P = "$P'_{5}$"


@pytest.fixture(scope="module")
def generator():
    import generate_p5p_csv

    return generate_p5p_csv


def test_fl_and_s5_are_read_from_the_same_table(generator):
    """Both angular coefficients must come from one consistently binned table.

    DEFECT: FL_FILE points at Table1.yaml (two wide bins: 1.1-6 and 15-19)
    while S5_FILE points at Table2.yaml (eight narrow bins). main() then zips
    the two lists positionally, pairing F_L(1.1-6) with S_5(0.1-0.98).
    """
    assert generator.FL_FILE.name == generator.S5_FILE.name, (
        f"F_L is read from {generator.FL_FILE.name} and S_5 from "
        f"{generator.S5_FILE.name}; these tables have different binnings"
    )


def test_extraction_selects_the_seven_analysis_bins(generator, hepdata):
    """The bin-matching filter must actually select seven bins.

    DEFECT: main() derives candidate bin centres from FL_FILE (Table1), giving
    3.55 and 17.0 GeV^2. Neither is within the 0.1 tolerance of any target
    centre, so `selected` is empty and the script writes a zero-row CSV
    without raising.
    """
    bins, _, _ = hepdata("Table1", FL)
    centres = [(lo + hi) / 2 for lo, hi in bins]
    targets = np.array([np.mean(b) for b in ANALYSIS_BINS])
    selected = [c for c in centres if np.any(np.abs(targets - c) < 0.1)]
    assert len(selected) == 7, (
        f"bin centres derived from {generator.FL_FILE.name} are {centres}; "
        f"{len(selected)} of the 7 required bins match"
    )


def test_committed_csv_matches_a_table2_reconstruction(observed, hepdata):
    """The committed values must be reproducible from the raw YAML.

    This one passes: the CSV is a faithful Table2 reconstruction with the
    photon-pole bin dropped. It pins the numbers so a future fix to the
    extraction script cannot silently change them.
    """
    _, values, errors = observed
    bins, fl, fl_err = hepdata("Table2", FL)
    _, s5, s5_err = hepdata("Table2", S5)

    keep = [i for i, b in enumerate(bins) if b in ANALYSIS_BINS]
    fl, fl_err = fl[keep], fl_err[keep]
    s5, s5_err = s5[keep], s5_err[keep]

    denom = np.sqrt(fl * (1 - fl))
    p5p = s5 / denom
    d_s5 = 1 / denom
    d_fl = -0.5 * s5 * (1 - 2 * fl) / denom ** 3
    sigma = np.sqrt((d_s5 * s5_err) ** 2 + (d_fl * fl_err) ** 2)

    np.testing.assert_allclose(values, p5p, rtol=1e-10)
    np.testing.assert_allclose(errors, sigma, rtol=1e-10)


def test_published_p5p_is_used_where_available(hepdata):
    """LHCb publishes P5' directly; reconstructing it is unnecessary.

    Table1 contains a measured $P'_{5}$ column for the 1.1-6 and 15-19 GeV^2
    bins, with the experimental correlations handled by the collaboration.
    Rebuilding P5' from F_L and S_5 under a diagonal-error assumption discards
    that treatment for no gain.
    """
    bins, values, errors = hepdata("Table1", P5P)
    assert len(values) == 2 and errors[0] > 0
    # The reference values the analysis should be anchored to.
    assert values[0] == pytest.approx(-0.049, abs=0.01)
    assert values[1] == pytest.approx(-0.684, abs=0.01)


def test_correlation_matrices_are_available_and_used():
    """The per-bin correlation matrices are in the repository but unused.

    Sec. 3.2 estimates that neglecting the F_L-S_5 correlation "affects the
    total uncertainty by at most 10%". No estimate is needed: HEPData ships the
    full likelihood correlation matrix for every q^2 bin, and those files are
    already vendored in data/raw/.

    DEFECT: no module in scripts/ reads them.
    """
    import yaml

    with open(YAML_DIR / "submission.yaml") as fh:
        docs = [d for d in yaml.safe_load_all(fh) if d and "data_file" in d]
    corr_tables = [
        d["data_file"] for d in docs
        if "correlation" in d.get("description", "").lower()
    ]
    assert len(corr_tables) >= 7, "expected one correlation matrix per q^2 bin"

    sources = (YAML_DIR.parent.parent.parent / "scripts").glob("*.py")
    users = [p.name for p in sources if "correlation" in p.read_text().lower()]
    assert users, (
        f"{len(corr_tables)} correlation matrices are vendored "
        f"({corr_tables[0]} ...) but no script in scripts/ reads them"
    )
