"""
W.I.N.G.S. — data ingestion tests.

Covers the two filename parsers (factorial ABM/WFM sweep and the Δp
fraction sweep), column normalisation, and a full directory round-trip
through ingest.main().
"""

import pandas as pd
import pytest

from wings.analysis import ingest, ingest_delta_p
from wings.models import wfm

# ======================================================================
#  Factorial-sweep filename parser
# ======================================================================

def test_parse_conditions_valid():
    fname = ("cytoplasmic_incompatibility_True_male_killing_False_"
             "increased_exploration_rate_True_increased_eggs_False_7.csv")
    assert ingest.parse_conditions_from_filename(fname) == (True, False, True, False, 7)


@pytest.mark.parametrize("bad", [
    "garbage.csv",
    "cytoplasmic_incompatibility_Maybe_male_killing_False_"
    "increased_exploration_rate_True_increased_eggs_False_7.csv",
    "summary.txt",
])
def test_parse_conditions_invalid_returns_none(bad):
    assert ingest.parse_conditions_from_filename(bad) is None


def test_normalise_columns_variants():
    df = pd.DataFrame({"population": [1], "infection_rate": [0.5]})
    out = ingest.normalise_columns(df)
    assert "Population Size" in out.columns
    assert "Infection Rate" in out.columns


# ======================================================================
#  Δp sweep filename parser
# ======================================================================

def test_delta_p_parse_filename():
    assert ingest_delta_p.parse_filename("CI_frac050_rep3.csv") == ("CI", 0.5, 3)
    assert ingest_delta_p.parse_filename("CI_ER_frac100_rep0.csv") == ("CI_ER", 1.0, 0)


def test_delta_p_parse_filename_invalid():
    assert ingest_delta_p.parse_filename("nope.csv") is None


# ======================================================================
#  Full directory round-trip
# ======================================================================

def test_ingest_directory_roundtrip(tmp_path):
    combos = [(True, False, True, False), (False, False, False, False)]
    for ci, mk, er, ie in combos:
        fname = wfm.make_filename(ci, mk, er, ie, 0)
        pd.DataFrame({
            "Population Size": [50, 51, 52],
            "Infection Rate": [0.10, 0.20, 0.30],
        }).to_csv(tmp_path / fname, index=False)

    out = tmp_path / "combined.csv"
    ingest.main(str(tmp_path), str(out))

    combined = pd.read_csv(out)
    assert len(combined) == 6                      # 2 combos × 3 rows
    assert list(combined["Day"].head(3)) == [1, 2, 3]
    for col in ["Cytoplasmic Incompatibility", "Male Killing",
                "Increased Exploration Rate", "Increased Eggs", "Replicate ID"]:
        assert col in combined.columns
    assert set(combined["Cytoplasmic Incompatibility"]) == {True, False}
