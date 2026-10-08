"""Tests for baseline.py. Run: python -m pytest -q test_baseline.py"""

import numpy as np
import pandas as pd
import pytest

import baseline as B

SLOPE = 0.3  # true effect of the score on log permits and log starts in the simulation


def panel(
    units: int = 40, periods: int = 24, pre_trend: float = 0.0, seed: int = 1
) -> pd.DataFrame:
    """A quarterly census-division panel with known slopes; half the divisions adopt.

    Column names are written the way a spreadsheet might have them, to test matching.
    pre_trend adds a trend (per quarter) to adopters only, before and after adoption.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2015Q1", periods=periods, freq="Q")
    adopter = np.arange(units) < units // 2
    adopt_at = rng.integers(8, periods - 4, units)
    each = lambda values: np.repeat(values, periods)  # noqa: E731
    t = np.tile(np.arange(periods), units)
    score = each(rng.uniform(0, 10, units))
    trend = pre_trend * t * each(adopter)
    noise = lambda: rng.normal(0, 0.1, units * periods)  # noqa: E731
    adopted = np.where(adopter, [str(quarters[i]) for i in adopt_at], "")
    return pd.DataFrame(
        {
            "Census Division": each([3500 + i for i in range(units)]),
            "Date": np.tile(quarters.to_timestamp().strftime("%Y-%m-%d"), units),
            "Score": score,
            "Permits": np.expm1(4 + SLOPE * score + trend + noise()).round(),
            "Starts": np.expm1(3 + SLOPE * score + trend + noise()).round(),
            "Treatment Date": each(adopted),
            "Mortgage Lending": rng.normal(100, 10, units * periods),
            "Development Charges": rng.normal(20, 2, units * periods),
            "BCPI": np.tile(np.linspace(100, 130, periods), units),
            "NHPI": rng.normal(100, 5, units * periods),
            "CPI Shelter": np.tile(120 * 1.01 ** np.arange(periods), units) + noise(),
            "Unemployment Rate": rng.normal(6, 1, units * periods),
            "Completions": rng.poisson(40, units * periods),
            "Under Construction": rng.poisson(90, units * periods),
        }
    )


def write(frame: pd.DataFrame, tmp_path, name: str = "data.csv"):
    path = tmp_path / name
    frame.to_excel(path, index=False) if name.endswith("xlsx") else frame.to_csv(path, index=False)
    return path


def prepared(frame: pd.DataFrame, tmp_path) -> pd.DataFrame:
    return B.prepare(B.load(write(frame, tmp_path)))[0]


@pytest.mark.parametrize(
    ("values", "first"),
    [
        (["2015Q1", "2015Q2"], "2015Q1"),
        (["Q3 2015", "Q4 2015"], "2015Q3"),
        (["2015-04", "2015-07"], "2015Q2"),  # StatCan quarterly REF_DATE
        (["2015-05-20", None], "2015Q2"),
        ([2015.0, 2016.0], "2015Q1"),
    ],
)
def test_to_period(values, first):
    assert str(B.to_period(pd.Series(values, name="date")).iloc[0]) == first


def test_prepare(tmp_path):
    frame = panel().astype({"NHPI": object})
    frame.loc[0, "NHPI"] = ".."  # StatCan's "not available"
    df, notes = B.prepare(B.load(write(frame, tmp_path)))
    assert notes == ["nhpi: 1 non-numeric values set to missing"]
    assert df["census_division"].iloc[0] == "3500"
    adopter = df[df["adopter"] == 1].groupby("census_division").first().iloc[0]
    rows = df[df["census_division"] == adopter.name]
    assert (rows["post"] == (rows["period"] >= adopter["adopted"])).all()
    assert (rows["event"] == rows["t"] - adopter["adopted"].ordinal).all()
    assert (df.loc[df["adopter"] == 0, "post"] == 0).all()


def test_errors(tmp_path):
    with pytest.raises(SystemExit, match=r"score \(did you mean scor\?\)"):
        B.load(write(panel().rename(columns={"Score": "Scor"}), tmp_path))
    frame = panel()
    with pytest.raises(SystemExit, match="repeat a period"):
        B.prepare(B.load(write(pd.concat([frame, frame.head(1)]), tmp_path)))
    with pytest.raises(SystemExit, match="expected an existing"):
        B.load(tmp_path / "data.txt")


def test_models_recover_the_known_slope(tmp_path):
    df = prepared(panel(), tmp_path)
    res = B.ols(df, "y_permits", ["score", "post", *map(B.clean, B.CONTROLS)])
    assert res.params["score"] == pytest.approx(SLOPE, abs=0.02)
    assert len(set(res.cov_kwds["groups"])) == 40  # clustered by census division


def test_pretrend_test(tmp_path):
    assert B.pretrend_test(prepared(panel(), tmp_path), "y_permits")["p"] > 0.05
    planted = B.pretrend_test(prepared(panel(pre_trend=0.02), tmp_path), "y_permits")
    assert planted["difference"] == pytest.approx(0.02, abs=0.005) and planted["p"] < 0.01


@pytest.mark.parametrize("name", ["data.csv", "data.xlsx"])
def test_report(tmp_path, name):
    out = tmp_path / "report.pdf"
    assert B.main([str(write(panel(), tmp_path, name)), "-o", str(out)]) == 0
    assert out.read_bytes().startswith(b"%PDF") and not list(tmp_path.glob(".*.part"))


def test_report_without_adopters_or_a_control(tmp_path):
    frame = panel().assign(**{"Treatment Date": "", "NHPI": ".."})  # nobody adopts; no NHPI
    out = tmp_path / "report.pdf"
    assert B.main([str(write(frame, tmp_path)), "-o", str(out)]) == 0
    with pytest.warns(UserWarning, match=r"left out post, nhpi \(no variation\)"):
        B.ols(prepared(frame, tmp_path), "y_starts", ["score", "post", "nhpi"])
