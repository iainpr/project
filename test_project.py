"""Tests for project.py. Run: python -m pytest -q (about a minute; the DiD runs are slow)."""

import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

import demo
import project as P


@pytest.fixture(scope="module")
def demo_config(tmp_path_factory):
    return demo.write_demo(tmp_path_factory.mktemp("demo"))


@pytest.fixture(scope="module")
def study(demo_config, tmp_path_factory):
    loaded = P.Study.load(demo_config)
    output = tmp_path_factory.mktemp("output")
    return P.Study(P.replace(loaded.settings, output=output), loaded.sources)


@pytest.fixture(scope="module")
def bylaws(demo_config):
    return pd.read_csv(demo_config.parent / "zoning_bylaws.csv")


def config_with(demo_config, tmp_path, **changes):
    """The demo config with some lines replaced, saved next to the demo data."""
    text = demo_config.read_text(encoding="utf-8")
    for old, new in changes.items():
        assert old in text
        text = text.replace(old, new)
    path = demo_config.parent / f"{tmp_path.name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- Settings ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("FVI_CSCE_AB", "fvi_csce_ab"),
        ("Median income", "median_income"),
        ("  Pop. density (km²) ", "pop_density_km"),
        ("2015", "v2015"),
        ("class", "class_"),
        ("fvi_csce_{region}", "fvi_csce_{region}"),
    ],
)
def test_clean(raw, cleaned):
    assert P.clean(raw) == cleaned


def test_parse_model_takes_names_only():
    assert P.parse_model("Permits ~ Population + pop_density") == (
        "permits",
        ["population", "pop_density"],
    )
    # Nothing is evaluated: code turns into a (missing) column name
    assert P.parse_model("y ~ __import__('os').system('x')")[1] == ["import_os_system_x"]
    for bad in ["permits", "a ~ b ~ c", "permits ~ "]:
        with pytest.raises(P.UserError):
            P.parse_model(bad)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('unit = "municipality"\n', "", "missing the setting 'unit'"),
        ('transform = "log"', 'transform = "sqrt"', "transform must be"),
        ('use = "effective"', 'use = "both"', "use must be"),
        ("window_quarterly = [-8, 12]", "window_quarterly = [0, 12]", "must start below -1"),
        ("window_annual = [-4, 6]", 'window_annual = "wide"', "two numbers"),
        ('"permits ~ population",', '"permits",', "should look like"),
        ("[outcomes]", "[outcomes", "not valid TOML"),
    ],
)
def test_settings_errors(demo_config, tmp_path, old, new, message):
    with pytest.raises(P.UserError, match=message):
        P.load_settings(config_with(demo_config, tmp_path, **{old: new}))


def test_missing_config(tmp_path):
    with pytest.raises(P.UserError, match="not found"):
        P.load_settings(tmp_path / "config.toml")


# --- Reading data -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "freq", "first"),
    [
        ([2015, 2016], "Y", "2015"),
        (["2015Q1", "2015Q2"], "Q", "2015Q1"),
        (["Q3 2015", "Q4 2015"], "Q", "2015Q3"),
        (["2015-01", "2015-04", "2015-07"], "Q", "2015Q1"),  # StatCan quarterly REF_DATE
        (["2015-01-01", "2015-02-01", None], "M", "2015-01"),
        (["2015-01-05", "2015-01-06"], "D", "2015-01-05"),
        (["2015-12-31", "2016-12-31"], "Y", "2015"),
    ],
)
def test_to_periods(values, freq, first):
    periods, found = P.to_periods(pd.Series(values))
    assert found == freq
    assert str(periods.iloc[0]) == first


@pytest.mark.parametrize("values", [[2015.5, 2016], ["soon", "2015-01-01"], [None, None]])
def test_to_periods_errors(values):
    with pytest.raises(P.UserError):
        P.to_periods(pd.Series(values, dtype="object" if None in values else None))


def test_valet_download_is_read_as_is(study):
    src = next(s for s in study.sources if s.path.name == "boc_valet.csv")
    assert (src.kind, src.freq) == ("series", "M")
    assert {"policy_rate", "fvi_csce_atl"} <= set(src.frame.columns)  # V39079 renamed


def test_read_source_problems(tmp_path, study):
    s = study.settings
    cases = {
        "repeat.csv": (
            "municipality,year,starts\nA,2015,1\nA,2015,2\n",
            "share a municipality and year",
        ),
        "reserved.csv": ("municipality,year,treat\nA,2015,1\n", "rename the column"),
        "blank.csv": ("municipality,year,starts\nA,2015,1\n,2016,2\n", "rows have no municipality"),
        "nothing.csv": ("name,value\nA,1\n", "neither the unit column"),
        "dates.csv": ("municipality,date\nA,2015-01-01\nA,someday\n", "cannot read 'someday'"),
        "table.txt": ("x\n", "use .csv or .xlsx"),
    }
    for name, (text, message) in cases.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(P.UserError, match=message):
            P.read_source(path, s)
    with pytest.raises(P.UserError, match="not found"):
        P.read_source(tmp_path / "missing.csv", s)


def test_numeric_unit_ids_match_across_files(tmp_path, study):
    path = tmp_path / "ids.csv"
    path.write_text("municipality,year,v\n3520005,2015,1\n,,\n3520006.0,2015,2\n", encoding="utf-8")
    ids = P.read_source(path, study.settings).frame["municipality"].tolist()
    assert ids == ["3520005", "3520006"]


def test_excel_files(tmp_path, study):
    path = tmp_path / "permits.xlsx"
    pd.DataFrame({"Municipality": ["A", "A"], "Year": [2015, 2016], "Permits": [3, 4]}).to_excel(
        path, index=False
    )
    src = P.read_source(path, study.settings)
    assert (src.kind, src.freq, list(src.frame.columns)) == (
        "panel",
        "Y",
        ["municipality", "permits", "_period"],
    )


def test_missing_column_suggests_names(study):
    with pytest.raises(P.UserError, match="Did you mean population"):
        study.find("populaton")
    with pytest.raises(P.UserError, match="needs to be in a municipal panel"):
        study.find("policy_rate", ("panel",))


# --- Panels -----------------------------------------------------------------------------


def test_lookup_averages_finer_and_repeats_coarser_data():
    panel = pd.DataFrame(
        {"_unit": ["A"] * 4, "_period": pd.period_range("2015Q1", periods=4, freq="Q")}
    )
    monthly = pd.DataFrame(
        {"_period": pd.period_range("2015-01", periods=12, freq="M"), "value": np.arange(12.0)}
    )
    assert P.lookup(panel, "Q", monthly, "M", []).tolist() == [1, 4, 7, 10]
    annual = pd.DataFrame(
        {"_unit": ["A"], "_period": pd.PeriodIndex(["2015"], freq="Y"), "value": [5.0]}
    )
    assert P.lookup(panel, "Q", annual, "Y", ["_unit"]).tolist() == [5.0] * 4


def test_treatment_timing(study, bylaws):
    p = study.panel("starts", "Q")
    d = p.data
    first = d.groupby("_unit")["first_id"].first()
    # Bylaws in effect before the data start: left out, not used as untreated rows
    early = bylaws.loc[pd.to_datetime(bylaws["effective_date"]) < "2010-01-01", "municipality"]
    assert set(p.excluded) == set(early) and not d["_unit"].isin(early).any()
    # Two bylaws: the earlier one starts treatment
    twice = bylaws["municipality"][bylaws["municipality"].duplicated()].iloc[0]
    earliest = pd.to_datetime(bylaws.loc[bylaws["municipality"] == twice, "effective_date"]).min()
    assert p.period(first[twice]) == pd.Period(earliest, "Q")
    # Passed but not yet in effect: never treated on the effective date, treated on the vote
    pending = bylaws.loc[bylaws["effective_date"].isna(), "municipality"].iloc[0]
    assert np.isnan(first[pending])
    by_vote = study.panel("starts", "Q", treatment_date="vote")
    vote_first = by_vote.data.groupby("_unit")["first_id"].first()
    assert by_vote.period(vote_first[pending]) == pd.Period("2024Q4")
    # treat and rel_time follow first_id; never-treated rows sit at the reference period
    assert (d["treat"] == (d["period_id"] >= d["first_id"])).all()
    assert d["rel_time"].between(*p.window).all()
    assert (d.loc[d["first_id"].isna(), "rel_time"] == -1).all()
    assert p.counts() == {
        "municipalities": 58,
        "adopt a bylaw": 38,
        "never treated": 20,
        "left out": 2,
    }


def test_controls(study, demo_config):
    p = study.panel("starts", "Q")
    # National series are absorbed by period fixed effects, and the notes say so
    assert {"cpi_shelter", "policy_rate"}.isdisjoint(p.controls)
    assert any(note.startswith("policy_rate is the same") for note in p.notes)
    assert {"pop_growth", "population", "market_barometer"} <= set(p.controls)
    assert p.data[list(p.all_controls)].notna().all().all()
    # Atlantic provinces use fvi_csce_atl, through [regions]
    path = demo_config.parent / "boc_valet.csv"
    valet = pd.read_csv(path, skiprows=P.valet_preamble(path)).dropna(subset=["FVI_CSCE_ATL"])
    atl = pd.Series(valet["FVI_CSCE_ATL"].to_numpy(), pd.PeriodIndex(valet["date"], freq="Q"))
    rows = p.data[p.data["_region"] == "ATL"].groupby("_period")["market_barometer"].first()
    assert len(rows) == 60
    assert np.corrcoef(rows, atl.reindex(rows.index))[0, 1] > 0.999


def test_monthly_outcome_adds_up_to_complete_quarters(tmp_path, study):
    path = tmp_path / "monthly.csv"
    months = pd.period_range("2015-01", "2015-08", freq="M").strftime("%Y-%m-01")
    pd.DataFrame({"municipality": "A", "date": months, "starts": 1}).to_csv(path, index=False)
    s = P.replace(study.settings, files=(path,), controls=())
    p = P.Study(s, [P.read_source(path, s)]).panel("starts", "Q", timing=False, transform="none")
    assert p.data["starts"].tolist() == [3, 3]  # 2015Q3 has only two months
    assert "1 quarterly totals of starts are left out" in p.notes[0]


# --- Sections ---------------------------------------------------------------------------


def test_sections_on_demo_data(study):
    s1 = P.section1(study, "starts", "Q")
    tables = dict(s1.tables)
    tests = tables["Parallel trends: are the pre-adoption estimates jointly zero?"]
    assert (tests["p_value"] > 0.05).all()  # the demo has no pre-trends
    robust = tables["Placebo and robustness: average effect after adoption"]
    placebo = robust.loc[robust.index.str.startswith("Placebo")].iloc[0]
    assert placebo["ci_low"] < 0 < placebo["ci_high"]
    assert len(robust) == 8 and robust["estimate"].notna().all()

    s2 = P.section2(study)
    ols = dict(s2.tables)["OLS estimates"]
    assert len(ols) == 6
    assert ols["estimate"].between(0.5, 1.5).all()  # starts and permits scale with population

    s3 = P.did_section(study, 3, "starts", "Q")
    s4 = P.did_section(study, 4, "permits", "Y")
    for result, window in ((s3, (-8, 12)), (s4, (-4, 6))):
        events, average = (frame for _, frame in result.tables[:2])
        assert events.index.min() == window[0] and events.index.max() == window[1]
        assert {"TWFE estimate", "did2s estimate", "dCDH estimate"} <= set(events.columns)
        # The demo's true effect averages about 0.1 (3-17% by bylaw score)
        assert average.loc["did2s (Gardner)", "estimate"] == pytest.approx(0.1, abs=0.05)
        assert average["ci_low"].gt(0).all()

    folder = P.save([s1, s2, s3, s4], study, "all sections")
    files = {f.name for f in folder.iterdir()}
    assert {"index.html", "run_info.json", "s3_starts_event_study.csv", "ols.tex"} <= files
    assert {"s3_starts_event_study.png", "s2_log_permits_vs_log_population.png"} <= files
    info = (folder / "run_info.json").read_text(encoding="utf-8")
    assert '"data_sha256"' in info and f'"program": "project.py {P.__version__}"' in info


def test_html_is_escaped(study):
    r = P.Result("x", "<b>title</b>", notes=["<script>alert(1)</script>"])
    r.tables.append(("<i>t</i>", pd.DataFrame({"a": ["<img src=x>"]})))
    page = (P.save([r], study, "escape") / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in page and "<img src=x>" not in page and "<b>title" not in page


def test_section2_reports_a_missing_column(study):
    s = P.replace(study.settings, ols_models=("permits ~ populaton",))
    with pytest.raises(P.UserError, match="Did you mean population"):
        P.section2(P.Study(s, study.sources))


# --- Menu and command line --------------------------------------------------------------


def run_menu(config, keys):
    out = io.StringIO()
    app = P.App(config, Console(file=out, width=120), stream=io.StringIO(keys))
    code = app.main()
    return code, out.getvalue(), app


def test_menu_session(demo_config, tmp_path):
    output = tmp_path.as_posix()
    config = config_with(
        demo_config, tmp_path, **{'folder = "../../output"': f'folder = "{output}"'}
    )
    # x: invalid choice; s 2 6 Enter b: switch off cpi_shelter; 2 n: preview only; 2 y: run
    code, text, app = run_menu(config, "x\ns\n2\n6\n\nb\n2\nn\n2\ny\nq\n")
    assert code == 0
    assert "Data check" in text and "Please type one of" in text
    assert "cpi_shelter" not in app.study.settings.controls
    assert text.count("Models (logs)") == 2 and "OLS estimates" in text and "Saved" in text
    assert len(list(tmp_path.glob("*_section2/index.html"))) == 1


def test_menu_reports_problems(demo_config, tmp_path):
    code, text, _ = run_menu(tmp_path / "nowhere.toml", "q\n")
    assert code == 1 and "Cannot load the data" in text
    bad = config_with(demo_config, tmp_path, **{'"pop_growth",': '"pop_growht",'})
    code, text, _ = run_menu(bad, "3\nq\n")
    assert code == 0 and "Did you mean pop_growth" in text and "✗" in text
    # A bad model typed in the settings menu is refused, and the session goes on
    code, text, _ = run_menu(demo_config, "s\n3\npermits\nb\nq\n")
    assert code == 0 and "should look like" in text


def test_command_line(demo_config, tmp_path):
    output = tmp_path.as_posix()
    config = config_with(
        demo_config, tmp_path, **{'folder = "../../output"': f'folder = "{output}"'}
    )
    assert P.main(["--config", str(config), "--run", "2"]) == 0
    assert P.main(["--config", str(tmp_path / "nowhere.toml"), "--run", "2"]) == 1
