"""Tests for portal.py. Run: python -m pytest -q test_portal.py"""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

import portal as P


def panel(units: int = 60, periods: int = 32, pre_trend: float = 0.0, seed: int = 3):
    """A quarterly panel with known answers: slope 0.3 on the score, 0.5 on x, a DiD effect
    of 0.2 from each adopter's treatment date, and a jump of 0.5 at score 5 in RD y.
    pre_trend adds a trend (per quarter) to the adopters' permits."""
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2016Q1", periods=periods, freq="Q")
    adopt = np.where(np.arange(units) < units // 2, rng.integers(8, periods - 6, units), -1)
    each = lambda values: np.repeat(values, periods)  # noqa: E731
    t, at = np.tile(np.arange(periods), units), each(adopt)
    score, x = each(rng.uniform(0, 10, units)), rng.normal(size=units * periods)
    noise = lambda: rng.normal(0, 0.1, units * periods)  # noqa: E731
    permits = 2 + 0.3 * score + 0.5 * x + each(rng.normal(0, 0.3, units)) + noise()
    permits += 0.2 * ((at >= 0) & (t >= at)) + pre_trend * t * (at >= 0)
    adopted = [str(quarters[a].start_time.date()) if a >= 0 else "" for a in adopt]
    return pd.DataFrame(
        {
            "Census Division": each([f"CD {i:02d}" for i in range(units)]),
            "Date": np.tile(quarters.astype(str), units),
            "Score": score,
            "X": x,
            "Urban": each(rng.integers(0, 2, units)),
            "Permits": permits,
            "RD y": 1 + 0.2 * score + 0.5 * (score >= 5) + noise(),
            "Treatment date": each(adopted),
            "Notes": "<script>alert(1)</script>",
        }
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    panel().to_csv(folder / "panel.csv", index=False)
    panel().to_excel(folder / "panel.xlsx", index=False, sheet_name="data")
    panel(pre_trend=0.01).to_csv(folder / "pretrend.csv", index=False)
    (folder / "notes.txt").write_text("not data")
    (folder / "~$panel.xlsx").write_text("an Excel lock file")
    portal = P.Portal(folder, 0)
    threading.Thread(target=portal.serve_forever, daemon=True).start()
    yield portal
    portal.shutdown()
    portal.server_close()


def call(server, path, body=None, token=None, host=None, kind="application/json"):
    """POST JSON (or GET without a body); returns the status, the reply and the CSP."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": kind, "X-Token": server.token if token is None else token}
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data, headers)
    if host:
        request.add_unredirected_header("Host", host)
    try:
        with urllib.request.urlopen(request) as response:
            raw, status, headers = response.read(), response.status, response.headers
    except urllib.error.HTTPError as error:
        raw, status, headers = error.read(), error.code, error.headers
    try:
        reply = json.loads(raw)
    except json.JSONDecodeError:
        reply = raw.decode()
    return status, reply, headers.get("Content-Security-Policy")


BASE = {"file": "panel.csv", "unit": "census_division", "time": "date", "y": "permits"}


def run(server, **spec):
    status, reply, _ = call(server, "/api/run", BASE | spec)
    assert status == 200, reply
    return reply


def estimate(reply, term, table=0):
    return next(r["estimate"] for r in reply["tables"][table]["rows"] if r["term"] == term)


def test_page_and_security(server):
    status, page, csp = call(server, "/")
    assert status == 200 and server.token in page and "{{" not in page
    assert "script-src 'nonce-" in csp and "frame-ancestors 'none'" in csp
    assert call(server, "/api/files", {}, token="wrong")[0] == 403  # other websites lack it
    assert call(server, "/", host="evil.example:80")[0] == 403  # DNS rebinding
    assert call(server, "/api/files", {}, kind="text/plain")[0] == 404
    assert call(server, "/api/nope", {})[0] == 404
    for name in ("../panel.csv", "/etc/passwd", "notes.txt", "~$panel.xlsx", ["panel.csv"]):
        assert call(server, "/api/profile", {"file": name})[0] == 400  # listed files only
    assert call(server, "/api/run", BASE | {"x": ["x" * 70_000]})[0] == 413


def test_files_and_profile(server):
    _, files, _ = call(server, "/api/files", {})
    assert [f["name"] for f in files["files"]] == ["panel.csv", "panel.xlsx", "pretrend.csv"]
    _, profile, _ = call(server, "/api/profile", {"file": "panel.xlsx"})
    assert profile["sheets"] == ["data"] and profile["rows"] == 60 * 32
    kinds = {c["name"]: c["kind"] for c in profile["columns"]}
    assert kinds["score"] == "number" and kinds["date"] == "date" and kinds["notes"] == "text"
    guess = {"unit": "census_division", "time": "date", "treat": "treatment_date"}
    assert profile["guess"] == guess | {"y": "permits", "x": "score"}


def test_models_recover_known_answers(server):
    ols = run(server, model="ols", x=["score", "x"])
    assert estimate(ols, "Score") == pytest.approx(0.3, abs=0.03)
    assert "clustered" in ols["tables"][0]["title"]
    fe = run(server, model="fe", x=["x", "score"], fe=["unit", "time"])
    assert estimate(fe, "X") == pytest.approx(0.5, abs=0.02)
    assert any("Score is absorbed" in n for n in fe["notes"])
    xs = run(server, model="xs", x=["score"])
    assert estimate(xs, "Score") == pytest.approx(0.3, abs=0.05) and xs["summary"][0][1] == 60
    one = run(server, model="xs", x=["score"], xs_mode="period", xs_period="2020Q1")
    assert one["summary"][0][1] == 60 and "2020Q1" in one["tables"][0]["title"]
    rdd = run(server, model="rdd", y="rd_y", running="score", cutoff=5, bandwidth=2)
    assert estimate(rdd, "Jump at the cutoff") == pytest.approx(0.5, abs=0.1)
    assert len(rdd["charts"]) == 3 and rdd["charts"][2]["series"][0]["points"]
    summary = dict(rdd["summary"])  # units, not the 32 rows each one repeats
    assert summary["Units below the cutoff"] + summary["Units at or above"] <= 60
    assert run(server, model="rdd", y="rd_y", running="score", cutoff=5, bandwidth=1)  # few units
    assert len(run(server, model="ols", x=["urban"])["charts"][0]["series"][0]["points"]) == 2


def test_did_and_its_pretrend_test(server):
    spec = {"model": "did", "treat": "treatment_date", "controls": ["x", "score"], "lo": -6}
    did = run(server, **spec)
    assert estimate(did, "Post (treated)") == pytest.approx(0.2, abs=0.03)
    leads = [e for k, e, *_ in did["charts"][1]["series"][0]["points"] if k < -1]
    assert len(leads) == 5 and max(map(abs, leads)) < 0.1  # vs an effect of 0.2
    assert dict(did["summary"])["Pre-trend p"] > 0.05
    assert any("Score is absorbed" in n for n in did["notes"])  # constant within a unit
    planted = run(server, file="pretrend.csv", **spec)
    assert dict(planted["summary"])["Pre-trend p"] < 0.01
    wide = run(server, **spec | {"lo": -50, "hi": 50})  # narrowed to the data, quietly
    assert not any("No treated rows" in n for n in wide["notes"])


def test_wild_bootstrap_keeps_its_size():
    """With 30 clusters, a clustered Wald test of 3 coefficients rejects true nulls too often."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 20)
    p_values = []
    for _ in range(100):
        X = pd.DataFrame(rng.normal(size=(600, 4)), columns=list("abcd"))
        y = pd.Series(X["a"] + rng.normal(size=600) + rng.normal(size=30)[groups])
        p_values.append(P.wild_p(X, y, ["b", "c", "d"], groups, reps=199))
    assert 0.01 <= np.mean(np.array(p_values) < 0.05) <= 0.10


def test_bad_requests_get_clear_messages(server):
    cases = [
        ({"model": "ols"}, "Choose at least one regressor"),
        ({"model": "ols", "x": ["nope"]}, "Unknown column in x: 'nope'"),
        ({"model": "ols", "x": [1]}, "x must be a list of column names"),
        ({"model": "ols", "x": ["score"], "y": ["permits"]}, "Unknown column for the outcome"),
        ({"model": "ols", "x": ["score"], "unit": ""}, "Clustered errors need the unit"),
        ({"model": "did", "x": ["score"]}, "treatment date"),
        ({"model": "rdd", "running": "score"}, "cutoff"),
        ({"model": "rdd", "running": "score", "cutoff": 5, "bandwidth": 0}, "must be positive"),
        ({"model": "ols", "x": ["score"], "transform": "sqrt"}, "transform must be"),
        ({"model": "fe", "x": ["score"], "fe": ["unit"]}, "absorbed"),
        ({"model": "fe", "x": ["x"], "fe": []}, "Choose unit fixed effects"),
        ({"model": "xs", "x": ["x"], "xs_mode": "period", "xs_period": "1990Q1"}, "No rows"),
        ({"model": "ols", "x": ["score"], "time": "notes"}, "Cannot read '<script>"),
    ]
    for spec, message in cases:
        status, reply, _ = call(server, "/api/run", BASE | spec)
        assert status == 400 and message in reply["error"], (spec, reply)
    assert run(server, model="ols", x=["score"], unit="", se="robust")["tables"]


def test_time_index():
    assert P.time_index(pd.Series([2015, 2016]))[1] == "Y"
    assert P.time_index(pd.Series(["2015-01", "2015-04"]))[1] == "Q"
    assert P.time_index(pd.Series(["Q1 2015", "Q2 2015", " "]))[1] == "Q"
    assert P.time_index(pd.Series([1, 2, 3]))[1] == "#"
    assert P.time_index(pd.Series(["5", ""]), "#")[0].iloc[0] == 5
    with pytest.raises(P.UserError, match="several dates in a month"):
        P.time_index(pd.Series(["2015-01-01", "2015-01-02"]))
