#!/usr/bin/env python3
"""Local web portal for regressions: OLS, panel fixed effects, cross-section, DiD and RDD.

    python portal.py                      # the files in data/, at http://127.0.0.1:8765
    python portal.py --data ~/thesis/data --port 8800 --no-browser

Choose a .xlsx or .csv file, a model and its variables, and run it: coefficients,
diagnostics and charts appear together, and pinned models line up side by side.

Everything stays on this computer. The server answers only on 127.0.0.1, checks the Host
header (against DNS rebinding) and a per-session token (against other websites), reads
only .xlsx and .csv files directly inside the data folder, and runs one model at a time.
Models are built from arrays, so no column name ever reaches eval or a formula.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import re
import secrets
import threading
import traceback
import warnings
import webbrowser
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.tools import add_constant

PAGE = Path(__file__).with_name("portal.html")
MAX_MB = 200  # largest data file read
MAX_BODY = 64 * 1024  # largest request accepted
POINTS = 2000  # most points drawn in one scatter plot
MODELS = ("ols", "fe", "xs", "did", "rdd")
TRANSFORMS = {"none": "{}", "log": "log({})", "log1p": "log(1 + {})"}
TERMS = {"const": "Constant", "_post": "Post (treated)", "_jump": "Jump at the cutoff"}
ERRORS = {
    "cluster": "errors clustered by unit",
    "robust": "robust errors",
    "classical": "classical errors",
}
HINTS = {  # column-name fragments suggesting each role, most specific first
    "unit": ("census_division", "division", "municipal", "geo", "region", "unit", "cd"),
    "time": ("ref_date", "date", "period", "quarter", "year", "month", "time"),
    "treat": ("treat", "effective", "adopt", "bylaw"),
    "y": ("permit", "start"),
    "x": ("score",),
}


class UserError(Exception):
    """A problem with the request or the data; the portal shows its message."""


# --- Data ------------------------------------------------------------------------------


def clean(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_") or "column"


def unique(names: list[str]) -> Iterator[str]:
    """Number repeated names: a, a -> a, a_2."""
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        yield name if seen[name] == 1 else f"{name}_{seen[name]}"


def data_files(folder: Path) -> dict[str, Path]:
    """The .xlsx and .csv files directly in folder, without hidden or Excel lock files."""
    paths = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    ok = (".xlsx", ".csv")
    return {p.name: p for p in paths if p.suffix.lower() in ok and p.name[0] not in ".~"}


@functools.lru_cache(maxsize=8)
def sheets(path: Path, mtime: float) -> list[str]:
    from openpyxl import load_workbook  # parses the XML with defusedxml when installed

    book = load_workbook(path, read_only=True)
    try:
        return list(book.sheetnames)
    finally:
        book.close()


@functools.lru_cache(maxsize=2)
def read(path: Path, sheet: str, mtime: float) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """A sheet or CSV file with clean, unique column names, cached until the file changes.

    Requests share the cached frame, so nothing may change it in place.
    """
    if path.stat().st_size > MAX_MB * 2**20:
        raise UserError(f"{path.name} is larger than {MAX_MB} MB")
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name=sheet)
    else:
        try:
            frame = pd.read_csv(path, thousands=",", encoding="utf-8-sig")
        except UnicodeDecodeError:  # a CSV saved by Excel on Windows
            frame = pd.read_csv(path, thousands=",", encoding="cp1252")
    labels = tuple(str(c) for c in frame.columns)
    frame.columns = list(unique([clean(c) for c in labels]))
    return frame.dropna(how="all").reset_index(drop=True), labels


def to_number(values: pd.Series) -> pd.Series:
    """Numbers as floats; text such as StatCan's ".." or "x" becomes missing."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    text = values.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce").astype(float)


def to_dates(values: pd.Series) -> pd.Series:
    """Years, quarters (2015Q1, Q1 2015) and dates as timestamps; NaT where unreadable."""
    text = values.astype("string").str.strip().str.upper().str.removesuffix(".0")
    text = text.str.replace(r"^Q([1-4])\s*-?\s*(\d{4})$", r"\2Q\1", regex=True)
    text = text.str.replace(r"^(\d{4})\s*-?\s*Q([1-4])$", r"\1Q\2", regex=True)
    return pd.to_datetime(text.replace("", pd.NA), errors="coerce", format="mixed")


def kind(values: pd.Series) -> str:
    """number, date or text, judged on a sample of the values."""
    if pd.api.types.is_numeric_dtype(values):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    sample = values.dropna().head(200)
    if sample.empty:
        return "text"
    if to_number(sample).notna().mean() > 0.9:
        return "number"
    return "date" if to_dates(sample).notna().mean() > 0.9 else "text"


def time_index(values: pd.Series, freq: str | None = None) -> tuple[pd.Series, str]:
    """Periods as integer ordinals, and the frequency: Y, Q, M or # (plain integers).

    The frequency follows from the spacing of the dates unless it is given, as it is for
    treatment dates (the time column's frequency). Blank cells are missing.
    """
    filled = values.astype("string").str.strip().fillna("") != ""
    if freq == "#" or (freq is None and pd.api.types.is_numeric_dtype(values)):
        numbers = to_number(values)
        if (filled & numbers.isna()).any() or not (numbers.dropna() % 1 == 0).all():
            raise UserError(f"{values.name}: periods must be whole numbers or dates")
        if freq == "#" or not numbers.dropna().between(1000, 9999).all():
            return numbers, "#"
    dates = to_dates(values)
    if (unread := filled & dates.isna()).any():
        raise UserError(f"Cannot read {values[unread].iloc[0]!r} in {values.name} as a date")
    if freq is None:
        months = np.unique((dates.dt.year * 12 + dates.dt.month).dropna().to_numpy())
        if len(months) < dates.dropna().nunique():
            raise UserError(f"{values.name} has several dates in a month: use months or longer")
        step = int(np.gcd.reduce(np.diff(months).astype(int))) if len(months) > 1 else 12
        freq = "Y" if step % 12 == 0 else "Q" if step % 3 == 0 else "M"
    ordinals = dates.dt.to_period(freq).array.asi8.astype(float)
    ordinals[dates.isna().to_numpy()] = np.nan
    return pd.Series(ordinals, index=values.index), freq


def period_label(ordinal: float, freq: str) -> str:
    return f"{ordinal:g}" if freq == "#" else str(pd.Period(ordinal=int(ordinal), freq=freq))


def period_x(ordinals: Any, freq: str) -> np.ndarray:
    """Periods as decimal years for charts: 2015Q2 -> 2015.25."""
    if freq == "#":
        return np.asarray(ordinals, dtype=float)
    start = pd.PeriodIndex.from_ordinals(np.asarray(ordinals, dtype=int), freq=freq).start_time
    return np.asarray(start.year + (start.month - 1) / 12, dtype=float)


def profile(frame: pd.DataFrame, labels: tuple[str, ...]) -> dict:
    """Each column's kind and summary, and guesses for the unit, time and other roles."""
    columns = []
    for name, label in zip(frame.columns, labels, strict=True):
        values = frame[name]
        entry = {"name": name, "label": label, "kind": kind(values)}
        entry |= {"missing": int(values.isna().sum()), "unique": int(values.nunique())}
        if entry["kind"] == "number":
            x = to_number(values)
            entry |= {"mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max()}
        columns.append(entry)
    kinds = {"unit": "text number", "time": "date number text", "treat": "date text number"}

    def find(role: str) -> str | None:
        for word in HINTS[role]:
            for c in columns:
                avoid = role == "time" and any(w in c["name"] for w in HINTS["treat"])
                if word in c["name"] and c["kind"] in kinds.get(role, "number") and not avoid:
                    return c["name"]
        return None

    return {"rows": len(frame), "columns": columns, "guess": {r: find(r) for r in HINTS}}


# --- Requests --------------------------------------------------------------------------


class Spec:
    """A run request, checked field by field against the dataset's columns."""

    def __init__(self, request: dict, columns: list[str]) -> None:
        self.request, self.columns = request, set(columns)

    def column(self, key: str, what: str, required: bool = True) -> str | None:
        name = self.request.get(key) or None
        if name is None and required:
            raise UserError(f"Choose the {what}")
        if name is not None and not (isinstance(name, str) and name in self.columns):
            raise UserError(f"Unknown column for the {what}: {name!r}")
        return name

    def names(self, key: str) -> list[str]:
        names = self.request.get(key) or []
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise UserError(f"{key} must be a list of column names")
        if unknown := [n for n in names if n not in self.columns]:
            raise UserError(f"Unknown column in {key}: {unknown[0]!r}")
        return list(dict.fromkeys(names))

    def choice(self, key: str, options: Any, default: str) -> str:
        value = str(self.request.get(key) or default)
        if value not in options:
            raise UserError(f"{key} must be one of {', '.join(options)}")
        return value

    def number(self, key: str, what: str, default: float | None = None) -> float | None:
        value = self.request.get(key)
        if value in (None, ""):
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise UserError(f"{what} must be a number") from None
        if not math.isfinite(number):
            raise UserError(f"{what} must be a finite number")
        return number


@dataclass
class Job:
    """A checked run: the analysis columns, and the labels to show them with."""

    spec: Spec
    frame: pd.DataFrame  # the file as read (shared, so read-only)
    data: pd.DataFrame  # _y, the regressors, and _unit and _t when chosen
    xs: list[str]
    controls: list[str]
    names: dict[str, str]  # column -> label shown
    ylabel: str
    se: str
    freq: str | None
    notes: list[str] = field(default_factory=list)


def make_job(frame: pd.DataFrame, labels: tuple[str, ...], request: dict) -> Job:
    spec = Spec(request, list(frame.columns))
    model = spec.choice("model", MODELS, "ols")
    names = dict(zip(frame.columns, labels, strict=True)) | TERMS
    y = spec.column("y", "outcome")
    xs = [c for c in spec.names("x") if c != y]
    controls = [c for c in spec.names("controls") if c not in (y, *xs)]
    se = spec.choice("se", ("cluster", "robust", "classical"), "cluster")
    one_period = model == "xs" and request.get("xs_mode") == "period"
    unit = spec.column("unit", "unit column", model in ("fe", "xs", "did"))
    if se == "cluster" and not unit and model in ("ols", "rdd"):
        raise UserError("Clustered errors need the unit column: choose it, or robust errors")
    time = spec.column("time", "time column", model == "did" or one_period)
    transform = spec.choice("transform", TRANSFORMS, "none")
    notes = []
    data = pd.DataFrame(index=frame.index)
    for column in dict.fromkeys([y, *xs, *controls]):
        data[column] = to_number(frame[column])
        if bad := int((data[column].isna() & frame[column].notna()).sum()):
            notes.append(f"{names[column]}: {bad} values that are not numbers count as missing")
    data["_y"] = data[y]
    if transform != "none":
        invalid = data[y] <= 0 if transform == "log" else data[y] < 0
        if invalid.any():
            notes.append(f"The {transform} transform leaves out {int(invalid.sum())} rows")
        data["_y"] = (np.log if transform == "log" else np.log1p)(data[y].where(~invalid))
    freq = None
    if unit:
        data["_unit"] = frame[unit].astype("string").str.strip().str.removesuffix(".0")
    if time:
        data["_t"], freq = time_index(frame[time].rename(names[time]))
    if keys := [c for c in ("_unit", "_t") if c in data]:
        data = data.dropna(subset=keys).sort_values(keys)  # in time order within each unit
    if unit and time and data.duplicated(keys).any():
        raise UserError(f"Some units repeat a period of {names[time]}: one row per unit and period")
    ylabel = TRANSFORMS[transform].format(names[y])
    return Job(spec, frame, data, xs, controls, names, ylabel, se, freq, notes)


# --- Estimation ------------------------------------------------------------------------


def fit(y: pd.Series, X: pd.DataFrame, se: str, groups: Any = None, weights: Any = None) -> Any:
    """OLS (WLS with weights); errors clustered by unit, robust (HC1) or classical. Clustered
    tests use t and F with units - 1 degrees of freedom, as Stata does."""
    model = OLS(y, X) if weights is None else WLS(y, X, weights=weights)
    if se == "cluster":
        groups = pd.factorize(groups)[0]
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    return model.fit(cov_type="HC1" if se == "robust" else "nonrobust", use_t=True)


def wild_p(X: pd.DataFrame, y: pd.Series, tested: list[str], groups: Any, reps: int = 999) -> float:
    """p-value for "the tested coefficients are all zero" from a wild score bootstrap that
    flips the sign of each unit's score (Kline and Santos 2012). The clustered F test
    rejects far too often when few units are treated; this one keeps its size."""
    rest = X.drop(columns=tested).to_numpy()

    def residual(v: np.ndarray) -> np.ndarray:
        return v - rest @ np.linalg.lstsq(rest, v, rcond=None)[0]

    u, Z = residual(y.to_numpy()), residual(X[tested].to_numpy())
    codes = pd.factorize(groups)[0]
    H = np.zeros((codes.max() + 1, len(tested)))
    np.add.at(H, codes, Z * u[:, None])  # each unit's score, under the null
    A = H @ np.linalg.pinv(H.T @ H) @ H.T
    signs = np.random.default_rng(0).choice([-1.0, 1.0], (reps, len(H)))  # same p each run
    draws = np.einsum("rg,gh,rh->r", signs, A, signs)
    return float((1 + (draws >= A.sum() * (1 - 1e-9)).sum()) / (reps + 1))


def demean(frame: pd.DataFrame, keys: list[pd.Series], tol: float = 1e-9) -> pd.DataFrame:
    """Remove fixed effects by subtracting group means for each key in turn until stable:
    one round for one key or a balanced panel, a few more for an unbalanced one."""
    out = frame.astype(float)
    scale = float(out.abs().max().max()) + 1.0
    for _ in range(1000):
        before = out
        for key in keys:
            out = out - out.groupby(key.to_numpy()).transform("mean")
        if len(keys) == 1 or float((out - before).abs().max().max()) < tol * scale:
            return out
    warnings.warn("The fixed effects did not fully converge", stacklevel=2)
    return out


def varying(job: Job, frame: pd.DataFrame, columns: list[str], why: str, need: bool = True) -> list:
    """The columns that vary; the others are noted and left out."""
    keep = [c for c in columns if frame[c].std() > 1e-9 * (1 + frame[c].abs().max())]
    job.notes += [f"{job.names[c]} {why}, so it is left out" for c in columns if c not in keep]
    if need and not keep:
        raise UserError(f"Nothing left to estimate: every regressor {why}")
    return keep


def coefficients(res: Any, names: dict[str, str]) -> list[dict]:
    ci = res.conf_int()
    stats = {"estimate": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues}
    table = pd.DataFrame(stats | {"low": ci[0], "high": ci[1]})
    return [{"term": names.get(t, t)} | row for t, row in table.to_dict("index").items()]


def diagnostics(res: Any, ordered: bool) -> list[list]:
    exog = res.model.exog
    rows = [
        ["Breusch-Pagan p, constant variance", het_breuschpagan(res.resid, exog)[1]],
        ["Jarque-Bera p, normal residuals", jarque_bera(res.resid)[1]],
        ["RESET p, functional form", linear_reset(res, power=2, use_f=True).pvalue],
    ]
    if exog.shape[1] > 2:
        vif = max(variance_inflation_factor(exog, i) for i in range(1, exog.shape[1]))
        rows.append(["Largest variance inflation factor", vif])
    if ordered:
        rows.append(["Durbin-Watson, 2 = no autocorrelation", durbin_watson(res.resid)])
    return rows


def series(name: str | None, kind: str, points: Any, color: int = 1, **extra: Any) -> dict:
    return {
        "name": name,
        "kind": kind,
        "points": np.asarray(points).tolist(),
        "color": color,
    } | extra


def chart(title: str, x: str, y: str, *lines: dict, **reference: float) -> dict:
    return {"title": title, "x": x, "y": y, "series": list(lines)} | reference


def binned(x: pd.Series, y: pd.Series, bins: int = 20) -> np.ndarray:
    data = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    few = data["x"].nunique() <= bins  # then a bin per value (the values, not the column)
    key = data["x"].to_numpy() if few else pd.qcut(data["x"], bins, duplicates="drop")
    return data.groupby(key, observed=True).mean()[["x", "y"]].to_numpy()


def partial_chart(job: Job, y: pd.Series, X: pd.DataFrame, focus: str, labels: Any = None) -> dict:
    """y against focus with the other regressors partialled out, so the line's slope is
    focus's coefficient (Frisch-Waugh-Lovell): bin means, or each unit when few."""
    others, x = X.drop(columns=focus), X[focus]
    if others.shape[1]:
        y = y - OLS(y, others).fit().fittedvalues + y.mean()
        x = x - OLS(x, others).fit().fittedvalues + x.mean()
    slope, intercept = np.polyfit(x, y, 1)
    ends = np.array([x.min(), x.max()])
    line = series("fitted line", "line", np.column_stack([ends, intercept + slope * ends]), 2)
    if labels is not None and len(x) <= POINTS:
        dots = series("units", "dots", np.column_stack([x, y]), labels=[str(u) for u in labels])
    else:
        dots = series("bin means", "dots", binned(x, y))
    name, held = (
        job.names[focus],
        ", other regressors held fixed" if set(others) - {"const"} else "",
    )
    return chart(f"{job.ylabel} against {name}{held}", name, job.ylabel, dots, line)


def residual_chart(res: Any) -> dict:
    # at most POINTS rows, the same ones each run
    keep = np.sort(np.random.default_rng(0).permutation(len(res.resid))[:POINTS])
    points = np.column_stack([res.fittedvalues.to_numpy()[keep], res.resid.to_numpy()[keep]])
    dots = series("rows", "dots", points, small=True)
    return chart("Residuals against fitted values", "fitted value", "residual", dots, hline=0)


def needs_regressors(job: Job) -> list[str]:
    if not job.xs + job.controls:
        raise UserError("Choose at least one regressor or control")
    return job.xs + job.controls


def result(title: str, summary: list, tables: list, checks: list, charts: list) -> dict:
    return dict(title=title, summary=summary, tables=tables, diagnostics=checks, charts=charts)


def table(title: str, rows: list[dict]) -> dict:
    return {"title": title, "rows": rows}


def fit_ols(job: Job) -> dict:
    regs = needs_regressors(job)
    d = job.data[["_y", *regs, *(c for c in ("_unit", "_t") if c in job.data)]].dropna()
    X = add_constant(d[varying(job, d, regs, "does not vary")], has_constant="add")
    res = fit(d["_y"], X, job.se, d.get("_unit"))
    return result(
        f"OLS: {job.ylabel}",
        [["Observations", res.nobs], ["R²", res.rsquared], ["Adjusted R²", res.rsquared_adj]],
        [table(f"Coefficients ({ERRORS[job.se]})", coefficients(res, job.names))],
        diagnostics(res, ordered="_t" in d),
        [partial_chart(job, d["_y"], X, X.columns[1]), residual_chart(res)],
    )


def fit_fe(job: Job) -> dict:
    effects = job.spec.request.get("fe", ["unit", "time"])
    if not isinstance(effects, list) or not effects or not {*map(str, effects)} <= {"unit", "time"}:
        raise UserError("Choose unit fixed effects, time fixed effects or both")
    if "time" in effects and "_t" not in job.data:
        raise UserError("Time fixed effects need the time column")
    regs = needs_regressors(job)
    d = job.data[["_y", *regs, *(c for c in ("_unit", "_t") if c in job.data)]].dropna()
    keys = [d["_unit"]] * ("unit" in effects) + [d["_t"]] * ("time" in effects)
    within = demean(d[["_y", *regs]], keys)
    regs = varying(job, within, regs, "is absorbed by the fixed effects")
    res = fit(within["_y"], within[regs], "cluster", d["_unit"])
    fe = " and ".join(e for e in ("unit", "time") if e in effects)
    return result(
        f"Panel fixed effects: {job.ylabel}",
        [
            ["Observations", res.nobs],
            ["Units", d["_unit"].nunique()],
            ["Within R²", 1 - res.ssr / float((within["_y"] ** 2).sum())],
        ],
        [
            table(
                f"Coefficients ({fe} fixed effects; {ERRORS['cluster']})",
                coefficients(res, job.names),
            )
        ],
        [
            ["Jarque-Bera p, normal residuals", jarque_bera(res.resid)[1]],
            ["Durbin-Watson, 2 = no autocorrelation", durbin_watson(res.resid)],
        ],
        [partial_chart(job, within["_y"], within[regs], regs[0]), residual_chart(res)],
    )


def fit_xs(job: Job) -> dict:
    regs = needs_regressors(job)
    if job.spec.choice("xs_mode", ("mean", "period"), "mean") == "mean":
        d, how = job.data.groupby("_unit")[["_y", *regs]].mean(), "unit averages of all periods"
    else:
        wanted = pd.Series([job.spec.request.get("xs_period") or ""], name="the period")
        period = time_index(wanted, job.freq)[0].iloc[0]
        d = job.data[job.data["_t"] == period].set_index("_unit")[["_y", *regs]]
        if d.empty:
            raise UserError(f"No rows in period {wanted.iloc[0]!r}")
        how = f"period {period_label(period, job.freq)}"
    d = d.dropna()
    X = add_constant(d[varying(job, d, regs, "does not vary across units")], has_constant="add")
    se = "robust" if job.se == "cluster" else job.se  # one row per unit: nothing to cluster
    res = fit(d["_y"], X, se)
    return result(
        f"Cross-section: {job.ylabel}",
        [["Units", res.nobs], ["R²", res.rsquared], ["Adjusted R²", res.rsquared_adj]],
        [
            table(
                f"Coefficients, one row per unit: {how} ({ERRORS[se]})",
                coefficients(res, job.names),
            )
        ],
        diagnostics(res, ordered=False),
        [partial_chart(job, d["_y"], X, X.columns[1], d.index), residual_chart(res)],
    )


def fit_did(job: Job) -> dict:
    """Staggered DiD by two-way fixed effects with never-treated units as the comparison:
    the average effect after treatment, and an event study (reference: the period before)."""
    treat = job.spec.column("treat", "treatment date column")
    controls = [c for c in job.controls if c != treat]
    lo = int(job.spec.number("lo", "The window's start", -8))
    hi = int(job.spec.number("hi", "The window's end", 8))
    if not lo <= -2 < 0 <= hi:
        raise UserError("The window needs 2 or more periods before treatment and 1 after")
    d = job.data.assign(_adopted=time_index(job.frame[treat].rename(job.names[treat]), job.freq)[0])
    if (d.groupby("_unit")["_adopted"].nunique() > 1).any():
        raise UserError(f"Some units have more than one {job.names[treat]}")
    d["_adopted"] = d.groupby("_unit")["_adopted"].transform("first")
    early = d["_adopted"] <= d.groupby("_unit")["_t"].transform("min")
    if n := d.loc[early, "_unit"].nunique():
        job.notes.append(f"{n} units treated in their first period or before are left out")
    d = d[~early].dropna(subset=["_y", *controls]).copy()
    d.loc[d["_adopted"] > d["_t"].max(), "_adopted"] = np.nan  # treated after the data end
    counts = d.groupby(d["_adopted"].notna())["_unit"].nunique()
    if counts.get(True, 0) < 2 or counts.get(False, 0) < 2:
        raise UserError("DiD needs at least 2 treated and 2 never-treated units")
    raw = d["_t"] - d["_adopted"]
    lo, hi = max(lo, int(raw.min())), min(hi, int(raw.max()))  # no wider than the data
    if lo > -2:
        raise UserError("The treated units need 2 or more periods before treatment")
    event = raw.clip(lo, hi)
    d["_post"] = (event >= 0).astype(float)
    ks = [k for k in range(lo, hi + 1) if k != -1 and (event == k).any()]
    if missing := [str(k) for k in range(lo, hi + 1) if k != -1 and k not in ks]:
        job.notes.append(f"No treated rows at t = {', '.join(missing)}, so they are left out")
    events = pd.DataFrame({f"_e{k}": (event == k).astype(float) for k in ks}, index=d.index)
    columns = pd.concat([d[["_y", "_post", *controls]], events], axis=1)
    within = demean(columns, [d["_unit"], d["_t"]])
    controls = varying(job, within, controls, "is absorbed by the fixed effects", need=False)
    res = fit(within["_y"], within[["_post", *controls]], "cluster", d["_unit"])
    es = fit(within["_y"], within[[*events, *controls]], "cluster", d["_unit"])
    leads = [f"_e{k}" for k in ks if k < -1]
    pre_p = wild_p(within[[*events, *controls]], within["_y"], leads, d["_unit"]) if leads else None
    edge = {lo: " or earlier", hi: " or later"}
    names = job.names | {f"_e{k}": f"t = {k}{edge.get(k, '')}" for k in ks}
    rows = coefficients(es, names)[: len(ks)]
    intervals = [[k, r["estimate"], r["low"], r["high"]] for k, r in zip(ks, rows, strict=True)]
    means = d.groupby([d["_adopted"].notna(), "_t"])["_y"].mean()
    trends = []
    for flag, name, color in ((True, "treated units", 1), (False, "never treated", 2)):
        m = means[flag]
        labels = [period_label(t, job.freq) for t in m.index]
        points = np.column_stack([period_x(m.index, job.freq), m])
        trends.append(series(name, "line", points, color, labels=labels))
    job.notes.append(
        "Two-way fixed effects can be biased when effects change over time or differ by "
        "treatment date; project.py has did2s and dCDH, which are not"
    )
    return result(
        f"Difference-in-differences: {job.ylabel}",
        [
            ["Observations", res.nobs],
            ["Treated units", counts[True]],
            ["Never-treated units", counts[False]],
            ["Within R²", 1 - res.ssr / float((within["_y"] ** 2).sum())],
            ["Pre-trend p", pre_p],
        ],
        [
            table(
                f"Average effect after treatment (two-way FE; {ERRORS['cluster']})",
                coefficients(res, names),
            ),
            table("Event study: the effect by period since treatment (reference t = -1)", rows),
        ],
        [["Pre-trend p: no effects before treatment (wild bootstrap by unit)", pre_p]],
        [
            chart(
                f"Mean {job.ylabel}, treated and never-treated units", "period", job.ylabel, *trends
            ),
            chart(
                "Event study, 95% intervals",
                "periods since treatment",
                "effect",
                series("estimate", "errors", intervals),
                series("reference (t = -1)", "dots", [[-1, 0]], 0),
                hline=0,
                vline=-0.5,
            ),
        ],
    )


def rd(d: pd.DataFrame, h: float, kernel: str, order: int, controls: list[str], se: str) -> Any:
    """Local polynomial regression within h of the cutoff; _jump is the discontinuity."""
    inside = d[d["_r"].abs() < h]
    above = (inside["_r"] >= 0).astype(float)
    if min(above.sum(), len(above) - above.sum()) < order + 3:
        raise UserError(f"Too few rows on one side of the cutoff within bandwidth {h:.4g}")
    X = pd.DataFrame({"_jump": above})
    for p in range(1, order + 1):
        X[f"_d{p}"], X[f"_a{p}"] = inside["_r"] ** p, above * inside["_r"] ** p
    X = add_constant(X.join(inside[controls]), has_constant="add")
    weights = 1 - inside["_r"].abs() / h if kernel == "triangular" else None
    return fit(inside["_y"], X, se, inside.get("_unit"), weights)


def jump(d: pd.DataFrame, h: float, kernel: str, order: int, controls: list, se: str) -> dict:
    """The jump at the cutoff as a coefficient row."""
    return coefficients(rd(d, h, kernel, order, controls, se), {})[1]


def fit_rdd(job: Job) -> dict:
    """Sharp RD: local linear or quadratic fits on each side of the cutoff."""
    running = job.spec.column("running", "running variable")
    if (cutoff := job.spec.number("cutoff", "The cutoff")) is None:
        raise UserError("Enter the cutoff")
    kernel = job.spec.choice("kernel", ("triangular", "uniform"), "triangular")
    order = int(job.spec.choice("order", ("1", "2"), "1"))
    controls = [c for c in job.controls if c != running]
    keep = ["_y", "_r", *controls, *(["_unit"] if "_unit" in job.data else [])]
    d = job.data.assign(_r=to_number(job.frame[running]) - cutoff)[keep].dropna()
    h = job.spec.number("bandwidth", "The bandwidth", float(d["_r"].std()))
    if not h > 0:
        raise UserError("The bandwidth must be positive")
    controls = varying(job, d, controls, "does not vary", need=False)
    args = (kernel, order, controls, job.se)
    res = rd(d, h, *args)
    names = job.names | {f"_d{p}": "distance" + "²" * (p - 1) for p in (1, 2)}
    names |= {f"_a{p}": f"{names[f'_d{p}']} x above" for p in (1, 2)}
    fitted = f"Local {('linear', 'quadratic')[order - 1]} fit, {kernel} kernel, bandwidth {h:.4g}"
    tables = [table(f"{fitted} ({ERRORS[job.se]})", coefficients(res, names))]
    placebo = [
        {"term": job.names[c]} | jump(d.assign(_y=d[c]), h, *args[:2], [], job.se) for c in controls
    ]
    if placebo:
        tables.append(table("Placebo: jumps in the controls at the cutoff (expect none)", placebo))
    sensitivity = []
    for factor in (0.5, 0.75, 1, 1.25, 1.5, 2):
        with contextlib.suppress(UserError):  # too few rows at the narrowest bandwidths
            r = jump(d, factor * h, *args)
            sensitivity.append([factor * h, r["estimate"], r["low"], r["high"]])
    # panel rows repeat a unit's running value: count each unit once per value, not each row
    who = "Units" if "_unit" in d else "Rows"
    distinct = d.drop_duplicates(["_unit", "_r"]) if "_unit" in d else d
    near = distinct.loc[distinct["_r"].abs() < h / 4, "_r"]
    density_p = binomtest(int((near >= 0).sum()), len(near)).pvalue if len(near) else None
    inside = distinct[distinct["_r"].abs() < h]

    def count(rows: pd.DataFrame) -> int:
        return rows["_unit"].nunique() if "_unit" in rows else len(rows)

    window = d[d["_r"].abs() <= 2 * h]
    sides = (window[window["_r"] < 0], window[window["_r"] >= 0])
    dots = np.vstack([binned(s["_r"] + cutoff, s["_y"], 15) for s in sides])
    level = res.params["const"] + (res.params[controls] * d[controls].mean()).sum()
    fits = []
    for side in (-1.0, 1.0):
        grid, above = np.linspace(0, side * h, 30), float(side > 0)
        slope = sum(
            (res.params[f"_d{p}"] + above * res.params[f"_a{p}"]) * grid**p
            for p in range(1, order + 1)
        )
        points = np.column_stack([grid + cutoff, level + above * res.params["_jump"] + slope])
        fits.append(series("local fit" if side < 0 else None, "line", points, 2))
    counts, edges = np.histogram(
        distinct.loc[distinct["_r"].abs() <= 2 * h, "_r"] + cutoff, bins=30
    )
    bars = np.column_stack([(edges[:-1] + edges[1:]) / 2, counts])
    label, y = job.names[running], job.ylabel
    return result(
        f"Regression discontinuity: {y} at {label} = {cutoff:g}",
        [
            ["Rows used", res.nobs],
            [f"{who} below the cutoff", count(inside[inside["_r"] < 0])],
            [f"{who} at or above", count(inside[inside["_r"] >= 0])],
            ["Bandwidth", h],
        ],
        tables,
        [
            [f"Density test p, {who.lower()} just below vs above (within h/4)", density_p],
            ["Jarque-Bera p, normal residuals", jarque_bera(res.resid)[1]],
        ],
        [
            chart(
                f"{y} by {label}: bin means and the local fit",
                label,
                y,
                series("bin means", "dots", dots),
                *fits,
                vline=cutoff,
            ),
            chart(
                f"{who} by {label} near the cutoff (bunching check)",
                label,
                who.lower(),
                series(who.lower(), "bars", bars, width=edges[1] - edges[0]),
                vline=cutoff,
            ),
            chart(
                "The jump at other bandwidths, 95% intervals",
                "bandwidth",
                "jump",
                series("jump", "errors", sensitivity),
                hline=0,
                vline=h,
            ),
        ],
    )


FITS = {"ols": fit_ols, "fe": fit_fe, "xs": fit_xs, "did": fit_did, "rdd": fit_rdd}


# --- Server ----------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Plain JSON: floats to 6 significant digits, NaN and infinity as null."""
    if isinstance(value, float | np.floating):
        return float(f"{value:.6g}") if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return value


class Portal(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, folder: Path, port: int) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.folder, self.token, self.lock = folder, secrets.token_urlsafe(32), threading.Lock()
        self.port = self.server_address[1]
        self.hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def load(self, request: dict) -> tuple[pd.DataFrame, tuple[str, ...], dict]:
        files = data_files(self.folder)
        name = request.get("file")
        if not isinstance(name, str) or name not in files:  # a listed name only: no paths
            raise UserError("Choose a file from the list")
        path = files[name]
        book = sheets(path, path.stat().st_mtime) if path.suffix.lower() == ".xlsx" else []
        sheet = (request.get("sheet") or book[0]) if book else ""
        if sheet not in [*book, ""]:
            raise UserError(f"No sheet {sheet!r} in {name}")
        try:
            frame, labels = read(path, sheet, path.stat().st_mtime)
        except UserError:
            raise
        except Exception as exc:  # pandas and openpyxl raise many types for unreadable files
            raise UserError(f"Cannot read {name}: {exc}") from None
        return frame, labels, {"file": name, "sheets": book, "sheet": sheet}

    def files(self, request: dict) -> dict:
        found = data_files(self.folder).items()
        return {
            "folder": str(self.folder),
            "files": [{"name": n, "bytes": p.stat().st_size} for n, p in found],
        }

    def profile(self, request: dict) -> dict:
        frame, labels, about = self.load(request)
        return about | profile(frame, labels)

    def run(self, request: dict) -> dict:
        frame, labels, _ = self.load(request)
        with self.lock, warnings.catch_warnings(record=True) as caught:  # one model at a time
            warnings.simplefilter("always")
            job = make_job(frame, labels, request)
            result = FITS[job.spec.choice("model", MODELS, "ols")](job)
        messages = [
            " ".join(str(w.message).split())
            for w in caught
            if not issubclass(w.category, DeprecationWarning | FutureWarning)
        ]
        return result | {"notes": list(dict.fromkeys(job.notes + messages))}


ROUTES = {"/api/files": "files", "/api/profile": "profile", "/api/run": "run"}


class Handler(BaseHTTPRequestHandler):
    server: Portal
    server_version, sys_version = "portal", ""

    def log_message(self, format: str, *args: Any) -> None:  # quiet: errors are printed
        pass

    def do_GET(self) -> None:
        if not self.trusted():
            return
        if self.path != "/":
            return self.send(404, b"Not found", "text/plain")
        nonce = secrets.token_urlsafe(16)
        page = PAGE.read_text(encoding="utf-8").replace("{{nonce}}", nonce)
        csp = (
            f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
        self.send(
            200,
            page.replace("{{token}}", self.server.token).encode(),
            "text/html; charset=utf-8",
            csp,
        )

    def do_POST(self) -> None:
        if not self.trusted():
            return
        if not secrets.compare_digest(self.headers.get("X-Token", ""), self.server.token):
            return self.send(403, b'{"error": "Missing or wrong token: reload the page"}')
        if self.path not in ROUTES or self.headers.get_content_type() != "application/json":
            return self.send(404, b'{"error": "Not found"}')
        if (length := int(self.headers.get("Content-Length") or 0)) > MAX_BODY:
            return self.send(413, b'{"error": "Request too large"}')
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(request, dict):
                raise UserError("The request must be a JSON object")
            status, result = 200, getattr(self.server, ROUTES[self.path])(request)
        except (UserError, ValueError) as exc:  # ValueError covers malformed JSON
            status, result = 400, {"error": str(exc)}
        except Exception as exc:  # shown in the page; the traceback goes to the terminal
            traceback.print_exc()
            status, result = 500, {"error": f"Unexpected error: {type(exc).__name__}: {exc}"}
        self.send(status, json.dumps(jsonable(result), allow_nan=False).encode())

    def trusted(self) -> bool:
        """Only this computer's browser: the Host header must name 127.0.0.1 or localhost."""
        if self.headers.get("Host") in self.server.hosts:
            return True
        self.send(403, b"Forbidden", "text/plain")
        return False

    def send(self, status: int, body: bytes, kind: str = "application/json", csp: str = "") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if csp:
            self.send_header("Content-Security-Policy", csp)
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="folder of .xlsx/.csv files",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="port on 127.0.0.1; 0 picks a free one"
    )
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    folder = args.data.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder: create it or pass --data")
    server = Portal(folder, args.port)
    url = f"http://127.0.0.1:{server.port}/"
    print(f"Regression portal: {url} (files in {folder}; Ctrl+C stops it)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
