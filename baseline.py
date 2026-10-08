#!/usr/bin/env python3
"""Baseline OLS of building permits and housing starts on the internal zoning score.

    python baseline.py data/final.xlsx        # writes data/final_baseline.pdf
    python baseline.py data/final.csv -o report.pdf

The data are one table with a row per census division and period; set the column
names under SETTINGS (matching ignores case, spaces and punctuation). Models, with
standard errors clustered by census division:

    (1) permits ~ score        (3) permits ~ score + post + controls
    (2) starts ~ score         (4) starts ~ score + post + controls

post is 1 from the period the bylaw took effect. The PDF holds summary statistics,
correlations, trends, pre-trends, the score plots, the models with diagnostics and
residual plots. The models measure associations: a first look at whether a DiD or
RDD design is worth building.
"""

from __future__ import annotations

import argparse
import difflib
import re
import textwrap
import warnings
from datetime import date
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.tools import add_constant

# --- SETTINGS: the dataset's column names --------------------------------------------
UNIT = "census_division"  # census division id
TIME = "date"  # dates, years or quarters such as 2015Q1
FREQ = "Q"  # the data's frequency: "M", "Q" or "Y"
SCORE = "score"  # internal zoning score
OUTCOMES = ["permits", "starts"]  # Table 34-10-0292-01 permits; CMHC starts
TREATED = "treatment_date"  # date the bylaw took effect; blank if it never did
CONTROLS = [
    "mortgage_lending",
    "development_charges",
    "bcpi",  # building construction price index
    "nhpi",  # new housing price index
    "cpi_shelter",
    "unemployment_rate",
]
EXTRA = ["completions", "under_construction"]  # in the summary statistics only
LOG = True  # model log(1 + outcome), which keeps periods with zero
CUTOFF = None  # a score threshold to mark in the score plots (RDD check), e.g. 50
WINDOW = 8  # periods either side of adoption in the pre-trend plots
MAX_MB = 200  # larger files are refused, to protect memory
# ---------------------------------------------------------------------------------------

PAGE = (11, 8.5)  # US Letter, landscape
LINES = 50  # text lines per page
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"  # colour-vision-safe pair; axis grey
DOTS = {"marker": "o", "ms": 6, "mec": "white", "mew": 1, "color": BLUE}
STYLE = {
    "font.size": 8,
    "axes.titlelocation": "left",
    "axes.edgecolor": GREY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embedded TrueType: the PDF's text can be searched and copied
}


def clean(name: object) -> str:
    """Lower case, other characters as "_": "CPI Shelter" becomes "cpi_shelter"."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def load(path: Path) -> pd.DataFrame:
    """Read only the SETTINGS columns of a .csv or .xlsx file, with cleaned names."""
    if path.suffix.lower() not in (".csv", ".xlsx") or not path.is_file():
        raise SystemExit(f"{path}: expected an existing .csv or .xlsx file")
    if path.stat().st_size > MAX_MB * 2**20:
        raise SystemExit(f"{path.name} is larger than {MAX_MB} MB (see MAX_MB)")
    wanted = {clean(c) for c in [UNIT, TIME, SCORE, TREATED, *OUTCOMES, *CONTROLS, *EXTRA]}
    seen: list[str] = []

    def keep(column: object) -> bool:  # pandas asks once per column; only these load
        seen.append(str(column))
        return clean(column) in wanted

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, usecols=keep)  # first sheet; defusedxml guards the XML
    else:
        try:
            df = pd.read_csv(path, usecols=keep, thousands=",", encoding="utf-8-sig")
        except UnicodeDecodeError:  # a CSV saved by Excel on Windows
            df = pd.read_csv(path, usecols=keep, thousands=",", encoding="cp1252")
    df.columns = [clean(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise SystemExit(f"{path.name}: two columns have the same name after cleaning")
    if missing := sorted(wanted - set(df.columns)):
        names = list(dict.fromkeys(seen))
        near = {m: difflib.get_close_matches(m, map(clean, names), n=1) for m in missing}
        hint = "; ".join(f"{m} (did you mean {n[0]}?)" if n else m for m, n in near.items())
        raise SystemExit(f"Missing column(s): {hint}\nIn {path.name}: {', '.join(names)}")
    return df


def to_period(values: pd.Series) -> pd.Series:
    """Years, quarters ("2015Q1", "Q1 2015") or dates as FREQ periods; blanks stay NaT."""
    text = values.astype("string").str.strip().str.upper().str.removesuffix(".0")
    text = text.str.replace(r"^Q([1-4])\s*(\d{4})$", r"\2Q\1", regex=True).replace("", pd.NA)
    dates = pd.to_datetime(text, errors="coerce", format="mixed")
    if (unread := text.notna() & dates.isna()).any():
        raise SystemExit(f"Cannot read {text[unread].iloc[0]!r} in {values.name} as a date")
    return dates.dt.to_period(FREQ)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Check the panel, make numbers numeric and add the model columns.

    Adds period, t (its ordinal), adopted (the period the bylaw took effect), adopter,
    event (periods since adoption), post and y_<outcome> (the outcome as modelled).
    Returns the data sorted by division and period, and notes on values set missing.
    """
    unit = clean(UNIT)
    df[unit] = df[unit].astype("string").str.strip().str.removesuffix(".0")  # 3520.0 -> 3520
    df["period"] = to_period(df[clean(TIME)])
    if df[unit].isna().any() or df["period"].isna().any():
        raise SystemExit(f"Every row needs a {UNIT} and a {TIME}")
    if df.duplicated([unit, "period"]).any():
        raise SystemExit(f"Some {UNIT}s repeat a period: is FREQ = {FREQ!r} right?")
    adopted = to_period(df[clean(TREATED)]).groupby(df[unit])
    if (adopted.nunique() > 1).any():
        raise SystemExit(f"Some {UNIT}s have more than one {TREATED}")
    df["adopted"] = adopted.transform("first")
    df["adopter"] = df["adopted"].notna().astype(int)
    df["t"] = df["period"].array.asi8
    start = np.where(df["adopter"] == 1, df["adopted"].array.asi8, np.nan)
    df["event"] = df["t"] - start
    df["post"] = (df["event"] >= 0).astype(int)
    notes = []
    for column in map(clean, [SCORE, *OUTCOMES, *CONTROLS, *EXTRA]):
        values = pd.to_numeric(df[column], errors="coerce")
        if bad := int((values.isna() & df[column].notna()).sum()):  # StatCan "..", "x"
            notes.append(f"{column}: {bad} non-numeric values set to missing")
        df[column] = values
    for column in map(clean, OUTCOMES):
        df[f"y_{column}"] = np.log1p(df[column].where(df[column] >= 0)) if LOG else df[column]
    return df.sort_values([unit, "period"], ignore_index=True), notes


def label(outcome: str) -> str:
    return f"log(1 + {outcome})" if LOG else outcome


# --- Models ----------------------------------------------------------------------------


def ols(df: pd.DataFrame, y: str, xs: list[str]) -> Any:
    """OLS of y on xs and a constant, standard errors clustered by division.

    Built from arrays rather than a formula string, so no column name is evaluated.
    Regressors with no values or no variation (post when nobody adopts) are left out.
    """
    if dropped := [x for x in xs if df[x].nunique() < 2]:
        warnings.warn(f"{y}: left out {', '.join(dropped)} (no variation)", stacklevel=2)
    xs = [x for x in xs if x not in dropped]
    data = df[[y, *xs, clean(UNIT)]].dropna()
    if data.empty:
        raise SystemExit(f"No row has {y} and all of: {', '.join(xs)}")
    groups = pd.factorize(data[clean(UNIT)])[0]
    model = OLS(data[y], add_constant(data[xs], has_constant="add"))
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


CHECKS = {  # regression diagnostics: name -> value for a fitted model
    "Observations": lambda r: r.nobs,
    "Census divisions (clusters)": lambda r: len(set(r.cov_kwds["groups"])),
    "R-squared": lambda r: r.rsquared,
    "Breusch-Pagan p, constant variance": lambda r: het_breuschpagan(r.resid, r.model.exog)[1],
    "Jarque-Bera p, normal residuals": lambda r: jarque_bera(r.resid)[1],
    "Durbin-Watson, 2 = no autocorrelation": lambda r: durbin_watson(r.resid),
    "RESET p, functional form": lambda r: linear_reset(r, power=2, use_f=True).pvalue,
    "Largest variance inflation factor": lambda r: max(
        (variance_inflation_factor(r.model.exog, i) for i in range(1, r.model.exog.shape[1])),
        default=np.nan,
    ),
}


def cell(res: Any, name: str) -> tuple[str, str]:
    """A coefficient with significance stars, and its standard error in parentheses."""
    if name not in res.params:
        return "", ""
    p = res.pvalues[name]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{res.params[name]:.4g}{stars}", f"({res.bse[name]:.3g})"


def regression_table(fits: dict[str, Any]) -> str:
    """Models side by side: coefficients, standard errors beneath, then diagnostics."""
    rows = [["", *fits]]
    for name in dict.fromkeys(n for res in fits.values() for n in res.params.index):
        cells = [cell(res, name) for res in fits.values()]
        rows += [[name, *(c for c, _ in cells)], ["", *(s for _, s in cells)]]
    rows.append([""] * (len(fits) + 1))
    rows += [[k, *(f"{check(r):.4g}" for r in fits.values())] for k, check in CHECKS.items()]
    first = max(len(row[0]) for row in rows) + 2
    width = max(len(c) for row in rows for c in row[1:]) + 2
    return "\n".join(row[0].ljust(first) + "".join(c.rjust(width) for c in row[1:]) for row in rows)


def pretrend_test(df: pd.DataFrame, y: str) -> dict[str, float]:
    """Pre-adoption trend of adopters minus that of never-adopters, per period.

    Uses adopters' rows before adoption and all never-adopter rows, removes each
    division's mean (division fixed effects) and regresses y on a trend and
    trend x adopter, clustered by division. Parallel pre-trends predict zero.
    """
    unit = clean(UNIT)
    pre = df.loc[(df["adopter"] == 0) | (df["event"] < 0), [unit, y, "t", "adopter"]].dropna()
    if pre.groupby("adopter")[unit].nunique().reindex([0, 1]).fillna(0).min() < 2:
        return {}
    pre["t_x_adopter"] = pre["t"] * pre["adopter"]
    columns = [y, "t", "t_x_adopter"]
    within = pre[columns] - pre.groupby(unit)[columns].transform("mean")
    groups = {"groups": pd.factorize(pre[unit])[0]}
    res = OLS(within[y], within[columns[1:]]).fit(cov_type="cluster", cov_kwds=groups)
    key = "t_x_adopter"
    return {"difference": res.params[key], "std. error": res.bse[key], "p": res.pvalues[key]}


def gaps(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Adopters minus the never-adopter mean of the same period, by periods since adoption.

    Each adopter's gap is measured from its own gap in the period before adoption, so
    the mean and standard error across adopters show changes, not level differences.
    """
    unit = clean(UNIT)
    never = df[df["adopter"] == 0].groupby("period")[y].mean()
    rows = df[(df["adopter"] == 1) & df["event"].between(-WINDOW, WINDOW)]
    gap = rows[y] - rows["period"].map(never)
    before = gap[rows["event"] == -1].groupby(rows[unit]).mean()
    change = gap - rows[unit].map(before)
    return change.groupby(rows["event"]).agg(["mean", "sem"])


# --- Report ----------------------------------------------------------------------------


def text_pages(pdf: PdfPages, title: str, text: str) -> None:
    """Monospaced text, continued onto further pages when long."""
    lines = text.splitlines()
    for start in range(0, max(len(lines), 1), LINES):
        fig = Figure(figsize=PAGE)
        fig.text(0.04, 0.94, title + (" (continued)" if start else ""), size=12, weight="bold")
        body = "\n".join(lines[start : start + LINES])
        fig.text(0.04, 0.9, body, family="monospace", size=8, va="top")
        pdf.savefig(fig)


def number(value: object) -> str:
    """4 significant digits, thousands separators when large, blank when missing."""
    if not isinstance(value, float):  # numpy floats are floats too
        return str(value)
    if np.isnan(value):
        return ""
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def show(frame: pd.DataFrame) -> str:
    """A table as text; wide tables continue in blocks of columns below."""
    return frame.map(number).to_string(line_width=140)


def statistics_text(df: pd.DataFrame, source: Path, notes: list[str]) -> str:
    unit, ys = clean(UNIT), [f"y_{o}" for o in map(clean, OUTCOMES)]
    columns = [clean(c) for c in [SCORE, *OUTCOMES, *CONTROLS, *EXTRA]]
    units = df.groupby(unit)
    adopted = units["adopted"].first()
    stats = df[columns].describe().T.rename(columns={"count": "values", "std": "sd"})
    stats.insert(1, "missing", len(df) - stats["values"])
    rows = units.size()
    first = df["adopted"].min()
    before = df[df["period"] < first] if pd.notna(first) else df.iloc[:0]
    balance = before.groupby("adopter")[columns].mean().T.rename(columns={0: "never", 1: "adopt"})
    correlations = df[[clean(SCORE), "post", *ys, *map(clean, CONTROLS)]].corr().round(2) + 0.0
    lines = [
        f"{source.name}: {len(df):,} rows, {units.ngroups} census divisions, "
        f"{df['period'].min()} to {df['period'].max()}; written {date.today()} by baseline.py",
        f"Bylaw in effect: {int(adopted.notna().sum())} divisions "
        f"({int((adopted <= units['period'].min()).sum())} before their data start); "
        f"never: {int(adopted.isna().sum())}. Rows per division: min {rows.min()}, "
        f"median {rows.median():g}, max {rows.max()}",
        *(textwrap.fill(f"Note: {n}", 140) for n in notes),
        "",
        "Summary statistics, all rows",
        show(stats),
        "",
        f"Means before the first bylaw took effect ({first}), by group",
        show(balance.rename_axis(columns=None)) if len(balance.columns) == 2 else "  not available",
        "",
        "Correlations (Pearson, pairwise complete rows; y_ is the outcome as modelled)",
        show(correlations),
    ]
    return "\n".join(lines)


def figure(title: str, nrows: int, ncols: int) -> tuple[Figure, np.ndarray]:
    """A page-sized figure outside pyplot, so nothing lingers in memory once saved."""
    fig = Figure(figsize=PAGE, layout="constrained")
    fig.suptitle(title, x=0.02, ha="left", size=12, weight="bold")
    return fig, fig.subplots(nrows, ncols, squeeze=False)


def trends_figure(df: pd.DataFrame) -> Figure:
    outcomes = list(map(clean, OUTCOMES))
    fig, axes = figure("Trends and pre-trends", 2, len(outcomes))
    for (top, bottom), outcome in zip(axes.T, outcomes, strict=True):
        y = f"y_{outcome}"
        for color, group, name in ((BLUE, 1, "adopt a bylaw"), (ORANGE, 0, "never adopt")):
            mean = df[df["adopter"] == group].groupby("period")[y].mean()
            top.plot(mean.index.to_timestamp(), mean.to_numpy(), color=color, label=name)
        top.set_title(f"Mean {label(outcome)} by period")
        top.legend()
        gap = gaps(df, y)
        band = 1.96 * gap["sem"]
        bottom.fill_between(
            gap.index, gap["mean"] - band, gap["mean"] + band, color=BLUE, alpha=0.1
        )
        bottom.plot(gap.index, gap["mean"], **DOTS)
        bottom.axhline(0, color=GREY, lw=0.75)
        bottom.axvline(-0.5, color=GREY, lw=0.75)
        bottom.set_title(f"{label(outcome)}: adopters minus never-adopters, 0 at -1, 95% band")
        bottom.set_xlabel("periods since the bylaw took effect")
    return fig


def score_figure(df: pd.DataFrame) -> Figure:
    outcomes = list(map(clean, OUTCOMES))
    cutoff = "" if CUTOFF is None else f" (cutoff {CUTOFF})"
    fig, axes = figure(f"The score{cutoff}", 1, len(outcomes) + 1)
    score = clean(SCORE)
    axes[0, 0].hist(df[score].dropna(), bins=30, color=BLUE, edgecolor="white")
    axes[0, 0].set_title("Distribution of the score, all rows")
    for ax, outcome in zip(axes[0, 1:], outcomes, strict=True):
        data = df[[score, f"y_{outcome}"]].dropna().set_axis(["x", "y"], axis=1)
        if data["x"].nunique() < 2:
            continue
        bins = data.groupby(pd.qcut(data["x"], 20, duplicates="drop"), observed=True).mean()
        ax.plot(bins["x"], bins["y"], ls="", label="mean of each score bin", **DOTS)
        sides = [data] if CUTOFF is None else [data[data.x < CUTOFF], data[data.x >= CUTOFF]]
        for i, side in enumerate(s for s in sides if s["x"].nunique() > 1):
            slope, intercept = np.polyfit(side["x"], side["y"], 1)
            x = np.array([side["x"].min(), side["x"].max()])
            ax.plot(x, intercept + slope * x, color=ORANGE, label=None if i else "OLS line")
        if CUTOFF is not None:
            ax.axvline(CUTOFF, color=GREY)
        ax.set_title(f"{label(outcome)} by score")
        ax.legend()
    return fig


def residual_figure(fits: dict[str, Any]) -> Figure:
    fig, axes = figure("Residuals of the models with controls", 2, len(fits))
    probs = np.linspace(0.005, 0.995, 199)
    normal = np.array([NormalDist().inv_cdf(p) for p in probs])
    for (top, bottom), (name, res) in zip(axes.T, fits.items(), strict=True):
        top.scatter(res.fittedvalues, res.resid, s=4, color=BLUE, alpha=0.3, rasterized=True)
        top.axhline(0, color=GREY, lw=0.75)
        top.set_title(f"{name}: residuals against fitted values")
        z = (res.resid - res.resid.mean()) / res.resid.std()
        bottom.plot(normal, np.quantile(z, probs), color=BLUE, label="standardized residuals")
        bottom.plot(normal[[0, -1]], normal[[0, -1]], color=ORANGE, label="normal")
        bottom.set_title(f"{name}: quantiles against the normal")
        bottom.legend()
    return fig


def report(df: pd.DataFrame, notes: list[str], source: Path, out: Path) -> None:
    """Fit the models and write the PDF via a temporary file, so a failure leaves none."""
    outcomes = list(map(clean, OUTCOMES))
    regressors = [clean(SCORE), "post", *map(clean, CONTROLS)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = len(outcomes)
        simple = {
            f"({i}) {o}": ols(df, f"y_{o}", regressors[:1]) for i, o in enumerate(outcomes, 1)
        }
        full = {f"({i}) {o}": ols(df, f"y_{o}", regressors) for i, o in enumerate(outcomes, n + 1)}
        table = regression_table(simple | full)
        tests = pd.DataFrame({label(o): pretrend_test(df, f"y_{o}") for o in outcomes}).T
    notes = notes + list(dict.fromkeys(str(w.message) for w in caught))
    models = "\n".join(
        [f"({i}) {label(o)} on the score" for i, o in enumerate(outcomes, 1)]
        + [
            f"({i}) {label(o)} on the score, post and controls"
            for i, o in enumerate(outcomes, n + 1)
        ]
    )
    regressions = (
        f"{models}\npost = 1 from the period the bylaw took effect\n\n{table}\n\n"
        "Standard errors clustered by census division in parentheses. "
        "* p<0.10, ** p<0.05, *** p<0.01\n\n"
        "Pre-trend test: pre-adoption trend of adopters minus never-adopters, per period\n"
        "(division fixed effects, clustered standard errors; parallel pre-trends predict 0)\n"
        + (show(tests) if not tests.empty else "  not available: needs 2+ divisions in each group")
    )
    tmp = out.with_name(f".{out.name}.part")
    try:
        with matplotlib.rc_context(STYLE), PdfPages(tmp) as pdf:
            text_pages(pdf, "Data and summary statistics", statistics_text(df, source, notes))
            pdf.savefig(trends_figure(df))
            pdf.savefig(score_figure(df))
            text_pages(pdf, "Baseline regressions", regressions)
            pdf.savefig(residual_figure(full))
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path, help=".csv or .xlsx, a row per division and period")
    parser.add_argument("-o", "--output", type=Path, help="PDF to write (default: beside the data)")
    args = parser.parse_args(argv)
    out = args.output or args.data.with_name(f"{args.data.stem}_baseline.pdf")
    if out.suffix.lower() != ".pdf":
        raise SystemExit(f"{out}: the report must be a .pdf file")
    if FREQ not in ("M", "Q", "Y"):
        raise SystemExit('FREQ must be "M", "Q" or "Y"')
    df, notes = prepare(load(args.data))
    report(df, notes, args.data, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
