"""Zoning bylaws and housing supply: a staggered difference-in-differences toolkit.

    python project.py               menu, using config.toml
    python project.py --demo        the menu on made-up data
    python project.py --run 1 3     run sections without the menu (--run all runs every one)

Section 1  Pre-tests and validity: parallel trends, placebo, robustness, balance
Section 2  Basic OLS, such as permits ~ population and starts ~ population
Section 3  Housing starts, quarterly: TWFE event study, did2s and dCDH
Section 4  Building permits, annual: the same estimators on annual data

config.toml names the data files, columns, controls and models. Each run saves its tables
(CSV, LaTeX), figures (PNG) and an index.html showing all of them in a new output folder.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import html
import importlib.metadata
import io
import json
import keyword
import platform
import re
import sys
import tomllib
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent
TIME_COLUMNS = ("date", "ref_date", "period", "quarter", "year", "month", "time")
FREQ_NAME = {"D": "daily", "M": "monthly", "Q": "quarterly", "Y": "annual"}
FREQ_RANK = {"D": 0, "M": 1, "Q": 2, "Y": 3}
PERIODS = {"Q": "quarters", "Y": "years"}
PARTS = {("M", "Q"): 3, ("M", "Y"): 12, ("Q", "Y"): 4}  # sub-periods in a complete period
RESERVED = {"y", "treat", "rel_time", "unit_id", "period_id", "first_id"}  # formula columns
PROVINCES = {
    "alberta": "AB",
    "british columbia": "BC",
    "manitoba": "MB",
    "new brunswick": "NB",
    "newfoundland and labrador": "NL",
    "nova scotia": "NS",
    "ontario": "ON",
    "prince edward island": "PE",
    "quebec": "QC",
    "québec": "QC",
    "saskatchewan": "SK",
}
# Palette slots 1-3 stay distinguishable with colour-vision deficiency; markers differ too.
COLORS, MARKERS = ("#2a78d6", "#eb6834", "#1baf7a"), ("o", "s", "^")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


class UserError(Exception):
    """A problem the user can fix in config.toml or the data; shown without a traceback."""


# --- Settings ---------------------------------------------------------------------------


def clean(name: Any) -> str:
    """A column name as formulas and output use it: lower case, other characters as "_".

    "FVI_CSCE_AB" becomes "fvi_csce_ab" and "Median income" becomes "median_income". The
    placeholder {region} is kept; names never start with a digit or equal a keyword.
    """
    parts = str(name).strip().lower().split("{region}")
    text = "{region}".join(re.sub(r"[^0-9a-z]+", "_", part) for part in parts).strip("_")
    if text[:1].isdigit():
        text = "v" + text
    return text + "_" if keyword.iskeyword(text) else text


@dataclass(frozen=True)
class Settings:
    """config.toml with column names cleaned. The menu changes a copy for the session."""

    files: tuple[Path, ...]
    unit: str
    region: str
    rename: dict[str, str]
    starts: str
    permits: str
    transform: str
    vote_date: str
    effective_date: str
    treatment_date: str  # "effective" or "vote"
    score: str
    zoning_type: str
    initiative: str
    controls: tuple[str, ...]
    barometer: tuple[str, ...]  # "-" in front flips a series
    regions: dict[str, str]
    ols_models: tuple[str, ...]
    ols_logs: bool
    windows: dict[str, tuple[int, int]]
    placebo_shift: dict[str, int]
    output: Path


def load_settings(path: Path) -> Settings:
    """Read config.toml; a missing or wrong setting is reported by name."""
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UserError(f"{path} not found. Copy config.toml there, or try --demo.") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise UserError(f"{path.name} is not valid TOML: {exc}") from None

    def get(section: str, key: str, default: Any = None) -> Any:
        value = cfg.get(section, {}).get(key, default)
        if value is None:
            raise UserError(f"{path.name}: [{section}] is missing the setting '{key}'")
        return value

    def items(section: str, key: str) -> list[str]:
        value = get(section, key, [])
        return [value] if isinstance(value, str) else list(value)

    def window(key: str, default: list[int]) -> tuple[int, int]:
        try:
            lo, hi = (int(v) for v in get("did", key, default))
        except (TypeError, ValueError):
            raise UserError(
                f"{path.name}: [did] {key} should be two numbers, like [-8, 12]"
            ) from None
        return lo, hi

    folder = path.resolve().parent
    treatment = cfg.get("treatment", {})
    settings = Settings(
        files=tuple(folder / f for f in items("data", "files")),
        unit=clean(get("data", "unit")),
        region=clean(get("data", "region", "province")),
        rename={clean(k): clean(v) for k, v in cfg.get("rename", {}).items()},
        starts=clean(get("outcomes", "starts")),
        permits=clean(get("outcomes", "permits")),
        transform=get("outcomes", "transform", "log"),
        vote_date=clean(get("treatment", "vote_date")),
        effective_date=clean(get("treatment", "effective_date")),
        treatment_date=get("treatment", "use", "effective"),
        score=clean(treatment.get("score", "")),
        zoning_type=clean(treatment.get("zoning_type", "")),
        initiative=clean(treatment.get("initiative", "")),
        controls=tuple(clean(c) for c in items("controls", "use")),
        barometer=tuple(
            ("-" if c.strip().startswith("-") else "") + clean(c)
            for c in items("market_barometer", "series")
        ),
        regions={str(k).upper(): str(v).upper() for k, v in cfg.get("regions", {}).items()},
        ols_models=tuple(items("ols", "models")),
        ols_logs=bool(get("ols", "logs", True)),
        windows={"Q": window("window_quarterly", [-8, 12]), "Y": window("window_annual", [-4, 6])},
        placebo_shift={
            "Q": int(get("did", "placebo_shift_quarterly", 8)),
            "Y": int(get("did", "placebo_shift_annual", 2)),
        },
        output=(folder / get("output", "folder", "output")).resolve(),
    )
    if not settings.files:
        raise UserError(f"{path.name}: [data] files lists no data files")
    return check_settings(settings)


def check_settings(s: Settings) -> Settings:
    """Reject settings that cannot work (also after changes in the menu)."""
    if s.transform not in ("log", "log1p", "none"):
        raise UserError('[outcomes] transform must be "log", "log1p" or "none"')
    if s.treatment_date not in ("effective", "vote"):
        raise UserError('[treatment] use must be "effective" or "vote"')
    for freq, (lo, hi) in s.windows.items():
        if not lo < -1 < 0 <= hi:
            raise UserError(
                f"The {FREQ_NAME[freq]} event window must start below -1 and end at 0 or later"
            )
    if min(s.placebo_shift.values()) < 1:
        raise UserError("[did] placebo shifts must be at least 1 period")
    for spec in s.ols_models:
        parse_model(spec)
    return s


def parse_model(spec: str) -> tuple[str, list[str]]:
    """Split "permits ~ population + pop_density" into the outcome and regressor names.

    Only names are allowed, each cleaned like a column name, so nothing typed in
    config.toml or the menu is ever run as code.
    """
    if not isinstance(spec, str) or spec.count("~") != 1:
        raise UserError(f'The OLS model "{spec}" should look like "permits ~ population"')
    left, right = spec.split("~")
    names = [clean(left), *(clean(term) for term in right.split("+"))]
    if "" in names:
        raise UserError(f'The OLS model "{spec}" has an empty name')
    return names[0], names[1:]


# --- Reading data -----------------------------------------------------------------------

QUARTER = re.compile(r"^(\d{4})\s*-?\s*Q([1-4])$|^Q([1-4])\s*-?\s*(\d{4})$", re.IGNORECASE)


def to_periods(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse a time column of years, quarter labels (2015Q1, Q1 2015) or dates.

    The spacing of dates sets the frequency, so StatCan's quarterly REF_DATE ("2015-01",
    "2015-04", ...) is quarterly; two dates in the same month make a column daily.
    """
    present = values.dropna()
    if present.empty:
        raise UserError("the time column is empty")
    if pd.api.types.is_numeric_dtype(present):
        if not ((present % 1 == 0).all() and present.between(1000, 9999).all()):
            raise UserError("numbers in a time column must be four-digit years")
        return as_periods(values.map(lambda v: f"{v:.0f}", na_action="ignore"), "Y"), "Y"
    text = values.astype("string").str.strip()
    quarter = text.str.extract(QUARTER)
    if quarter.notna().any(axis=1)[present.index].all():
        labels = quarter[0].fillna(quarter[3]) + "Q" + quarter[1].fillna(quarter[2])
        return as_periods(labels, "Q"), "Q"
    dates = pd.to_datetime(text, errors="coerce", format="mixed")
    unread = text.notna() & dates.isna()
    if unread.any():
        raise UserError(f"cannot read {text[unread].iloc[0]!r} as a date")
    freq = date_frequency(dates)
    return dates.dt.to_period(freq), freq


def as_periods(labels: pd.Series, freq: str) -> pd.Series:
    values = [None if pd.isna(v) else str(v) for v in labels]
    return pd.Series(pd.PeriodIndex(values, freq=freq), index=labels.index)


def date_frequency(dates: pd.Series) -> str:
    """Daily if two dates share a month, else annual, quarterly or monthly by spacing."""
    distinct = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    if len(distinct) < 2:
        raise UserError("a time column needs at least two different dates")
    months = np.asarray(distinct.year * 12 + distinct.month)
    if len(np.unique(months)) < len(months):
        return "D"
    step = int(np.gcd.reduce(np.diff(months)))
    return "Y" if step % 12 == 0 else "Q" if step % 3 == 0 else "M"


@dataclass
class Source:
    """One data file: a municipal "panel", "units" (a row per municipality) or "series"."""

    path: Path
    kind: str
    freq: str | None
    frame: pd.DataFrame  # cleaned column names; "_period" holds the parsed time column

    def describe(self) -> str:
        if self.kind == "units":
            return "one row per municipality"
        span = f"{self.frame['_period'].min()} to {self.frame['_period'].max()}"
        kind = "municipal panel" if self.kind == "panel" else "time series"
        return f"{kind}, {FREQ_NAME[self.freq]}, {span}"


def valet_preamble(path: Path) -> int:
    """Lines before the data in a Bank of Canada Valet CSV download (0 for other files)."""
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        if "TERMS AND CONDITIONS" not in fh.readline().upper():
            return 0
        for number, line in enumerate(fh, start=2):
            if line.strip().strip('"').upper() == "OBSERVATIONS":
                return number
    raise UserError(f"{path.name} looks like a Valet download but has no OBSERVATIONS section")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(path)  # openpyxl parses the XML with defusedxml when installed
    if suffix != ".csv":
        raise UserError(f"{path.name}: use .csv or .xlsx files (save .xls files as .xlsx)")
    skip = valet_preamble(path)
    try:
        return pd.read_csv(path, skiprows=skip, thousands=",", encoding="utf-8-sig")
    except UnicodeDecodeError:  # a CSV saved by Excel on Windows
        return pd.read_csv(path, skiprows=skip, thousands=",", encoding="cp1252")


def read_source(path: Path, s: Settings) -> Source:
    """Read one file, clean its column names and parse its time column."""
    if not path.is_file():
        raise UserError(f"Data file not found: {path}\nCheck [data] files in config.toml.")
    try:
        frame = read_table(path)
    except UserError:
        raise
    except Exception as exc:  # pandas and openpyxl raise many types for unreadable files
        raise UserError(f"Cannot read {path.name}: {exc}") from None
    frame = frame.dropna(how="all").rename(columns=clean).rename(columns=s.rename)
    clashes = set(frame.columns[frame.columns.duplicated()]) | (RESERVED & set(frame.columns))
    if clashes:
        raise UserError(
            f"{path.name}: rename the column(s) {', '.join(sorted(clashes))} "
            "(repeated after cleaning, or a name the program uses)"
        )
    unit, time = s.unit, next((c for c in TIME_COLUMNS if c in frame.columns), None)
    if unit in frame.columns:
        if frame[unit].isna().any():
            raise UserError(f"{path.name}: {int(frame[unit].isna().sum())} rows have no {unit}")
        ids = frame[unit]
        if pd.api.types.is_float_dtype(ids) and (ids % 1 == 0).all():
            ids = ids.astype("Int64")  # 3520005.0 -> 3520005, as in files without blanks
        frame[unit] = ids.astype("string").str.strip()
    elif time is None:
        raise UserError(f"{path.name} has neither the unit column '{unit}' nor a time column")
    if time is None:
        return Source(path, "units", None, frame)
    try:
        periods, freq = to_periods(frame[time])
    except UserError as exc:
        raise UserError(f"{path.name}, column {time}: {exc}") from None
    frame = frame.drop(columns=time).assign(_period=periods)
    keys = [unit, "_period"] if unit in frame.columns else ["_period"]
    repeated = frame.duplicated(keys, keep=False)
    if repeated.any():
        example = ", ".join(frame.loc[repeated, keys].iloc[0].astype(str))
        what = " and ".join(time if key == "_period" else key for key in keys)
        raise UserError(f"{path.name}: {int(repeated.sum())} rows share a {what} (e.g. {example})")
    return Source(path, "panel" if unit in frame.columns else "series", freq, frame)


# --- Building an analysis panel ---------------------------------------------------------


@dataclass
class PanelData:
    """One outcome at one frequency, with controls and (for DiD) bylaw timing.

    ``data`` columns: _unit, _period, _region, the outcome and controls under their own
    names, y (the outcome after the transform), unit_id, period_id and, with timing,
    first_id (adoption period; NaN if never treated), treat (bylaw in effect) and
    rel_time (periods since adoption, binned at the window ends; -1 if never treated).
    """

    data: pd.DataFrame
    freq: str
    outcome: str
    label: str
    controls: list[str]  # those that vary within municipality and period (usable with FE)
    all_controls: list[str]
    window: tuple[int, int]
    start: pd.Period  # the period with period_id 1
    bylaws: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)  # bylaw in effect before data start

    def period(self, period_id: float) -> pd.Period:
        return self.start + int(period_id) - 1

    def counts(self) -> dict[str, int]:
        first = self.data.groupby("_unit")["first_id"].first()
        return {
            "municipalities": len(first),
            "adopt a bylaw": int(first.notna().sum()),
            "never treated": int(first.isna().sum()),
            "left out": len(self.excluded),
        }


class Study:
    """The settings and data files of a session, and the panels built from them."""

    def __init__(self, settings: Settings, sources: list[Source]) -> None:
        self.settings, self.sources = settings, sources
        self._panels: dict[tuple, PanelData] = {}

    @classmethod
    def load(cls, config: Path) -> Study:
        settings = load_settings(config)
        return cls(settings, [read_source(path, settings) for path in settings.files])

    def find(self, column: str, kinds: Sequence[str] = ("panel", "units", "series")) -> Source:
        """The first file of one of these kinds with this column."""
        for src in self.sources:
            if src.kind in kinds and column in src.frame.columns:
                return src
        for src in self.sources:
            if column in src.frame.columns:
                raise UserError(
                    f"'{column}' is in {src.path.name} ({src.describe()}), but "
                    f"it needs to be in a municipal panel (municipality and time)"
                )
        known = sorted({c for src in self.sources for c in src.frame.columns} - {"_period"})
        close = difflib.get_close_matches(column, known, n=3)
        hint = f" Did you mean {' or '.join(close)}?" if close else ""
        raise UserError(f"No data file has a column '{column}'.{hint}")

    def panel(self, outcome: str, freq: str, **options: Any) -> PanelData:
        """build_panel(), remembered for the session (options must be hashable)."""
        key = (outcome, freq, *sorted(options.items()))
        if key not in self._panels:
            self._panels[key] = build_panel(self, outcome, freq, **options)
        return self._panels[key]

    def regions(self) -> pd.Series:
        """Region code of each municipality: its province code, mapped through [regions]."""
        s = self.settings
        pairs = [
            src.frame[[s.unit, s.region]]
            for src in self.sources
            if s.unit in src.frame.columns and s.region in src.frame.columns
        ]
        if not pairs:
            return pd.Series(dtype="string")
        region = pd.concat(pairs).dropna().drop_duplicates(s.unit).set_index(s.unit)[s.region]
        codes = region.astype("string").str.strip()
        codes = codes.map(lambda v: PROVINCES.get(v.lower(), v).upper())
        return codes.map(lambda v: s.regions.get(v, v))

    def bylaws(self) -> pd.DataFrame:
        """Each municipality's earliest bylaw: dates, score and types, indexed by unit."""
        s = self.settings
        dates = [s.vote_date, s.effective_date]
        frames = [
            src.frame
            for src in self.sources
            if s.unit in src.frame.columns and set(dates) & set(src.frame.columns)
        ]
        if not frames:
            raise UserError(
                f"No file has the bylaw date columns '{s.vote_date}' or "
                f"'{s.effective_date}' (see [treatment] in config.toml)"
            )
        z = pd.concat(frames, ignore_index=True).rename(columns={s.unit: "_unit"})
        wanted = (*dates, s.score, s.zoning_type, s.initiative)
        z = z[["_unit", *(c for c in dict.fromkeys(wanted) if c and c in z.columns)]]
        for col in dates:
            if col in z.columns:
                parsed = pd.to_datetime(z[col], errors="coerce", format="mixed")
                unread = z[col].notna() & parsed.isna()
                if unread.any():
                    raise UserError(
                        f"Cannot read {z.loc[unread, col].iloc[0]!r} in {col} as a date"
                    )
                z[col] = parsed
        earliest = z[[c for c in dates if c in z.columns]].min(axis=1)
        z = z.assign(_order=earliest).sort_values("_order").drop_duplicates("_unit")
        return z.drop(columns="_order").set_index("_unit")


def lookup(
    panel: pd.DataFrame, freq: str, right: pd.DataFrame, right_freq: str | None, keys: list[str]
) -> pd.Series:
    """right["value"] for every panel row, matched on keys and the period.

    Finer data (a monthly series for a quarterly panel) are averaged within each period;
    coarser data (annual population for a quarterly panel) repeat within the year.
    """
    right = right.dropna(subset=["value"])
    left, on = panel[[*keys, "_period"]], list(keys)
    if right_freq is None:  # one value per municipality
        right = right.drop_duplicates(keys)
    else:
        if FREQ_RANK[right_freq] < FREQ_RANK[freq]:
            right = right.assign(_period=right["_period"].dt.asfreq(freq))
            right = right.groupby([*keys, "_period"], as_index=False)["value"].mean()
        elif FREQ_RANK[right_freq] > FREQ_RANK[freq]:
            left = left.assign(_period=left["_period"].dt.asfreq(right_freq))
        on.append("_period")
    merged = left.merge(right[[*on, "value"]], on=on, how="left", validate="many_to_one")
    return pd.Series(merged["value"].to_numpy(dtype=float), index=panel.index)


def numeric(src: Source, column: str) -> pd.Series:
    values = src.frame[column]
    if not pd.api.types.is_numeric_dtype(values):
        example = values.dropna().iloc[0] if values.notna().any() else ""
        raise UserError(f"'{column}' in {src.path.name} is not numeric (e.g. {example!r})")
    return values


def control_values(study: Study, panel: pd.DataFrame, freq: str, name: str) -> pd.Series:
    """One control for every panel row.

    A municipal column is matched on municipality and period, a national series on the
    period only, and a name with {region} (fvi_csce_{region}) on each municipality's
    region and the period. market_barometer averages the series in [market_barometer]
    after standardizing each within region.
    """
    s = study.settings
    if name == "market_barometer":
        parts = []
        for entry in s.barometer:
            values = control_values(study, panel, freq, entry.lstrip("-"))
            region = values.groupby(panel["_region"].fillna(""))
            z = (values - region.transform("mean")) / region.transform("std")
            parts.append(-z if entry.startswith("-") else z)
        return pd.concat(parts, axis=1).mean(axis=1)
    if "{region}" in name:
        pattern = re.compile(re.escape(name).replace(r"\{region\}", "([a-z]+)"))
        found = [
            (src, col, match[1].upper())
            for src in study.sources
            if src.kind == "series"
            for col in src.frame.columns
            if (match := pattern.fullmatch(col))
        ]
        if not found:
            example = name.replace("{region}", "on")
            raise UserError(f"No time-series column matches '{name}' (such as {example})")
        values = pd.Series(np.nan, index=panel.index)
        for src, col, code in found:
            right = pd.DataFrame(
                {"_region": code, "_period": src.frame["_period"], "value": numeric(src, col)}
            )
            values = values.combine_first(lookup(panel, freq, right, src.freq, ["_region"]))
        return values
    src = study.find(name)
    right = src.frame.assign(value=numeric(src, name)).rename(columns={s.unit: "_unit"})
    return lookup(panel, freq, right, src.freq, [] if src.kind == "series" else ["_unit"])


def outcome_rows(study: Study, outcome: str, freq: str) -> tuple[pd.DataFrame, list[str]]:
    """The outcome by municipality and period, added up from finer data if needed."""
    src = study.find(outcome, ("panel",))
    if FREQ_RANK[src.freq] > FREQ_RANK[freq]:
        raise UserError(
            f"{outcome} is {FREQ_NAME[src.freq]} in {src.path.name}; this section "
            f"needs {FREQ_NAME[freq]} data"
        )
    unit = study.settings.unit
    data = src.frame[[unit, "_period"]].assign(**{outcome: numeric(src, outcome)})
    data, notes = data.rename(columns={unit: "_unit"}), []
    if src.freq != freq:  # flows such as monthly starts add up to quarters and years
        grouped = data.assign(_period=data["_period"].dt.asfreq(freq)).groupby(["_unit", "_period"])
        totals = grouped[outcome].sum(min_count=1)
        need = PARTS.get((src.freq, freq))
        if need:
            incomplete = grouped[outcome].count() < need
            if incomplete.any():
                notes.append(
                    f"{int(incomplete.sum())} {FREQ_NAME[freq]} totals of {outcome} are "
                    f"left out because some {FREQ_NAME[src.freq]} values are missing"
                )
            totals = totals[~incomplete]
        data = totals.reset_index()
    return data.dropna(subset=[outcome]).reset_index(drop=True), notes


def build_panel(
    study: Study,
    outcome: str,
    freq: str,
    *,
    controls: tuple[str, ...] | None = None,
    timing: bool = True,
    treatment_date: str | None = None,
    transform: str | None = None,
) -> PanelData:
    """One outcome at one frequency with controls and, if timing, bylaw adoption."""
    s = study.settings
    controls = list(s.controls if controls is None else controls)
    transform = transform or s.transform
    data, notes = outcome_rows(study, outcome, freq)
    data["_region"] = data["_unit"].map(study.regions())
    for name in controls:
        if name not in data.columns:
            data[name] = control_values(study, data, freq, name)
    y = data[outcome].astype(float)
    if transform == "log":
        if (y <= 0).any():
            notes.append(
                f"log leaves out {int((y <= 0).sum())} rows where {outcome} is 0 or "
                'less; transform = "log1p" keeps them'
            )
        data = data[y > 0].reset_index(drop=True)
        data["y"] = np.log(data[outcome].astype(float))
    else:
        data["y"] = np.log1p(y) if transform == "log1p" else y
    data["unit_id"] = pd.factorize(data["_unit"], sort=True)[0] + 1
    ordinals = data["_period"].array.asi8
    origin = ordinals.min() - 1
    data["period_id"] = ordinals - origin
    p = PanelData(
        data=data,
        freq=freq,
        outcome=outcome,
        label=outcome if transform == "none" else f"{transform}({outcome})",
        controls=[],
        all_controls=controls,
        window=s.windows[freq],
        start=data["_period"].min(),
        notes=notes,
    )
    if timing:
        add_timing(p, study, origin, treatment_date or s.treatment_date)
    p.controls = usable_controls(p)
    return p


def add_timing(p: PanelData, study: Study, origin: int, use: str) -> None:
    """first_id, treat and rel_time from each municipality's earliest bylaw."""
    s, data = study.settings, p.data
    p.bylaws = study.bylaws()
    date = s.effective_date if use == "effective" else s.vote_date
    if date not in p.bylaws.columns:
        raise UserError(f"Treatment starts at the {use} date, but no file has the column '{date}'")
    adoption = p.bylaws[date].dropna().dt.to_period(p.freq)
    first = pd.Series(adoption.array.asi8 - origin, index=adoption.index)
    data["first_id"] = data["_unit"].map(first).astype(float)
    unmatched = sorted(set(p.bylaws.index) - set(data["_unit"]))
    if unmatched:
        more = ", ..." if len(unmatched) > 6 else ""
        p.notes.append(
            f"{len(unmatched)} municipalities in the bylaw file have no {p.outcome} "
            f"data (check the spelling): {', '.join(unmatched[:6])}{more}"
        )
    early = data["first_id"] <= data.groupby("_unit")["period_id"].transform("min")
    p.excluded = sorted(data.loc[early, "_unit"].unique())
    if p.excluded:
        p.notes.append(
            "Left out of the DiD because their bylaw was already in effect when "
            f"their data start: {', '.join(p.excluded)}"
        )
    data = data[~early].reset_index(drop=True)
    data.loc[data["first_id"] > data["period_id"].max(), "first_id"] = np.nan  # after the data
    data["treat"] = (data["period_id"] >= data["first_id"]).astype(int)
    data["rel_time"] = (data["period_id"] - data["first_id"]).clip(*p.window).fillna(-1)
    p.data = data.astype({"rel_time": int})


def usable_controls(p: PanelData) -> list[str]:
    """Controls that fixed effects do not absorb; the rest are noted and kept for pooled OLS."""
    usable = []
    for name in p.all_controls:
        values = p.data[name]
        if values.notna().sum() == 0:
            p.notes.append(f"{name} has no values for these rows and is left out")
            continue
        if values.isna().any():
            p.notes.append(
                f"{name} is missing for {values.isna().mean():.0%} of rows; models "
                "that use it leave those rows out"
            )
        if p.data.groupby("period_id")[name].nunique().max() <= 1:
            p.notes.append(
                f"{name} is the same for every municipality in a period, so period "
                "fixed effects absorb it: it is used in pooled OLS only"
            )
        elif p.data.groupby("unit_id")[name].nunique().max() <= 1:
            p.notes.append(
                f"{name} never changes within a municipality, so municipality "
                "fixed effects absorb it: it is used in pooled OLS only"
            )
        else:
            usable.append(name)
    return usable


# --- Estimators -------------------------------------------------------------------------


def quietly(fit: Callable[[], Any], notes: list[str], who: str) -> Any:
    """Run an estimator with its printing hidden and its warnings kept as notes."""
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        result = fit()
    for w in caught:
        text = f"{who}: " + " ".join(str(w.message).split())
        if not issubclass(w.category, DeprecationWarning | FutureWarning) and text not in notes:
            notes.append(text)
    return result


def attempt(fit: Callable[[], Any], notes: list[str], what: str) -> Any:
    """An optional estimate: on failure, a note and None instead of a failed section."""
    try:
        return fit()
    except Exception as exc:  # the estimators raise many types; the section goes on
        notes.append(f"{what} could not be estimated: {type(exc).__name__}: {exc}")
        return None


def plus(names: Sequence[str]) -> str:
    return "".join(f" + {n}" for n in names)


def ols(formula: str, data: pd.DataFrame, notes: list[str], who: str) -> Any:
    """OLS (pyfixest formula), standard errors clustered by municipality."""
    return quietly(lambda: pf.feols(formula, data, vcov={"CRV1": "unit_id"}), notes, who)


def twfe(p: PanelData, rhs: str, notes: list[str], controls: Sequence[str] | None = None) -> Any:
    """Two-way fixed effects: municipality and period effects, the panel's controls."""
    controls = p.controls if controls is None else controls
    return ols(f"y ~ {rhs}{plus(controls)} | unit_id + period_id", p.data, notes, "TWFE")


def did2s(
    p: PanelData,
    second: str,
    notes: list[str],
    controls: Sequence[str] | None = None,
    data: pd.DataFrame | None = None,
) -> Any:
    """Gardner's two-stage DiD: fixed effects and controls are fit on untreated rows only."""
    controls = p.controls if controls is None else controls
    data = (p.data if data is None else data).dropna(subset=["y", *controls])
    first = f"~ {' + '.join(controls) or '0'} | unit_id + period_id"
    fit = partial(pf.did2s, data, "y", first, second, "treat", cluster="unit_id")
    return quietly(fit, notes, "did2s")


TIDY = {"Estimate": "estimate", "Std. Error": "se", "2.5%": "ci_low", "97.5%": "ci_high"}
DCDH = {"Estimate": "estimate", "SE": "se", "LB CI": "ci_low", "UB CI": "ci_high"}


def dcdh(
    p: PanelData, notes: list[str], never_only: bool = False
) -> tuple[pd.DataFrame, dict, float]:
    """de Chaisemartin-D'Haultfoeuille: event-time rows, average effect, placebo p-value."""
    import polars as pl  # loaded here: the package takes a second or two to import
    from did_multiplegt_dyn import DidMultiplegtDyn

    data = p.data.dropna(subset=["y", *p.controls])
    data = data[["unit_id", "period_id", "y", *p.controls]].assign(d=data["treat"].astype(float))
    lo, hi = p.window
    # same_switchers stays off: with only_never_switchers it crashes version 0.1.9.
    options = {"effects": hi + 1, "placebo": -lo - 1, "only_never_switchers": never_only}
    if p.controls:
        options["controls"] = list(p.controls)

    def fit() -> Any:
        model = DidMultiplegtDyn(pl.from_pandas(data), "y", "unit_id", "period_id", "d", **options)
        model.fit()
        return model, model.summary().set_index("Block")

    model, table = quietly(fit, notes, "dCDH")
    rows = table[table.index.str.fullmatch(r"(Effect|Placebo)_\d+")]
    lag = rows.index.str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    # dCDH counts from the last untreated period: Effect_l is event time l-1, Placebo_l is -l-1
    event_time = np.where(rows.index.str.startswith("Effect"), lag - 1, -lag - 1)
    events = rows[list(DCDH)].rename(columns=DCDH).assign(estimator="dCDH", event_time=event_time)
    average = table.loc["Average_Total_Effect", list(DCDH)].rename(DCDH).to_dict()
    placebo_p = model.result["did_multiplegt_dyn"].get("p_jointplacebo", np.nan)
    return events.sort_values("event_time").reset_index(drop=True), average, float(placebo_p)


def event_rows(model: Any, estimator: str) -> pd.DataFrame:
    """Event-time coefficients (rel_time::k) of a pyfixest model as rows."""
    t = model.tidy()
    t = t[t.index.str.startswith("rel_time::")].rename(columns=TIDY)[list(TIDY.values())]
    times = [int(float(name.split("::")[1])) for name in t.index]
    return t.assign(estimator=estimator, event_time=times).reset_index(drop=True)


def effect(specification: str, model: Any, term: str = "treat", **extra: Any) -> dict:
    """One coefficient as a table row."""
    row = model.tidy().loc[term, list(TIDY)].rename(TIDY)
    return {"specification": specification, **row.to_dict(), **extra}


def pretrend_test(test: str, model: Any) -> dict:
    """Wald test that every pre-adoption event-time coefficient is zero."""
    names = list(model.coef().index)
    pre = [i for i, name in enumerate(names) if name.startswith("rel_time::-")]
    restrictions = np.zeros((len(pre), len(names)))
    restrictions[np.arange(len(pre)), pre] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pyfixest says which distribution it uses
        result = model.wald_test(R=restrictions)
    return {
        "test": test,
        "pre-periods": len(pre),
        "statistic": float(result["statistic"]),
        "p_value": float(result["pvalue"]),
    }


def reading(p_value: float) -> str:
    if np.isnan(p_value):
        return "not available"
    if p_value < 0.05:
        return "pre-trends differ (p < 0.05)"
    if p_value < 0.10:
        return "weak evidence of different pre-trends"
    return "no evidence against parallel trends"


# --- Figures ----------------------------------------------------------------------------


def axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Any]:
    """A figure outside pyplot's global state, so nothing needs closing."""
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    return fig, ax


def legend(ax: Any, loc: str = "best") -> None:
    ax.legend(loc=loc, frameon=False, fontsize=8, labelcolor=MUTED)


def event_plot(rows: pd.DataFrame, title: str, unit: str) -> Figure:
    fig, ax = axes(
        title,
        f"{unit} since adoption (the end points pool earlier and later {unit})",
        "Effect and 95% interval",
    )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(-0.5, color=GRID, linewidth=1)
    names = list(dict.fromkeys(rows["estimator"]))
    for i, name in enumerate(names):
        r = rows[rows["estimator"] == name]
        x = r["event_time"] + (i - (len(names) - 1) / 2) * 0.15  # side by side, not on top
        err = np.clip([r["estimate"] - r["ci_low"], r["ci_high"] - r["estimate"]], 0, None)
        ax.errorbar(
            x,
            r["estimate"],
            yerr=err,
            fmt=MARKERS[i % 3],
            color=COLORS[i % 3],
            markersize=5,
            elinewidth=1.2,
            capsize=0,
            label=name,
        )
    ax.plot([-1], [0], "o", mfc="white", mec=MUTED, label="reference period (-1)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend(ax, "upper left")  # "best" ignores error bars; pre-periods sit near zero
    return fig


def scatter_plot(x: pd.Series, y: pd.Series, title: str) -> Figure:
    fig, ax = axes(title, str(x.name), str(y.name))
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=10, alpha=0.5, color=COLORS[0], linewidths=0)
    if ok.sum() > 2:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        grid = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(grid, intercept + slope * grid, color=COLORS[1], linewidth=2, label="OLS line")
        legend(ax)
    return fig


def over_time(p: PanelData, title: str, ylabel: str) -> tuple[Figure, Any]:
    fig, ax = axes(title, "", ylabel)
    ticks = np.unique(np.linspace(1, p.data["period_id"].max(), 8).astype(int))
    ax.set_xticks(ticks, [str(p.period(t)) for t in ticks])
    return fig, ax


def trends_plot(p: PanelData) -> Figure:
    fig, ax = over_time(p, f"Mean {p.label}: adopters and never-treated", p.label)
    groups = {
        "adopt a bylaw": p.data["first_id"].notna(),
        "never treated": p.data["first_id"].isna(),
    }
    for i, (name, rows) in enumerate(groups.items()):
        means = p.data[rows].groupby("period_id")["y"].mean()
        ax.plot(means.index, means.to_numpy(), color=COLORS[i], linewidth=2, label=name)
    legend(ax)
    return fig


def rollout_plot(p: PanelData) -> Figure:
    fig, ax = over_time(p, "Share of municipalities with a bylaw in effect", "share")
    share = p.data.groupby("period_id")["treat"].mean()
    ax.step(share.index, share.to_numpy(), where="post", color=COLORS[0], linewidth=2)
    ax.set_ylim(0, 1)
    return fig


def forest_plot(table: pd.DataFrame, title: str) -> Figure:
    fig, ax = axes(title, "Average effect after adoption, 95% interval", "")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    pos = np.arange(len(table))[::-1]
    err = np.clip(
        [table["estimate"] - table["ci_low"], table["ci_high"] - table["estimate"]], 0, None
    )
    ax.errorbar(table["estimate"], pos, xerr=err, fmt="o", color=COLORS[0])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(pos, table.index)
    return fig


# --- Sections ---------------------------------------------------------------------------


@dataclass
class Result:
    key: str  # file name prefix, e.g. "s3_starts"
    title: str
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    latex: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def section1(study: Study, outcome: str, freq: str) -> Result:
    """Pre-tests and validity: coverage, timing, parallel trends, placebo, robustness, balance."""
    s, p = study.settings, study.panel(outcome, freq)
    d, unit = p.data, PERIODS[freq]
    title = f"Section 1 · Pre-tests and validity · {p.label}, {FREQ_NAME[freq]}"
    r = Result(f"s1_{outcome}", title, notes=list(p.notes))

    n = p.counts()
    span = f"{d['_period'].min()} to {d['_period'].max()}"
    data = pd.Series({**n, "periods": span, "rows": len(d)}, name="value").rename_axis("item")
    r.tables.append(("Data", data.to_frame()))
    cohorts = d.groupby("_unit")["first_id"].first().dropna().map(lambda t: str(p.period(t)))
    cohorts = cohorts.value_counts().sort_index().rename_axis("adoption").rename("municipalities")
    r.tables.append(("Adoption cohorts", cohorts.to_frame()))
    if n["municipalities"] < 30 or n["adopt a bylaw"] < 10:
        r.notes.append(
            "Few municipalities or adopters: clustered standard errors can be too "
            "small, so read p-values with care."
        )
    if {s.vote_date, s.effective_date} <= set(p.bylaws.columns):
        lag = (p.bylaws[s.effective_date] - p.bylaws[s.vote_date]).dt.days.dropna()
        if (lag < 0).any():
            r.notes.append(f"{int((lag < 0).sum())} bylaws take effect before their council vote")
        summary = lag.describe()[["count", "min", "50%", "max"]].rename({"50%": "median"})
        summary = summary.rename("days").rename_axis("statistic").to_frame()
        r.tables.append(("Days from council vote to effective date", summary))
    r.figures += [("Rollout", rollout_plot(p)), ("Trends", trends_plot(p))]

    tw = twfe(p, "i(rel_time, ref=-1)", r.notes)
    d2 = did2s(p, "~ i(rel_time, ref=-1)", r.notes)
    dc = attempt(lambda: dcdh(p, r.notes), r.notes, "dCDH")
    tests, events = [pretrend_test("TWFE event study", tw), pretrend_test("did2s", d2)], []
    if dc:
        pre = int((dc[0]["event_time"] < 0).sum())
        tests.append(
            {"test": "dCDH placebos", "pre-periods": pre, "statistic": np.nan, "p_value": dc[2]}
        )
        events.append(dc[0])
    tests = pd.DataFrame(tests).set_index("test")
    tests["reading"] = tests["p_value"].map(reading)
    r.tables.append(("Parallel trends: are the pre-adoption estimates jointly zero?", tests))
    r.notes.append(
        "Not rejecting parallel pre-trends is not proof of parallel trends: with "
        "few municipalities these tests have little power."
    )
    rows = pd.concat([event_rows(tw, "TWFE"), event_rows(d2, "did2s"), *events])
    r.figures.append(("Pre-trends", event_plot(rows, f"Event study of {p.label}", unit)))

    shift = s.placebo_shift[freq]
    robust = [effect("Baseline (did2s)", did2s(p, "~ treat", r.notes))]
    untreated = d[d["first_id"].isna() | (d["period_id"] < d["first_id"])]
    fake = untreated["first_id"] - shift
    placebo = untreated[
        fake.isna() | (fake > untreated.groupby("_unit")["period_id"].transform("min"))
    ]
    placebo = placebo.assign(
        treat=(placebo["period_id"] >= placebo["first_id"] - shift).astype(int)
    )
    if placebo["treat"].any():
        fit = attempt(lambda: did2s(p, "~ treat", r.notes, data=placebo), r.notes, "Placebo")
        if fit is not None:
            robust.append(effect(f"Placebo: adoption {shift} {unit} early (expect 0)", fit))
    other = "vote" if s.treatment_date == "effective" else "effective"
    try:
        q = study.panel(outcome, freq, treatment_date=other)
        robust.append(effect(f"Treatment from the {other} date", did2s(q, "~ treat", r.notes)))
    except UserError as exc:
        r.notes.append(f"Skipped the {other}-date check: {exc}")
    robust.append(effect("No controls", did2s(p, "~ treat", r.notes, controls=[])))
    if {s.vote_date, s.effective_date} <= set(p.bylaws.columns):
        vote = d["_unit"].map(p.bylaws[s.vote_date].dt.to_period(freq))
        takes_effect = d["_unit"].map(p.bylaws[s.effective_date].dt.to_period(freq))
        between = (d["_period"] >= vote) & (d["_period"] < takes_effect)
        if between.any():
            fit = did2s(p, "~ treat", r.notes, data=d[~between])
            robust.append(effect("Without the vote-to-effective periods", fit))
    robust.append(effect("TWFE (can be biased with staggered timing)", twfe(p, "treat", r.notes)))
    if dc:
        robust.append({"specification": "dCDH", **dc[1]})
    never = attempt(lambda: dcdh(p, r.notes, never_only=True), r.notes, "dCDH (never)")
    if never:
        robust.append({"specification": "dCDH, never-treated comparisons only", **never[1]})
    robust = pd.DataFrame(robust).set_index("specification")
    r.tables.append(("Placebo and robustness: average effect after adoption", robust))
    r.figures.append(("Robustness", forest_plot(robust, f"Placebo and robustness, {p.label}")))

    before = d[d["period_id"] < d["first_id"].min()]
    adopter = before["first_id"].notna()
    balance = {}
    varies = [c for c in p.all_controls if before.groupby("period_id")[c].nunique().max() > 1]
    for name in ["y", *varies]:
        a, b = before.loc[adopter, name], before.loc[~adopter, name]
        gap = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        balance[p.label if name == "y" else name] = {
            "adopters": a.mean(),
            "never treated": b.mean(),
            "std. difference": gap,
        }
    r.tables.append(
        (
            "Balance before the first adoption (|std. difference| > 0.25 is large)",
            pd.DataFrame(balance).T.rename_axis("variable"),
        )
    )
    if p.all_controls:
        r.notes.append(
            "Controls measured after adoption can respond to the bylaw (bad "
            "controls); compare the 'No controls' row."
        )
    return r


def section2(study: Study) -> Result:
    """Basic OLS for each [ols] model: pooled, pooled with controls, and with fixed effects."""
    s = study.settings
    r = Result("s2", "Section 2 · Basic OLS (standard errors clustered by municipality)")
    rows, fits = [], []
    for spec in s.ols_models:
        outcome, regressors = parse_model(spec)
        freq = study.find(outcome, ("panel",)).freq
        extra = tuple(c for c in s.controls if c not in regressors)
        p = study.panel(
            outcome, freq, controls=(*regressors, *extra), timing=False, transform="none"
        )
        r.notes += [note for note in p.notes if note not in r.notes]
        d, names = p.data, [outcome, *regressors]
        if s.ols_logs:
            positive = (d[names] > 0).all(axis=1)
            if not positive.all():
                r.notes.append(
                    f"{spec}: logs leave out {int((~positive).sum())} rows with values of 0 or less"
                )
            d = d[positive].assign(**{f"log_{n}": np.log(d.loc[positive, n]) for n in names})
            names = [f"log_{n}" for n in names]
        y, xs = names[0], names[1:]
        specs = {"pooled": f"{y} ~ {' + '.join(xs)}"}
        if extra:
            specs["pooled + controls"] = specs["pooled"] + plus(extra)
        fe = [c for c in p.controls if c in extra]
        specs["municipality and period FE"] = f"{specs['pooled']}{plus(fe)} | unit_id + period_id"
        for label, formula in specs.items():
            fit = attempt(partial(ols, formula, d, r.notes, spec), r.notes, f"{spec} ({label})")
            if fit is None:
                continue
            fits.append(fit)
            for x in xs:
                if x in fit.coef().index:
                    rows.append(
                        {**effect(label, fit, x), "model": f"{y} ~ {x}", "N": fit._N, "R2": fit._r2}
                    )
        fig = scatter_plot(d[xs[0]], d[y], f"{y} against {xs[0]} ({FREQ_NAME[freq]})")
        r.figures.append((f"{y} vs {xs[0]}", fig))
    if rows:
        table = pd.DataFrame(rows).set_index(["model", "specification"])
        r.tables.append(("OLS estimates", table))
    with contextlib.suppress(Exception):  # LaTeX is a convenience; the CSV has the numbers
        r.latex.append(("ols", pf.etable(fits, type="tex")))
    return r


def did_section(study: Study, number: int, outcome: str, freq: str) -> Result:
    """Sections 3 and 4: event studies and average effects from three estimators."""
    p = study.panel(outcome, freq)
    title = f"Section {number} · {p.label}, {FREQ_NAME[freq]} · staggered DiD"
    r = Result(f"s{number}_{outcome}", title, notes=list(p.notes))
    tw = twfe(p, "i(rel_time, ref=-1)", r.notes)
    d2 = did2s(p, "~ i(rel_time, ref=-1)", r.notes)
    dc = attempt(lambda: dcdh(p, r.notes), r.notes, "dCDH")
    rows = pd.concat([event_rows(tw, "TWFE"), event_rows(d2, "did2s"), *([dc[0]] if dc else [])])
    wide = {
        f"{name} {stat}": part.set_index("event_time")[stat]
        for name, part in rows.groupby("estimator", sort=False)
        for stat in ("estimate", "se")
    }
    r.tables.append(
        ("Event study: effect by period since adoption", pd.DataFrame(wide).sort_index())
    )
    r.figures.append(("Event study", event_plot(rows, f"Bylaw effect on {p.label}", PERIODS[freq])))

    static_tw = twfe(p, "treat", r.notes)
    static_d2 = did2s(p, "~ treat", r.notes)
    average = [
        effect("did2s (Gardner)", static_d2),
        effect("TWFE (can be biased with staggered timing)", static_tw),
    ]
    if dc:
        average.insert(1, {"specification": "dCDH average total effect", **dc[1]})
    r.tables.append(("Average effect after adoption", percent(pd.DataFrame(average), p)))
    r.tables.append(
        ("Effect by bylaw type and score (did2s)", percent(heterogeneity(study, p, r.notes), p))
    )
    with contextlib.suppress(Exception):
        r.latex.append((f"section{number}", pf.etable([static_d2, static_tw], type="tex")))
    return r


def percent(table: pd.DataFrame, p: PanelData) -> pd.DataFrame:
    """Index by specification; with a log outcome, add the effect as a % change."""
    table = table.set_index("specification") if "specification" in table else table
    if p.label.startswith("log") and "estimate" in table:
        table["% change"] = 100 * np.expm1(table["estimate"])
    return table


def heterogeneity(study: Study, p: PanelData, notes: list[str]) -> pd.DataFrame:
    """did2s effect for adopters of each bylaw type, against the never-treated."""
    s = study.settings
    adopters = p.data.loc[p.data["first_id"].notna(), "_unit"].unique()
    bylaws = p.bylaws[p.bylaws.index.isin(adopters)]
    groups = {}
    for col in (s.zoning_type, s.initiative):
        if col in bylaws:
            for value, units in bylaws.groupby(col).groups.items():
                groups[f"{col}: {value}"] = units
    if s.score in bylaws:
        score = pd.to_numeric(bylaws[s.score], errors="coerce")
        median = score.median()
        if pd.notna(median):
            groups[f"{s.score} at or above {median:g} (median)"] = bylaws.index[score >= median]
            groups[f"{s.score} below {median:g}"] = bylaws.index[score < median]
    rows = []
    for name, units in groups.items():
        if len(units) < 3:
            notes.append(f"{name}: {len(units)} adopter(s), too few to estimate an effect")
            continue
        sample = p.data[p.data["first_id"].isna() | p.data["_unit"].isin(units)]
        fit = attempt(lambda d=sample: did2s(p, "~ treat", notes, data=d), notes, name)
        if fit is not None:
            rows.append(effect(name, fit, adopters=len(units)))
    return pd.DataFrame(rows)


# --- Saving -----------------------------------------------------------------------------


def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")[:60] or "item"


def save(results: list[Result], study: Study, label: str) -> Path:
    """Write every table (CSV), figure (PNG), LaTeX table and an index.html; return the folder."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{slug(label)}"
    folder, n = study.settings.output / stamp, 1
    while folder.exists():
        n += 1
        folder = study.settings.output / f"{stamp}_{n}"
    folder.mkdir(parents=True)
    used: set[str] = set()

    def name_for(text: str, extension: str) -> str:
        """A short unique file name: the text up to its first ":" or "(", in lower case."""
        base = slug(re.split(r"[:(]", text)[0]).lower()
        name, count = f"{base}.{extension}", 1
        while name in used:
            count += 1
            name = f"{base}_{count}.{extension}"
        used.add(name)
        return name

    page = [
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(label)}</title><style>"
        "body{font:14px sans-serif;max-width:60em;margin:auto;padding:1em}table{border-"
        "collapse:collapse}td,th{padding:2px 8px;border-bottom:1px solid #ddd;text-align:"
        "right}img{max-width:100%}</style>",
        f"<h1>{html.escape(label)}</h1><p>{datetime.now():%Y-%m-%d %H:%M}</p>",
    ]
    for r in results:
        page.append(f"<h2>{html.escape(r.title)}</h2>")
        for title, table in r.tables:
            table.to_csv(folder / name_for(f"{r.key} {title}", "csv"))
            body = table.to_html(float_format=cell, na_rep="")
            page.append(f"<h3>{html.escape(title)}</h3>{body}")
        for title, fig in r.figures:
            name = name_for(f"{r.key} {title}", "png")
            fig.savefig(folder / name, dpi=150)
            page.append(f"<p><img src='{name}' alt='{html.escape(title)}'></p>")
        for title, tex in r.latex:
            (folder / name_for(title, "tex")).write_text(tex, encoding="utf-8")
        if r.notes:
            items = "".join(f"<li>{html.escape(note)}</li>" for note in r.notes)
            page.append(f"<h3>Notes</h3><ul>{items}</ul>")
    (folder / "index.html").write_text("\n".join(page), encoding="utf-8")
    info = json.dumps(provenance(study), indent=2, default=str)
    (folder / "run_info.json").write_text(info, encoding="utf-8")
    return folder


def provenance(study: Study) -> dict:
    """What produced a run: settings, data file fingerprints and package versions."""
    versions = {}
    for package in ("numpy", "pandas", "pyfixest", "py-did-multiplegt-dyn", "matplotlib"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": versions,
        "data_sha256": {
            str(src.path): hashlib.sha256(src.path.read_bytes()).hexdigest()
            for src in study.sources
        },
        "settings": vars(study.settings),
    }


# --- Data check -------------------------------------------------------------------------


def control_source(study: Study, name: str) -> str:
    s = study.settings
    if name == "market_barometer":
        return "average z-score of " + ", ".join(s.barometer)
    if "{region}" in name:
        return "regional series, matched on each municipality's region"
    src = study.find(name)
    return f"{src.path.name} ({src.describe()})"


def data_check(study: Study) -> list[tuple[str, str, str]]:
    """What the program found, as (status, item, detail) rows; status is ok, warn or bad.

    Shown before anything is estimated, so problems with the data surface first.
    """
    s, rows, shown = study.settings, [], set()
    for src in study.sources:
        columns = [c for c in src.frame.columns if c not in ("_period", s.unit)]
        listed = ", ".join(columns[:8]) + (", ..." if len(columns) > 8 else "")
        rows.append(("ok", src.path.name, f"{src.describe()}, {len(src.frame):,} rows: {listed}"))
    try:
        bylaws = study.bylaws()
        date = s.effective_date if s.treatment_date == "effective" else s.vote_date
        when = bylaws[date].dropna() if date in bylaws else pd.Series(dtype="datetime64[ns]")
        span = f", {when.min():%Y-%m} to {when.max():%Y-%m}" if len(when) else ""
        rows.append(
            (
                "ok",
                "bylaws",
                f"{len(bylaws)} municipalities with a bylaw, {len(when)} with "
                f"a date in {date}{span}; treatment starts at the {s.treatment_date} date",
            )
        )
    except UserError as exc:
        rows.append(("bad", "bylaws", str(exc)))
    panels = []
    for outcome, freq, number in ((s.starts, "Q", 3), (s.permits, "Y", 4)):
        try:
            p = study.panel(outcome, freq)
        except UserError as exc:
            rows.append(("bad", f"{outcome} (Section {number})", str(exc)))
            continue
        panels.append(p)
        n = p.counts()
        rows.append(
            (
                "ok",
                f"{p.label} (Section {number})",
                f"{FREQ_NAME[freq]}, {p.data['_period'].min()} to {p.data['_period'].max()}: "
                f"{n['municipalities']} municipalities, {n['adopt a bylaw']} adopt a bylaw, "
                f"{n['never treated']} never do, {n['left out']} left out",
            )
        )
        for note in p.notes:
            if note not in shown and not note.startswith(tuple(f"{c} " for c in s.controls)):
                shown.add(note)
                rows.append(("warn", "", note))
    for name in s.controls:
        try:
            where = control_source(study, name)
        except UserError as exc:
            rows.append(("bad", f"control {name}", str(exc)))
            continue
        found = dict.fromkeys(n for p in panels for n in p.notes if n.startswith(f"{name} "))
        details = [where, *(n.removeprefix(f"{name} ") for n in found)]
        rows.append(("warn" if found else "ok", f"control {name}", "; ".join(details)))
    return rows


# --- Menu -------------------------------------------------------------------------------

SECTIONS = {
    "1": "Section 1 · Pre-tests and validity",
    "2": "Section 2 · Basic OLS",
    "3": "Section 3 · Housing starts, quarterly",
    "4": "Section 4 · Building permits, annual",
}
MARKS = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "bad": "[red]✗[/red]"}


class App:
    """The terminal menu. ``stream`` stands in for the keyboard (the tests use it)."""

    def __init__(self, config: Path, console: Console | None = None, stream: Any = None) -> None:
        self.config, self.stream = config, stream
        self.console = console or Console(highlight=False)
        self.study: Study | None = None

    def say(self, *things: Any) -> None:
        self.console.print(*things)

    def ask(self, prompt: str, choices: Sequence[str] = (), default: str = "") -> str:
        shown = f" [dim]({default})[/dim]" if default else ""
        while True:
            line = self.console.input(f"[bold]{prompt}[/bold]{shown}: ", stream=self.stream)
            if self.stream is not None and not line:
                raise EOFError
            answer = line.strip() or default
            if not choices or answer.lower() in choices:
                return answer.lower() if choices else answer
            self.say(f"[red]Please type one of: {', '.join(choices)}[/red]")

    def problem(self, text: str, title: str = "Please fix") -> None:
        self.say(Panel(text, title=title, title_align="left", border_style="red"))

    def load(self) -> bool:
        try:
            with self.console.status("Reading the data files..."):
                self.study = Study.load(self.config)
        except UserError as exc:
            self.problem(str(exc), "Cannot load the data")
            return False
        return True

    def main(self) -> int:
        self.say(
            Panel.fit(
                f"[bold]Zoning bylaws and housing supply[/bold]\nStaggered "
                f"difference-in-differences · settings from {self.config}"
            )
        )
        if not self.load():
            return 1
        self.check()
        actions = {
            "1": self.choose_section1,
            "2": lambda: self.run(["2"]),
            "3": lambda: self.run(["3"]),
            "4": lambda: self.run(["4"]),
            "5": lambda: self.run(["1", "2", "3", "4"]),
            "s": self.settings_menu,
            "d": self.check,
            "r": lambda: self.load() and self.check(),
        }
        while True:
            self.menu()
            try:
                choice = self.ask("Choose", [*actions, "q"])
            except (EOFError, KeyboardInterrupt):
                return 0
            if choice == "q":
                return 0
            try:
                actions[choice]()
            except (EOFError, KeyboardInterrupt):
                self.say("[dim]cancelled[/dim]")
            except UserError as exc:
                self.problem(str(exc))

    def menu(self) -> None:
        s = self.study.settings
        details = {
            "1": "parallel trends, placebo, robustness, balance",
            "2": "; ".join(s.ols_models),
            "3": f"{s.starts}: TWFE event study, did2s, dCDH",
            "4": f"{s.permits}: TWFE event study, did2s, dCDH",
        }
        table = Table(show_header=False, box=None, padding=(0, 2))
        for key, title in SECTIONS.items():
            table.add_row(f"[bold]{key}[/bold]", title, f"[dim]{details[key]}[/dim]")
        table.add_row("[bold]5[/bold]", "Run all four sections", "")
        table.add_row(
            "[bold]s[/bold]",
            "Settings for this session",
            f"[dim]treatment from the {s.treatment_date} date, {len(s.controls)} "
            f"controls, {s.transform} outcomes[/dim]",
        )
        table.add_row(
            "[bold]d[/bold]", "Data check", "[dim]files, outcomes, bylaws, controls[/dim]"
        )
        table.add_row("[bold]r[/bold]", "Reload config.toml and the data", "")
        table.add_row("[bold]q[/bold]", "Quit", "")
        self.say("", Panel(table, title="Main menu", title_align="left"))

    def check(self) -> None:
        with self.console.status("Checking the data..."):
            rows = data_check(self.study)
        table = Table(title="Data check", title_justify="left", box=None, padding=(0, 1))
        table.add_column(width=1)
        table.add_column(style="bold", no_wrap=True)
        table.add_column()
        for status, item, detail in rows:
            table.add_row(MARKS[status], item, detail)
        self.say(table)
        if any(status == "bad" for status, _, _ in rows):
            self.problem("Fix the ✗ items in config.toml or the data, then choose r to reload.")

    def preview(self, outcome: str, freq: str) -> str:
        """What a DiD section will estimate, in plain words, shown before it runs."""
        p = self.study.panel(outcome, freq)
        n, (lo, hi) = p.counts(), p.window
        return (
            f"{p.label}, {FREQ_NAME[freq]}: {n['adopt a bylaw']} adopters and "
            f"{n['never treated']} never-treated municipalities; treatment from the "
            f"{self.study.settings.treatment_date} date; event window {lo} to {hi} "
            f"{PERIODS[freq]}\nControls with fixed effects: {', '.join(p.controls) or 'none'}"
        )

    def choose_section1(self) -> None:
        s = self.study.settings
        both = [(s.starts, "Q"), (s.permits, "Y")]
        choice = self.ask(
            f"Section 1 for 1 = {s.starts}, 2 = {s.permits}, 3 = both", ["1", "2", "3"], "3"
        )
        self.run(["1"], both if choice == "3" else [both[int(choice) - 1]])

    def run(
        self,
        numbers: Sequence[str],
        section1_outcomes: Sequence[tuple[str, str]] = (),
        confirm: bool = True,
    ) -> Path | None:
        """Preview, confirm, estimate, show and save the chosen sections."""
        study, s = self.study, self.study.settings
        starts, permits = (s.starts, "Q"), (s.permits, "Y")
        jobs: list[tuple[str, str, Callable[[], Result]]] = []  # title, preview, estimation
        for number in numbers:
            if number == "1":
                for outcome, freq in section1_outcomes or (starts, permits):
                    run = partial(section1, study, outcome, freq)
                    jobs.append((f"{SECTIONS['1']}: {outcome}", self.preview(outcome, freq), run))
            elif number == "2":
                form = "logs" if s.ols_logs else "levels"
                about = (
                    f"Models ({form}): {'; '.join(s.ols_models)}\nEach one pooled, pooled with "
                    "the controls, and with municipality and period fixed effects"
                )
                jobs.append((SECTIONS["2"], about, partial(section2, study)))
            else:
                outcome, freq = starts if number == "3" else permits
                run = partial(did_section, study, int(number), outcome, freq)
                jobs.append((SECTIONS[number], self.preview(outcome, freq), run))
        for title, about, _ in jobs:
            self.say(Panel(about, title=title, title_align="left"))
        if confirm and self.ask("Run now? (y/n)", ["y", "n"], "y") != "y":
            return None
        results = []
        for title, _, job in jobs:
            with self.console.status(f"{title}: estimating..."):
                try:
                    results.append(job())
                except UserError as exc:
                    self.problem(str(exc), f"{title} could not run")
                except Exception as exc:  # report a failed estimator and keep the session
                    self.problem(f"{type(exc).__name__}: {exc}", f"{title} failed")
        for r in results:
            self.show(r)
        if not results:
            return None
        folder = save(results, study, "+".join(f"section{n}" for n in numbers))
        index = folder / "index.html"
        self.say(
            f"\n[green]Saved[/green] in {folder}\nOpen [link={index.as_uri()}]index.html[/link] "
            "there to see every table and figure."
        )
        return folder

    def show(self, r: Result) -> None:
        self.say("", Panel(f"[bold]{r.title}[/bold]", border_style="cyan"))
        for title, frame in r.tables:
            table = Table(
                title=title, title_justify="left", box=box.SIMPLE_HEAD, min_width=len(title)
            )
            shown = frame.reset_index()
            for col in shown.columns:
                number = shown[col].dtype.kind in "fiu"
                table.add_column(str(col), justify="right" if number else "left", no_wrap=number)
            for row in shown.itertuples(index=False):
                table.add_row(*(cell(v) for v in row))
            self.say(table)
        if r.figures:
            self.say(f"[dim]Figures: {', '.join(title for title, _ in r.figures)}[/dim]")
        for note in r.notes:
            self.say(f"[yellow]note[/yellow] {note}")

    def settings_menu(self) -> None:
        while True:
            s = self.study.settings
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_row("1", "Treatment starts at", f"the {s.treatment_date} date")
            table.add_row("2", "Controls", ", ".join(s.controls) or "none")
            table.add_row("3", "OLS models", "; ".join(s.ols_models))
            table.add_row("4", "Outcome transform", s.transform)
            table.add_row("b", "Back to the main menu", "")
            self.say(
                Panel(
                    table,
                    title="Settings for this session (edit config.toml to keep them)",
                    title_align="left",
                )
            )
            choice = self.ask("Change", ["1", "2", "3", "4", "b"], "b")
            if choice == "b":
                return
            if choice == "1":
                s = replace(
                    s, treatment_date="vote" if s.treatment_date == "effective" else "effective"
                )
            elif choice == "2":
                s = replace(s, controls=self.pick_controls(s))
            elif choice == "3":
                text = self.ask(
                    "Models separated by ; (e.g. permits ~ population; starts ~ pop_density)",
                    default="; ".join(s.ols_models),
                )
                s = replace(s, ols_models=tuple(m.strip() for m in text.split(";") if m.strip()))
            else:
                s = replace(
                    s, transform=self.ask("Transform", ["log", "log1p", "none"], s.transform)
                )
            self.study = Study(check_settings(s), self.study.sources)

    def pick_controls(self, s: Settings) -> tuple[str, ...]:
        """Switch controls on and off by number, or add one by name."""
        configured = load_settings(self.config).controls
        options = list(dict.fromkeys([*configured, *s.controls]))
        chosen = set(s.controls)
        while True:
            for i, name in enumerate(options, 1):
                self.say(f" {i:>2}  [{'x' if name in chosen else ' '}] {name}")
            answer = self.ask("Numbers to switch on or off, or a column name to add (Enter: done)")
            if not answer:
                return tuple(name for name in options if name in chosen)
            for token in answer.replace(",", " ").split():
                if token.isdigit() and 1 <= int(token) <= len(options):
                    chosen ^= {options[int(token) - 1]}
                elif name := clean(token):
                    options += [] if name in options else [name]
                    chosen.add(name)


def cell(value: Any) -> str:
    """A table value for people: 4 significant digits, thousands separators, no 1e-15."""
    if not isinstance(value, float):
        return str(value)
    if np.isnan(value):
        return ""
    if abs(value) >= 1e4:
        return f"{value:,.0f}"
    return "0" if abs(value) < 1e-9 else f"{value:.4g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml", help="settings file")
    parser.add_argument("--demo", action="store_true", help="write made-up data and use it")
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["1", "2", "3", "4", "all"],
        metavar="N",
        help="run sections 1-4 (or all) without the menu",
    )
    args = parser.parse_args(argv)
    config = args.config
    if args.demo:
        from demo import write_demo

        config = write_demo(ROOT / "data" / "demo")
    app = App(config)
    if not args.run:
        return app.main()
    if not app.load():
        return 1
    app.check()
    numbers = ["1", "2", "3", "4"] if "all" in args.run else sorted(set(args.run))
    return 0 if app.run(numbers, confirm=False) else 1


if __name__ == "__main__":
    sys.exit(main())
