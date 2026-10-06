"""Panel difference-in-differences toolkit with an interactive menu.

Usage
-----
    python project.py                                  interactive menu
    python project.py --file data/raw/panel.xlsx       menu with a file preloaded
    python project.py --analyze --file data/raw/panel.csv
    python project.py --settings settings.json --run all
    python project.py --demo                           menu on a synthetic demo panel
    python project.py --self-test

Choose a CSV or Excel panel and read the automatic analysis: panel structure, frequency,
treatment timing, and variables that fixed effects absorb or that duplicate others.
Then assign variable roles and run any stage on its own: descriptives, OLS with
controls, propensity score matching, or a staggered DiD estimator. Every run writes its
tables, figures, a PDF report, the settings it used and provenance to a new folder
under output/. Settings saved from the menu replay the same analysis without prompts.

Install the pinned versions: requirements.txt (requirements-lock.txt has every package
with hashes); requirements-honestdid.txt adds the sensitivity stage, which needs PyTorch.
Required: numpy pandas scipy statsmodels matplotlib pyarrow openpyxl defusedxml
Optional (a stage is skipped without its package): py-did-multiplegt-dyn + polars (dCDH),
csdid, pyfixest (did2s), honestdid (sensitivity), wbgapi (World Bank indicators).

Conventions: every period range is inclusive at both ends. Treatment is a policy date
per unit (absorbing adoption), a 0/1 indicator, or a nonnegative dose. Matching,
Callaway-Sant'Anna, did2s, the TWFE event study and HonestDiD need absorbing binary
treatment; dCDH also handles doses and reversals.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import traceback
import tracemalloc
import uuid
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.diagnostic import het_breuschpagan

try:
    import resource  # Unix only: peak RSS for --profile
except ImportError:  # pragma: no cover - Windows
    resource = None

# --- Constants and settings --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("project")
LOG.setLevel(logging.INFO)
# Run messages go to each run's log file, not to root-logger handlers that third-party
# packages may install.
LOG.propagate = False

DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm")
FREQ_NAMES = {"Y": "annual", "Q": "quarterly", "M": "monthly"}
PERIODS_PER_YEAR = {"Y": 1, "Q": 4, "M": 12}
DEFAULT_EVENT_WINDOW = {"Y": (-4, 6), "Q": (-8, 12), "M": (-12, 24)}
TREATMENT_KINDS = ("date", "indicator", "dose")
REFERENCE_EVENT_TIME = -1  # normalized to zero in event studies; never-treated rows sit here
MIN_RELIABLE_CLUSTERS = 30  # rule of thumb for cluster-robust standard errors
COOK_TOP_N = 10
NEAR_EXACT_R2 = 0.9999  # the analysis flags columns this well explained by other columns
COLLINEARITY_SAMPLE_ROWS = 5000
REPORT_WIDTH, REPORT_LINES = 105, 68
CONSOLE_TABLE_LINES = 30
# Region suffixes in Canadian series names that cover several provinces (FVI_CSCE_ATL).
REGION_ALIASES = {"ATL": ("NB", "NL", "NS", "PE")}
PROVINCE_CODES = {
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB", "new brunswick": "NB",
    "newfoundland and labrador": "NL", "nova scotia": "NS", "ontario": "ON",
    "prince edward island": "PE", "quebec": "QC", "québec": "QC", "saskatchewan": "SK",
    "northwest territories": "NT", "nunavut": "NU", "yukon": "YT",
}  # fmt: skip
# Plot inks: slots 1-2 of a palette validated for colour-vision deficiency, text, grid.
SERIES_COLORS = ("#2a78d6", "#eb6834")
INK, INK_SECONDARY, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
PACKAGES = (
    "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pyarrow", "openpyxl",
    "defusedxml", "polars", "py-did-multiplegt-dyn", "pyfixest", "formulaic", "csdid",
    "drdid", "honestdid", "wbgapi",
)  # fmt: skip
INTERNAL_PREFIX = "_"  # internal panel columns (_unit, _t, _y, ...) start with this
PROFILE_NOTE = (
    "seconds is wall-clock time and cpu_seconds is process CPU time per stage.\n"
    "python_peak_bytes is the tracemalloc peak during the stage; it misses most native\n"
    "allocations (NumPy, BLAS, Arrow). process_maxrss_raw is the cumulative process peak\n"
    "RSS (KiB on Linux, bytes on macOS), not a per-stage figure. The design-matrix and\n"
    "matching budgets are conservative guards, not measured memory limits."
)


@dataclass(frozen=True)
class Config:
    """Environment settings: where files go, resource budgets, diagnostics."""

    output_dir: Path = PROJECT_ROOT / "output"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    seed: int = 20261003
    max_match_bytes: int = 512 * 1024**2
    max_design_bytes: int = 512 * 1024**2
    profile: bool = False
    spreadsheet_safe: bool = False
    http_timeout: float = 60.0
    wdi_max_age_days: int = 30


@dataclass(frozen=True)
class DcdhOptions:
    effects: int = 5
    placebo: int = 3
    same_switchers: bool = True
    effects_equal: bool = True


@dataclass(frozen=True)
class Spec:
    """Everything that defines an analysis. Saved as settings JSON and replayable.

    ``sample`` and ``baseline`` are inclusive period labels such as ("2015Q1", "2019Q4");
    ``event_window`` counts periods relative to adoption (None: a default per frequency).
    """

    file: str | None = None
    sheet: str | None = None
    unit: str | None = None
    time: str | None = None
    outcome: str | None = None
    log_outcome: bool = False
    treatment: str | None = None
    treatment_kind: str | None = None
    adoption_rule: str = "containing"
    regressors: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    combine: tuple[str, ...] = ()
    frequency: str = "native"
    sum_columns: tuple[str, ...] = ()
    sample: tuple[str, str] | None = None
    baseline: tuple[str, str] | None = None
    min_baseline_obs: int | None = None
    event_window: tuple[int, int] | None = None
    caliper_sd: float = 0.2
    control_group: str = "notyettreated"
    # HonestDiD solves many linear programs: about a minute per M-bar value on a quarterly
    # event window with 1000 grid points, so only three M-bar values by default.
    sensitivity_mbar: tuple[float, ...] = (0.5, 1.0, 2.0)
    sensitivity_grid: int = 1000
    left_censored: str = "error"
    missing_treatment_zero: bool = False
    dcdh: DcdhOptions = field(default_factory=DcdhOptions)
    wdi: tuple[str, ...] = ()


SPEC_CHOICES = {
    "treatment_kind": (None, *TREATMENT_KINDS),
    "adoption_rule": ("containing", "next"),
    "frequency": ("native", "annual"),
    "control_group": ("notyettreated", "nevertreated"),
    "left_censored": ("error", "drop", "keep"),
}


def validate_spec(spec: Spec) -> Spec:
    """Check option values that the menu, the CLI and settings files can all set."""
    for name, allowed in SPEC_CHOICES.items():
        if getattr(spec, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {getattr(spec, name)!r}")
    if spec.event_window is not None:
        if len(spec.event_window) != 2:
            raise ValueError("the event window needs two numbers, e.g. -8 12")
        lo, hi = spec.event_window
        if not lo < REFERENCE_EVENT_TIME < 0 <= hi:
            raise ValueError("event window must start before -1 and end at 0 or later")
    if spec.min_baseline_obs is not None and spec.min_baseline_obs < 1:
        raise ValueError("minimum baseline observations must be at least 1")
    if not np.isfinite(spec.caliper_sd) or spec.caliper_sd < 0:
        raise ValueError("caliper must be a nonnegative number")
    mbar = np.asarray(spec.sensitivity_mbar, dtype=float)
    if not mbar.size or not np.isfinite(mbar).all() or (mbar < 0).any():
        raise ValueError("HonestDiD M-bar values must be nonnegative numbers")
    if spec.sensitivity_grid < 10:
        raise ValueError("HonestDiD needs at least 10 grid points")
    if spec.dcdh.effects < 1 or spec.dcdh.placebo < 0:
        raise ValueError("dCDH needs at least one effect and a nonnegative number of placebos")
    for name in ("sample", "baseline"):
        bounds = getattr(spec, name)
        if bounds is not None and len(bounds) != 2:
            raise ValueError(f"{name} needs exactly two period labels")
    return spec


def spec_to_json(spec: Spec) -> dict[str, Any]:
    return asdict(spec)


def spec_from_json(data: dict[str, Any]) -> Spec:
    """Rebuild a Spec from settings JSON, rejecting unknown keys (typos)."""
    names = {f.name for f in fields(Spec)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    values = dict(data)
    for name, value in data.items():
        if name == "dcdh" and isinstance(value, dict):
            values[name] = DcdhOptions(**value)
        elif isinstance(value, list):
            values[name] = tuple(value)
    return validate_spec(Spec(**values))


# --- Small utilities -----------------------------------------------------------------


def _creation_mode() -> int:
    """Permission bits that open() gives new files under the current umask."""
    mask = os.umask(0)  # the umask can only be read by setting it
    os.umask(mask)
    return 0o666 & ~mask


FILE_MODE = _creation_mode()


def atomic_write(path: Path, writer: Callable[[Path], Any]) -> None:
    """Write ``path`` through a sibling temporary file and an atomic rename.

    Readers never see a partial file, even if ``writer`` raises. The file is not fsynced,
    so a power cut can still lose the newest version.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        writer(temp)
        os.chmod(temp, FILE_MODE)  # mkstemp creates owner-only (0600) files
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, default=str, allow_nan=False)
    atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(name: str) -> str:
    """ASCII file-name stem for ``name``, made unique by a short hash."""
    stem = "".join(c if c.isascii() and c.isalnum() else "_" for c in name).strip("_")[:80]
    return f"{stem or 'item'}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"


def spreadsheet_text(value: Any) -> Any:
    """Neutralize text that a spreadsheet would run as a formula (CSV injection)."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def spreadsheet_safe(table: pd.DataFrame) -> pd.DataFrame:
    safe = table.copy()
    for col in safe.columns:
        if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(spreadsheet_text)

    def labels(index: pd.Index) -> pd.Index:
        if isinstance(index, pd.MultiIndex):
            rows = [tuple(spreadsheet_text(x) for x in row) for row in index]
            return pd.MultiIndex.from_tuples(rows, names=index.names)
        return index.map(spreadsheet_text)

    safe.index, safe.columns = labels(safe.index), labels(safe.columns)
    return safe


def display_path(path: str | Path) -> str:
    """Path as recorded in shared outputs: project-relative, else just the file name."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def short_list(items: Sequence[Any], limit: int = 8) -> str:
    items = [str(x) for x in items]
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


class StageSkipped(Exception):
    """A stage could not run; ``kind`` is missing_package, not_applicable or upstream_failed."""

    def __init__(self, reason: str, kind: str = "not_applicable") -> None:
        super().__init__(reason)
        self.reason, self.kind = reason, kind


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def optional(name: str) -> Any:
    """Import an optional package, or raise StageSkipped if it is not installed."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            raise StageSkipped(
                f"optional package {name} is not installed", "missing_package"
            ) from exc
        raise ImportError(f"{name} is installed but its dependency {exc.name} is missing") from exc


# --- Reading files -----------------------------------------------------------------------


def list_sheets(path: Path) -> list[str]:
    with pd.ExcelFile(path) as book:
        return [str(name) for name in book.sheet_names]


def read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV/TSV file or one Excel sheet.

    Excel formulas are read as their cached values; nothing in the workbook is executed.
    Without defusedxml installed, openpyxl parses the workbook without XML-attack
    protection, so a warning is raised.
    """
    suffix = path.suffix.lower()
    if suffix not in DATA_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}; use {', '.join(DATA_SUFFIXES)}")
    if suffix in (".xlsx", ".xlsm"):
        if not has_package("defusedxml"):
            warnings.warn(
                "defusedxml is not installed: the workbook is parsed without "
                "protection against malicious XML",
                stacklevel=2,
            )
        with pd.ExcelFile(path) as book:
            names = [str(name) for name in book.sheet_names]
            name = names[0] if sheet is None else sheet
            if name not in names:
                raise ValueError(f"Sheet {name!r} not found; sheets: {names}")
            data = book.parse(name)
    elif suffix == ".txt":
        data = pd.read_csv(path, sep=None, engine="python")  # sniff the delimiter
    else:
        data = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    data.columns = [str(c).strip() for c in data.columns]
    duplicated = sorted(set(data.columns[data.columns.duplicated()]))
    if duplicated:
        raise ValueError(f"Duplicate column names: {duplicated}")
    data = data.dropna(how="all")  # blank spreadsheet rows
    if data.empty:
        raise ValueError(f"{path.name} has no data rows")
    return data.reset_index(drop=True)


# --- Time handling -------------------------------------------------------------------------

_QUARTER = re.compile(r"(\d{4})\s*[-_/ ]?\s*[Qq]([1-4])|[Qq]([1-4])\s*[-_/ ]?\s*(\d{4})")
_MONTH = re.compile(r"(\d{4})\s*(?:[-_/]|[Mm])\s*(\d{1,2})")
_YEAR = re.compile(r"\d{4}")


def _label_period(text: str) -> tuple[pd.Period, str] | None:
    if match := _QUARTER.fullmatch(text):
        year, quarter = (match[1], match[2]) if match[1] else (match[4], match[3])
        return pd.Period(year=int(year), quarter=int(quarter), freq="Q"), "Q"
    if (match := _MONTH.fullmatch(text)) and 1 <= int(match[2]) <= 12:
        return pd.Period(year=int(match[1]), month=int(match[2]), freq="M"), "M"
    if _YEAR.fullmatch(text):
        return pd.Period(year=int(text), freq="Y"), "Y"
    return None


def _to_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values)
    text = values.astype("string").str.strip().replace("", pd.NA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "could not infer format"
        try:
            return pd.to_datetime(text, errors="raise")
        except (ValueError, TypeError):
            return pd.to_datetime(text, errors="raise", format="mixed")


def _period_series(values: Sequence[Any], index: pd.Index, freq: str) -> pd.Series:
    return pd.Series(pd.PeriodIndex(list(values), freq=freq), index=index)


def date_frequency(dates: pd.Series) -> str:
    """Classify dates as annual, quarterly or monthly from the spacing of distinct values."""
    distinct = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    if len(distinct) < 2:
        raise ValueError("need at least two distinct dates")
    months = np.asarray(distinct.year * 12 + distinct.month)
    if len(np.unique(months)) < len(months):
        raise ValueError(
            "several dates fall in one month: aggregate daily or weekly data "
            "to months, quarters or years first"
        )
    step = int(np.gcd.reduce(np.diff(months)))
    if step % 12 == 0:
        return "Y"
    return "Q" if step % 3 == 0 else "M"


def parse_periods(values: pd.Series) -> tuple[pd.Series, str]:
    """Convert a time column to pandas Periods.

    Parameters
    ----------
    values : pandas.Series
        Four-digit years, quarter labels ("2015Q1", "2015-Q1", "Q1 2015"), month labels
        ("2015-01", "2015M01") or dates.

    Returns
    -------
    periods : pandas.Series
        Periods aligned with ``values`` (NaT where a value is missing).
    freq : str
        "Y", "Q" or "M". Dates are classified by the spacing between distinct values.

    Raises
    ------
    ValueError
        If the values are not a recognised time format, or are daily or weekly.
    """
    present = values.dropna()
    if present.empty:
        raise ValueError("time column has no values")
    if pd.api.types.is_bool_dtype(present):
        raise ValueError("a boolean column is not a time column")
    if pd.api.types.is_numeric_dtype(present):
        numbers = pd.to_numeric(present)
        if not ((numbers % 1 == 0).all() and numbers.between(1000, 9999).all()):
            raise ValueError("numeric time values must be four-digit years")
        years = [pd.Period(year=int(v), freq="Y") if pd.notna(v) else pd.NaT for v in values]
        return _period_series(years, values.index, "Y"), "Y"
    if pd.api.types.is_datetime64_any_dtype(present):
        dates = pd.to_datetime(values)
    else:
        text = values.astype("string").str.strip()
        labels = {u: _label_period(u) for u in text.dropna().unique()}
        if all(labels.values()):
            freqs = {freq for _, freq in labels.values()}
            if len(freqs) > 1:
                raise ValueError(f"mixed period formats ({', '.join(sorted(freqs))})")
            freq = freqs.pop()
            mapped = [labels[v][0] if pd.notna(v) else pd.NaT for v in text]
            return _period_series(mapped, values.index, freq), freq
        dates = _to_dates(text)
    freq = date_frequency(dates)
    return dates.dt.to_period(freq), freq


def ordinals(periods: pd.Series) -> pd.Series:
    """Period ordinals as floats (NaN for NaT); consecutive periods differ by one."""
    raw = np.asarray(periods.array.asi8, dtype=float)
    return pd.Series(np.where(periods.isna(), np.nan, raw), index=periods.index)


def panel_keys(raw: pd.DataFrame, unit: str, time_col: str) -> tuple[pd.DataFrame, str]:
    """Unit ids and periods for every row, checking that they identify rows uniquely."""
    for col in (unit, time_col):
        if col not in raw.columns:
            raise ValueError(f"Column {col!r} not found")
    if unit == time_col:
        raise ValueError("unit and time must be different columns")
    if raw[unit].isna().any():
        raise ValueError(f"{unit}: {int(raw[unit].isna().sum())} rows have no unit id")
    units = raw[unit].astype("string").str.strip()
    if units.eq("").any():
        raise ValueError(f"{unit}: some unit ids are empty")
    periods, freq = parse_periods(raw[time_col])
    if periods.isna().any():
        raise ValueError(f"{time_col}: {int(periods.isna().sum())} rows have no time value")
    keys = pd.DataFrame({"unit": units, "period": periods}, index=raw.index)
    duplicated = keys.duplicated(keep=False)
    if duplicated.any():
        examples = keys[duplicated].drop_duplicates().head(4)
        shown = ", ".join(
            f"{u} {p}" for u, p in zip(examples["unit"], examples["period"], strict=True)
        )
        raise ValueError(
            f"{int(duplicated.sum())} rows share a unit-period (e.g. {shown}); "
            "each unit needs one row per period"
        )
    keys["ord"] = ordinals(periods).astype("int64")
    return keys, freq


# --- Treatment timing ----------------------------------------------------------------------


@dataclass
class Timing:
    """Treatment per row and adoption per unit, derived from the full treatment history."""

    kind: str
    d: pd.Series  # per row; NaN where treatment cannot be known
    adoption: pd.Series  # per unit: first treated Period, NaT if never treated
    binary: bool
    absorbing: bool
    left_censored: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    multiple_dates: list[str] = field(default_factory=list)
    reversals: list[str] = field(default_factory=list)

    @property
    def treated_units(self) -> list[str]:
        return sorted(self.adoption.index[self.adoption.notna()])

    @property
    def never_units(self) -> list[str]:
        return sorted(self.adoption.index[self.adoption.isna()])

    def summary(self) -> str:
        treated = self.adoption.dropna()
        if len(treated):
            cohorts = treated.astype(str).value_counts().sort_index()
            shown = ", ".join(p if n == 1 else f"{p} x{n}" for p, n in cohorts.head(6).items())
            more = f" (+{len(cohorts) - 6} more)" if len(cohorts) > 6 else ""
            parts = [f"{len(treated)} treated, first treated {shown}{more}"]
        else:
            parts = ["no treated units"]
        parts.append(f"{len(self.adoption) - len(treated)} never treated")
        problems = (
            ("left-censored", self.left_censored),
            ("ambiguous adoption", self.ambiguous),
            ("no treatment records", self.absent),
            ("switch off again", self.reversals),
            ("several policy dates", self.multiple_dates),
        )
        parts += [f"{label}: {short_list(units, 5)}" for label, units in problems if units]
        if not self.binary:
            parts.append("dose: dCDH only")
        return "; ".join(parts)


def treatment_timing(
    keys: pd.DataFrame,
    values: pd.Series,
    kind: str,
    freq: str,
    *,
    adoption_rule: str = "containing",
    missing_zero: bool = False,
) -> Timing:
    """Derive treatment per row and adoption per unit from the complete history.

    Timing is computed before rows are dropped for missing outcomes or sample limits,
    so adoption never depends on which outcome rows exist.

    Parameters
    ----------
    keys : pandas.DataFrame
        Columns unit, period, ord from ``panel_keys``, indexed like ``values``.
    values : pandas.Series
        The treatment column.
    kind : str
        "date" (policy date per unit), "indicator" (0/1) or "dose" (>= 0).
    freq : str
        Panel frequency, used to place policy dates in periods.
    adoption_rule : str
        For dates: "containing" treats the period that contains the date as the first
        treated period; "next" uses the first period that starts on or after it.
    missing_zero : bool
        Treat units with no treatment records at all as never treated. Missing values
        inside a unit's history are never zero-filled.
    """
    if kind == "date":
        return _timing_from_dates(keys, values, freq, adoption_rule)
    if kind in ("indicator", "dose"):
        return _timing_from_values(keys, values, kind, freq, missing_zero)
    raise ValueError(f"Unknown treatment kind {kind!r}")


def _timing_from_dates(
    keys: pd.DataFrame, values: pd.Series, freq: str, adoption_rule: str
) -> Timing:
    dates = _to_dates(values)
    per_unit = pd.DataFrame({"unit": keys["unit"], "date": dates}).groupby("unit")["date"]
    first, n_dates = per_unit.min(), per_unit.nunique()
    adoption = first.dt.to_period(freq)
    if adoption_rule == "next":
        starts_on_period = first.isna() | (first <= adoption.dt.start_time)
        adoption = adoption.where(starts_on_period, adoption + 1)
    adopt_ord = ordinals(adoption)
    d = (keys["ord"] >= keys["unit"].map(adopt_ord)).astype(float)
    first_ord = keys.groupby("unit")["ord"].min()
    left = adopt_ord.index[adopt_ord <= first_ord.reindex(adopt_ord.index)]
    return Timing(
        "date",
        d,
        adoption,
        binary=True,
        absorbing=True,
        left_censored=sorted(left),
        multiple_dates=sorted(n_dates.index[n_dates > 1]),
    )


def _timing_from_values(
    keys: pd.DataFrame, values: pd.Series, kind: str, freq: str, missing_zero: bool
) -> Timing:
    v = pd.to_numeric(values, errors="raise").astype(float)
    if np.isinf(v).any() or (v < 0).any():
        raise ValueError("treatment must be finite and nonnegative")
    if kind == "indicator" and not v.dropna().isin([0.0, 1.0]).all():
        raise ValueError("an indicator treatment must be 0/1")
    d = v.copy()
    adoption: dict[str, Any] = {}
    left, ambiguous, absent, reversals = [], [], [], []
    ordered = keys.sort_values(["unit", "ord"])
    for unit, rows in ordered.groupby("unit", sort=True):
        vals = v.loc[rows.index].to_numpy()
        ords = rows["ord"].to_numpy()
        observed = ~np.isnan(vals)
        if not observed.any():
            absent.append(unit)
            adoption[unit] = pd.NaT
            if missing_zero:
                d.loc[rows.index] = 0.0
            continue
        positive = observed & (vals > 0)
        first_pos = int(np.argmax(positive)) if positive.any() else None
        cut = len(vals) if first_pos is None else first_pos
        zeros = np.flatnonzero(observed[:cut] & (vals[:cut] == 0))
        last_zero = int(zeros[-1]) if len(zeros) else None
        if first_pos is None:
            adoption[unit] = pd.NaT
        else:
            adoption[unit] = rows["period"].iloc[first_pos]
            if (observed[first_pos:] & (vals[first_pos:] == 0)).any():
                reversals.append(unit)
            if last_zero is None:
                left.append(unit)  # treated at its first record: adoption could be earlier
            elif ords[first_pos] - ords[last_zero] > 1:
                ambiguous.append(unit)  # missing or absent periods hide the adoption period
        if kind == "indicator" and unit not in reversals:
            # Absorbing treatment implies 0 before an observed 0 and 1 after an observed 1.
            filled = vals.copy()
            head = slice(0, 0 if last_zero is None else last_zero + 1)
            filled[head] = np.where(np.isnan(filled[head]), 0.0, filled[head])
            if first_pos is not None:
                tail = slice(first_pos, None)
                filled[tail] = np.where(np.isnan(filled[tail]), 1.0, filled[tail])
            d.loc[rows.index] = filled
    units = sorted(adoption)
    adoption_series = _period_series(
        [adoption[u] for u in units], pd.Index(units, name="unit"), freq
    )
    binary = kind == "indicator"
    return Timing(
        kind,
        d,
        adoption_series,
        binary=binary,
        absorbing=binary and not reversals,
        left_censored=left,
        ambiguous=ambiguous,
        absent=absent,
        reversals=reversals,
    )


def infer_treatment_kind(values: pd.Series) -> str:
    if column_kind(values) == "date":
        return "date"
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        raise ValueError("a treatment column must hold policy dates or numbers")
    return "indicator" if numbers.isin([0, 1]).all() else "dose"


# --- Dataset analysis ----------------------------------------------------------------------


def _looks_like_dates(sample: pd.Series) -> bool:
    text = sample.astype(str)
    if not text.str.contains(r"\d{4}").mean() >= 0.9:
        return False
    try:
        parsed = _to_dates(text)
    except (ValueError, TypeError, OverflowError):
        return False
    return bool(parsed.notna().mean() >= 0.9)


def column_kind(values: pd.Series) -> str:
    """Classify a column as binary, numeric, date or text."""
    if pd.api.types.is_bool_dtype(values):
        return "binary"
    if pd.api.types.is_numeric_dtype(values):
        present = values.dropna()
        return "binary" if len(present) and present.isin([0, 1]).all() else "numeric"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    sample = values.dropna().head(200)
    return "date" if len(sample) and _looks_like_dates(sample) else "text"


def guess_panel_keys(raw: pd.DataFrame) -> tuple[str | None, str | None]:
    """The (unit, time) column pair that identifies rows uniquely and fills the grid best."""
    times = []
    for col in raw.columns:
        values = raw[col]
        if values.isna().any():
            continue
        if not pd.api.types.is_numeric_dtype(values):
            probe = values.astype("string").drop_duplicates().head(50)
            if not all(_label_period(str(v)) for v in probe) and not _looks_like_dates(probe):
                continue
        try:
            periods, _ = parse_periods(values)
        except (ValueError, TypeError, OverflowError):
            continue
        if periods.nunique() > 1:
            times.append((col, periods))
    units = [
        c
        for c in raw.columns
        if raw[c].notna().all()
        and not pd.api.types.is_float_dtype(raw[c])
        and 2 <= raw[c].nunique() <= len(raw) // 2
    ]
    best, best_fill = (None, None), 0.0
    for time_col, periods in times:
        for unit in units:
            if unit == time_col:
                continue
            ids = raw[unit].astype("string")
            if pd.DataFrame({"u": ids, "p": periods}).duplicated().any():
                continue
            fill = len(raw) / (ids.nunique() * periods.nunique())
            if fill > best_fill:
                best, best_fill = (unit, time_col), fill
    return best


@dataclass
class Family:
    """Columns such as FVI_CSCE_AB, FVI_CSCE_BC whose suffixes name the panel's units."""

    prefix: str
    sources: dict[str, str]  # unit -> source column
    columns: list[str]
    unmatched: list[str]  # units with no source column

    def describe(self) -> str:
        suffixes = ",".join(c[len(self.prefix) + 1 :] for c in self.columns)
        aliases = [
            f"{a} = {'/'.join(m)}"
            for a, m in REGION_ALIASES.items()
            if f"{self.prefix}_{a}" in self.columns or f"{self.prefix}.{a}" in self.columns
        ]
        note = f" ({'; '.join(aliases)})" if aliases else ""
        missing = f"; no column for {short_list(self.unmatched, 5)}" if self.unmatched else ""
        return (
            f"{self.prefix}_{{{suffixes}}} -> one unit-matched column {self.prefix}{note}{missing}"
        )


_FAMILY_MEMBER = re.compile(r"(?P<prefix>.+?)[_.](?P<code>[A-Za-z]{2,4})")


def _unit_code(unit: str) -> str:
    return PROVINCE_CODES.get(unit.strip().lower(), unit).strip().upper()


def find_families(raw: pd.DataFrame, keys: pd.DataFrame) -> dict[str, Family]:
    """Groups of numeric columns whose name suffixes match unit ids or region aliases."""
    units = sorted(keys["unit"].unique())
    by_code: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        by_code[_unit_code(unit)].append(unit)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for col in raw.columns:
        match = _FAMILY_MEMBER.fullmatch(col)
        if match and pd.api.types.is_numeric_dtype(raw[col]):
            groups[match["prefix"]].append((match["code"].upper(), col))
    families = {}
    for prefix, members in groups.items():
        sources: dict[str, str] = {}
        for code, col in members:
            for unit in by_code.get(code, []):
                sources[unit] = col
        for code, col in members:
            for member in REGION_ALIASES.get(code, ()):
                for unit in by_code.get(member, []):
                    sources.setdefault(unit, col)
        if len(members) > 1 and len(set(sources.values())) > 1:
            unmatched = [u for u in units if u not in sources]
            families[prefix] = Family(prefix, sources, [c for _, c in members], unmatched)
    return families


def combine_family(raw: pd.DataFrame, keys: pd.DataFrame, family: Family) -> pd.Series:
    """One column in which each row takes the value from its own unit's source column."""
    combined = pd.Series(np.nan, index=raw.index, dtype=float)
    by_column: dict[str, list[str]] = defaultdict(list)
    for unit, col in family.sources.items():
        by_column[col].append(unit)
    for col, units in by_column.items():
        rows = keys["unit"].isin(units)
        combined[rows] = pd.to_numeric(raw.loc[rows, col], errors="coerce")
    return combined


def exact_dependencies(raw: pd.DataFrame, columns: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Columns (near-)exactly explained by earlier columns: R² >= NEAR_EXACT_R2."""
    data = raw[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2 * len(columns) + 2:
        return []
    if len(data) > COLLINEARITY_SAMPLE_ROWS:
        data = data.sample(COLLINEARITY_SAMPLE_ROWS, random_state=0)
    x = data.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    scale = x.std(axis=0)
    found, basis = [], []
    for j, name in enumerate(columns):
        if scale[j] == 0:
            continue
        target = x[:, j] / scale[j]
        if basis:
            b = x[:, basis] / scale[basis]
            coef, *_ = np.linalg.lstsq(b, target, rcond=None)
            resid = target - b @ coef
            if 1 - (resid @ resid) / (target @ target) >= NEAR_EXACT_R2:
                found.append(
                    (name, [columns[k] for k, c in zip(basis, coef, strict=True) if abs(c) > 1e-6])
                )
                continue
        basis.append(j)
    return found


@dataclass
class DatasetReport:
    """What the analysis found; ``render`` formats it for the terminal and reports."""

    name: str
    rows: int
    kinds: dict[str, str]
    missing: dict[str, float]
    unit: str | None = None
    time: str | None = None
    freq: str | None = None
    span: tuple[str, str] | None = None
    n_units: int = 0
    n_periods: int = 0
    findings: list[str] = field(default_factory=list)
    time_only: list[str] = field(default_factory=list)
    unit_only: list[str] = field(default_factory=list)
    constant: list[str] = field(default_factory=list)
    low_frequency: dict[str, str] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    dependencies: list[tuple[str, list[str]]] = field(default_factory=list)
    treatment_candidates: list[tuple[str, str, str]] = field(default_factory=list)
    treated_only: list[str] = field(default_factory=list)

    def tags(self, column: str) -> str:
        if column == self.time:
            return f"time, {FREQ_NAMES[self.freq]}"
        tags = [self.kinds.get(column, "?")]
        if column == self.unit:
            tags.append("unit")
        if column in self.time_only:
            tags.append("same for all units")
        if column in self.unit_only:
            tags.append("time-invariant")
        if column in self.low_frequency:
            tags.append("lower frequency")
        if self.missing.get(column, 0) > 0:
            tags.append(f"{self.missing[column]:.0%} missing")
        return ", ".join(tags)

    def render(self) -> str:
        lines = [f"File: {self.name} ({self.rows} rows x {len(self.kinds)} columns)"]
        if self.unit and self.time and self.freq:
            lines.append(
                f"Panel: unit = {self.unit} ({self.n_units} units), time = {self.time} "
                f"({FREQ_NAMES[self.freq]}, {self.span[0]} to {self.span[1]}, "
                f"{self.n_periods} periods)"
            )
        lines += [f"  - {f}" for f in self.findings]
        if self.treatment_candidates:
            lines.append("Treatment candidates:")
            grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
            for col, kind, summary in self.treatment_candidates:
                grouped[kind, summary].append(col)  # e.g. several scores with one timing
            lines += [
                f"  {short_list(cols, 4)} [{kind}]: {summary}"
                for (kind, summary), cols in grouped.items()
            ]
        groups = (
            (
                "Same value for every unit in a period (absorbed by period fixed effects; "
                "usable only in pooled OLS)",
                self.time_only,
            ),
            ("Time-invariant within units (absorbed by unit fixed effects)", self.unit_only),
            ("Constant (no information)", self.constant),
            (
                "Defined only for treated units (post-treatment: not a valid control)",
                self.treated_only,
            ),
        )
        for title, cols in groups:
            if cols:
                lines.append(f"{title}:")
                lines += textwrap.wrap(
                    ", ".join(cols), 96, initial_indent="  ", subsequent_indent="  "
                )
        if self.low_frequency:
            lines.append("Lower frequency than the panel (consider Options > frequency: annual):")
            lines += [f"  {c}: {how}" for c, how in self.low_frequency.items()]
        if self.families:
            lines.append("Column families (Set variable roles > combine):")
            lines += [f"  {f.describe()}" for f in self.families.values()]
        if self.dependencies:
            lines.append(
                "Exactly or almost exactly explained by other columns (never include "
                "both in one model):"
            )
            lines += [f"  {c} ~ {' + '.join(parts)}" for c, parts in self.dependencies]
        heavy = {c: s for c, s in self.missing.items() if s >= 0.05}
        if heavy:
            shown = ", ".join(f"{c} {s:.0%}" for c, s in sorted(heavy.items(), key=lambda x: -x[1]))
            lines.append(f"Missing values (>= 5%): {shown}")
        return "\n".join(lines)


def _low_frequency(raw: pd.DataFrame, keys: pd.DataFrame, columns: Sequence[str]) -> dict[str, str]:
    year = keys["period"].dt.year
    grouped = raw[list(columns)].groupby([keys["unit"], year])
    counts, distinct = grouped.count(), grouped.nunique()
    found = {}
    for col in columns:
        present = counts[col] > 0
        if present.sum() == 0:
            continue
        if (counts.loc[present, col] <= 1).all() and (counts[col] > 0).mean() > 0.5:
            found[col] = "one value per year"
        elif (distinct.loc[present, col] <= 1).all() and (counts.loc[present, col] > 1).any():
            found[col] = "the same value in every period of a year"
    return found


def analyze_dataset(
    raw: pd.DataFrame,
    name: str,
    unit: str | None = None,
    time_col: str | None = None,
    adoption_rule: str = "containing",
) -> DatasetReport:
    """Describe a raw file before any roles are chosen.

    Finds the panel keys (or checks the given ones), the frequency and balance, treatment
    candidates and their timing, variables that unit or period fixed effects absorb,
    lower-frequency series, column families that should become one unit-matched column,
    columns that other columns explain exactly, and post-treatment columns.
    """
    report = DatasetReport(
        name,
        len(raw),
        {c: column_kind(raw[c]) for c in raw.columns},
        {c: float(raw[c].isna().mean()) for c in raw.columns},
    )
    if unit is None or time_col is None:
        guessed = guess_panel_keys(raw)
        unit, time_col = unit or guessed[0], time_col or guessed[1]
    if unit is None or time_col is None:
        report.findings.append(
            "No unit/time column pair identifies the rows uniquely; "
            "set them under 'Set variable roles'."
        )
        return report
    try:
        keys, freq = panel_keys(raw, unit, time_col)
    except ValueError as exc:
        report.findings.append(f"{unit} x {time_col} is not a valid panel: {exc}")
        return report
    report.unit, report.time, report.freq = unit, time_col, freq
    periods = keys["period"]
    report.span = (str(periods.min()), str(periods.max()))
    report.n_units = keys["unit"].nunique()
    report.n_periods = int(keys["ord"].max() - keys["ord"].min() + 1)
    span = keys.groupby("unit")["ord"].agg(["min", "max", "count"])
    gaps = span.index[span["count"] < span["max"] - span["min"] + 1]
    late = span.index[span["min"] > keys["ord"].min()]
    early = span.index[span["max"] < keys["ord"].max()]
    if len(gaps) or len(late) or len(early):
        report.findings.append(
            f"Unbalanced: internal gaps {short_list(gaps, 5) or 'none'}; "
            f"late entry {short_list(late, 5) or 'none'}; "
            f"early exit {short_list(early, 5) or 'none'}"
        )
    if report.n_units < MIN_RELIABLE_CLUSTERS:
        report.findings.append(
            f"Only {report.n_units} units: cluster-robust standard errors "
            f"are unreliable below about {MIN_RELIABLE_CLUSTERS} clusters"
        )
    values = [c for c in raw.columns if c not in (unit, time_col)]
    numeric = [c for c in values if report.kinds[c] in ("numeric", "binary")]
    if numeric:
        by_period = raw[numeric].groupby(keys["ord"]).nunique().max()
        by_unit = raw[numeric].groupby(keys["unit"]).nunique().max()
        for col in numeric:
            if raw[col].nunique() <= 1:
                report.constant.append(col)
            elif by_period[col] <= 1:
                report.time_only.append(col)
            elif by_unit[col] <= 1:
                report.unit_only.append(col)
        if freq != "Y":
            report.low_frequency = _low_frequency(raw, keys, numeric)
        report.families = find_families(raw, keys)
        usable = [c for c in numeric if c not in report.constant and report.missing[c] <= 0.5]
        report.dependencies = exact_dependencies(raw, usable)
    main_timing = None
    for col in values:
        kind = {"date": "date", "binary": "indicator"}.get(report.kinds[col])
        if kind is None and report.kinds[col] == "numeric" and _looks_like_dose(raw[col], keys):
            kind = "dose"
        if kind is None:
            continue
        try:
            timing = treatment_timing(keys, raw[col], kind, freq, adoption_rule=adoption_rule)
        except (ValueError, TypeError) as exc:
            report.treatment_candidates.append((col, kind, f"not usable: {exc}"))
            continue
        if kind == "date" and not timing.treated_units:
            continue
        report.treatment_candidates.append((col, kind, timing.summary()))
        if main_timing is None and kind in ("date", "indicator"):
            main_timing = (col, timing)
    if main_timing is not None:
        col, timing = main_timing
        never = keys["unit"].isin(timing.never_units)
        if never.any() and (~never).any():
            report.treated_only = [
                c
                for c in values
                if c != col and raw.loc[never, c].isna().all() and raw.loc[~never, c].notna().any()
            ]
    return report


def _looks_like_dose(values: pd.Series, keys: pd.DataFrame) -> bool:
    """Nonnegative, zero throughout for some units and switching on for others."""
    v = pd.to_numeric(values, errors="coerce")
    if v.isna().all() or (v < 0).any() or not (v == 0).any():
        return False
    unit = keys["unit"]
    has_zero = v.eq(0).groupby(unit).any()
    has_positive = v.gt(0).groupby(unit).any()
    return bool((has_zero & ~has_positive).any() and (has_zero & has_positive).any())


# --- Panel construction --------------------------------------------------------------------


@dataclass
class Panel:
    """Estimation-ready panel.

    ``data`` holds _unit, _period, _t (1, 2, ... over consecutive periods), _y (outcome,
    logged if requested) and, with a treatment, _d, _first (adoption _t, NaN if never
    treated in the sample), _rel (event time binned to the event window; never-treated
    rows sit at the reference period) and _treated. Regressors, controls and covariates
    keep their own column names.
    """

    data: pd.DataFrame
    freq: str
    start_period: pd.Period  # the period with _t == 1
    timing: Timing | None
    outcome_label: str
    window: tuple[int, int]
    notes: list[str]
    signature: str
    adoption_problem: str | None  # why adoption designs cannot run (None: they can)

    @property
    def adoption_ok(self) -> bool:
        return self.adoption_problem is None

    def period_at(self, t: float) -> pd.Period:
        return self.start_period + (int(t) - 1)

    def has_never_treated(self) -> bool:
        return bool(self.timing is not None and (self.data["_treated"] == 0).any())

    def describe(self) -> str:
        d = self.data
        text = (
            f"{d['_unit'].nunique()} units x {d['_t'].max()} {FREQ_NAMES[self.freq]} periods "
            f"({d['_period'].min()} to {d['_period'].max()}), {len(d)} rows"
        )
        if self.timing is not None:
            treated = d.loc[d["_treated"] == 1, "_unit"].nunique()
            text += f"; {treated} treated, {d['_unit'].nunique() - treated} never treated"
        return text


def collapse_annual(work: pd.DataFrame, per_year: int, rules: dict[str, str]) -> pd.DataFrame:
    """Aggregate a sub-annual panel to one row per unit-year.

    ``rules`` maps columns to "sum", "max", "mean" or "first". Sums are kept only for
    complete years, because a partial year would be understated.
    """
    year = work["_period"].dt.year.rename("_year")
    grouped = work.groupby([work["_unit"], year], sort=True)
    out = {}
    for col, rule in rules.items():
        values = grouped[col]
        if rule == "sum":
            out[col] = values.sum(min_count=1).where(values.count() == per_year)
        else:
            out[col] = getattr(values, rule)()
    result = pd.DataFrame(out).reset_index()
    result["_period"] = pd.PeriodIndex(result["_year"].astype(str), freq="Y")
    result["_ord"] = ordinals(result["_period"]).astype("int64")
    return result.drop(columns="_year")


def build_panel(raw: pd.DataFrame, spec: Spec, data_hash: str = "") -> Panel:
    """Turn the raw file and the chosen roles into an estimation-ready Panel.

    Order matters: column families are combined, the frequency is collapsed if asked,
    treatment timing is derived from the full history, and only then are the sample
    limits and left-censoring rules applied.
    """
    missing_roles = [r for r in ("unit", "time", "outcome") if getattr(spec, r) is None]
    if missing_roles:
        raise ValueError(f"Set the {', '.join(missing_roles)} column under 'Set variable roles'")
    keys, freq = panel_keys(raw, spec.unit, spec.time)
    notes: list[str] = []
    combined: dict[str, pd.Series] = {}
    families = find_families(raw, keys) if spec.combine else {}
    for prefix in spec.combine:
        if prefix not in families:
            raise ValueError(f"No column family {prefix!r} matches the unit ids")
        if prefix in raw.columns:
            raise ValueError(f"Cannot combine {prefix}: a column with that name already exists")
        combined[prefix] = combine_family(raw, keys, families[prefix])
        if families[prefix].unmatched:
            notes.append(
                f"{prefix} is missing for {short_list(families[prefix].unmatched)} "
                "(no source column)"
            )

    def column(name: str) -> pd.Series:
        if name in combined:
            return combined[name]
        if name not in raw.columns:
            raise ValueError(f"Column {name!r} not found")
        return raw[name]

    variables = list(dict.fromkeys([*spec.regressors, *spec.controls, *spec.covariates]))
    clashes = [v for v in variables if v.startswith(INTERNAL_PREFIX)]
    if clashes:
        raise ValueError(f"Rename columns starting with '{INTERNAL_PREFIX}': {clashes}")
    outcome = pd.to_numeric(column(spec.outcome), errors="coerce")
    bad = column(spec.outcome).notna() & outcome.isna()
    if bad.any():
        examples = short_list(column(spec.outcome)[bad].astype(str).unique(), 3)
        raise ValueError(f"Outcome {spec.outcome} has non-numeric values: {examples}")
    work = pd.DataFrame(
        {"_unit": keys["unit"], "_period": keys["period"], "_ord": keys["ord"], "_y": outcome},
        index=raw.index,
    )
    kind = None
    if spec.treatment is not None:
        work["_treat"] = column(spec.treatment)
        kind = spec.treatment_kind or infer_treatment_kind(work["_treat"])
    for name in variables:
        values = column(name)
        numeric = pd.to_numeric(values, errors="coerce")
        work[name] = numeric if numeric.notna().sum() == values.notna().sum() else values

    if spec.frequency == "annual" and freq != "Y":
        rules = {"_y": "sum" if spec.outcome in spec.sum_columns else "mean"}
        if kind is not None:
            rules["_treat"] = {"date": "first", "indicator": "max", "dose": "mean"}[kind]
        for name in variables:
            numeric = pd.api.types.is_numeric_dtype(work[name])
            rules[name] = "sum" if name in spec.sum_columns else ("mean" if numeric else "first")
        work = collapse_annual(work, PERIODS_PER_YEAR[freq], rules)
        summed = [c for c in spec.sum_columns if c == spec.outcome or c in variables]
        notes.append(
            f"{FREQ_NAMES[freq]} data collapsed to annual: summed (complete years "
            f"only) {short_list(summed) or 'nothing'}; other numbers averaged"
        )
        freq = "Y"

    timing = None
    if kind is not None:
        tkeys = work[["_unit", "_period", "_ord"]].set_axis(["unit", "period", "ord"], axis=1)
        timing = treatment_timing(
            tkeys,
            work["_treat"],
            kind,
            freq,
            adoption_rule=spec.adoption_rule,
            missing_zero=spec.missing_treatment_zero,
        )
        if timing.absent and not spec.missing_treatment_zero:
            raise ValueError(
                f"No treatment records for {short_list(timing.absent)}. Fix the "
                "data, or set Options > units without treatment records are "
                "never treated."
            )
        if timing.ambiguous:
            raise ValueError(
                f"Adoption period unknown for {short_list(timing.ambiguous)}: "
                "treatment is missing between the last untreated and first "
                "treated period. Fill those records or restrict the sample."
            )
        if timing.absent:
            notes.append(
                f"No treatment records, treated as never treated: {short_list(timing.absent)}"
            )
        if timing.multiple_dates:
            notes.append(
                f"Several policy dates; the earliest is used: {short_list(timing.multiple_dates)}"
            )
        if timing.reversals:
            notes.append(f"Treatment switches off again for {short_list(timing.reversals)}")

    if spec.sample is not None:
        lo, hi = (pd.Period(label, freq=freq) for label in spec.sample)
        work = work[(work["_period"] >= lo) & (work["_period"] <= hi)]
        if work.empty:
            raise ValueError(f"No rows between {lo} and {hi}")
    if spec.log_outcome:
        if (work["_y"].dropna() <= 0).any():
            raise ValueError(f"log({spec.outcome}) needs positive values")
        work = work.assign(_y=np.log(work["_y"]))

    min_ord = int(work["_ord"].min())
    data = pd.DataFrame(
        {
            "_unit": work["_unit"],
            "_period": work["_period"],
            "_t": (work["_ord"] - min_ord + 1).astype("int64"),
            "_y": work["_y"],
        }
    )
    window = spec.event_window or DEFAULT_EVENT_WINDOW[freq]
    adoption_problem = None
    if timing is not None:
        adopt_ord = ordinals(timing.adoption)
        first_ord = work.groupby("_unit")["_ord"].min().reindex(adopt_ord.index)
        left = sorted(set(timing.left_censored) | set(adopt_ord.index[adopt_ord <= first_ord]))
        late = adopt_ord.index[adopt_ord > work["_ord"].max()]
        if len(late):
            notes.append(
                f"First treated after the sample ends (never treated here): {short_list(late)}"
            )
            adopt_ord[late] = np.nan
        if left and spec.left_censored == "error":
            raise ValueError(
                f"Treated at or before their first period, so adoption is "
                f"unknown: {short_list(left)}. Set Options > left-censored units "
                "to drop, or to keep (dCDH only)."
            )
        data["_d"] = timing.d.loc[work.index]
        data["_first"] = data["_unit"].map(adopt_ord - min_ord + 1)
        if left and spec.left_censored == "drop":
            data = data[~data["_unit"].isin(left)]
            notes.append(f"Left-censored units dropped: {short_list(left)}")
        elif left:
            adoption_problem = f"left-censored units kept ({short_list(left)})"
        rel = (data["_t"] - data["_first"]).clip(*window)
        data["_rel"] = rel.fillna(REFERENCE_EVENT_TIME).astype("int64")
        data["_treated"] = data["_first"].notna().astype("int8")
        if not timing.binary:
            adoption_problem = "the treatment is a dose (only dCDH applies)"
        elif not timing.absorbing:
            adoption_problem = "treatment switches off for some units (only dCDH applies)"
        elif not data["_treated"].any():
            adoption_problem = "no unit is treated within the sample"
    else:
        adoption_problem = "no treatment variable is set"
    for name in variables:
        data[name] = work.loc[data.index, name]
    label = f"log({spec.outcome})" if spec.log_outcome else spec.outcome
    payload = json.dumps(spec_to_json(spec), sort_keys=True, default=str) + data_hash
    signature = hashlib.sha256(payload.encode()).hexdigest()[:16]
    start = work.loc[work["_ord"] == min_ord, "_period"].iloc[0]
    return Panel(
        data.sort_values(["_unit", "_t"]),
        freq,
        start,
        timing,
        label,
        window,
        notes,
        signature,
        adoption_problem,
    )


# --- World Bank indicators (optional) ------------------------------------------------------

WDI_CACHE_SCHEMA = 3  # bump when the fetch or filtering logic changes: keys cover only requests
MIN_SAFE_PYARROW = (14, 0, 1)  # CVE-2023-47248: older pyarrow can run code from Parquet files


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def remove_stale_temp_files(directory: Path, max_age_seconds: float = 86400) -> None:
    """Delete atomic_write temporaries left behind by a killed process."""
    cutoff = time.time() - max_age_seconds
    for temp in directory.glob(".tmp-*"):
        with contextlib.suppress(OSError):
            if temp.stat().st_mtime < cutoff:
                temp.unlink()


def fetch_wdi(code: str, years: tuple[int, int], cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One WDI indicator for all economies (aggregates excluded), long format, cached.

    Returns a frame with iso3, year, value (rows without data dropped) and a source record.
    """
    identity = {
        "schema": WDI_CACHE_SCHEMA,
        "code": code,
        "years": list(years),
        "skip_aggregates": True,
    }
    # One file per indicator, overwritten on refresh, so the cache cannot grow without bound.
    path = Path(cfg.cache_dir) / f"wdi_{safe_component(code)}.parquet"
    meta_path = path.with_suffix(".json")
    if path.parent.is_dir():
        remove_stale_temp_files(path.parent)
    try:
        safe_arrow = _version_tuple(importlib.metadata.version("pyarrow")) >= MIN_SAFE_PYARROW
    except importlib.metadata.PackageNotFoundError:
        safe_arrow = False
    if safe_arrow and path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            extracted = datetime.fromisoformat(meta["extracted_at"])
            age = (datetime.now(UTC) - extracted).total_seconds()
            if (
                meta["identity"] == identity
                and age <= cfg.wdi_max_age_days * 86400
                and meta["parquet_sha256"] == fingerprint(path)
            ):
                return pd.read_parquet(path), {**meta, "cache_hit": True}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Ignoring invalid WDI cache for %s: %s", code, exc)
    wb = importlib.import_module("wbgapi")
    saved = wb.get_options
    wb.get_options = {**saved, "timeout": cfg.http_timeout}  # wbgapi sets no timeout itself
    try:
        wide = wb.data.DataFrame(
            code,
            economy="all",
            time=range(years[0], years[1] + 1),
            skipAggs=True,
            skipBlanks=True,
            numericTimeKeys=True,
        )
    finally:
        wb.get_options = saved
    # melt (not stack) keeps the same rows on every pandas version; drop gaps explicitly.
    long = (
        wide.rename_axis("iso3")
        .reset_index()
        .melt(id_vars="iso3", var_name="year", value_name="value")
        .dropna(subset=["value"])
    )
    long["year"] = long["year"].astype("int64")
    long["value"] = pd.to_numeric(long["value"], errors="raise")
    if long.empty:
        raise ValueError(f"WDI {code}: no data for {years[0]}-{years[1]}")
    atomic_write(path, lambda p: long.to_parquet(p, index=False))
    meta = {
        "identity": identity,
        "extracted_at": datetime.now(UTC).isoformat(),
        "parquet_sha256": fingerprint(path),
    }
    write_json(meta_path, meta)
    return long, {**meta, "cache_hit": False}


def add_wdi_columns(
    raw: pd.DataFrame, unit: str, time_col: str, codes: Sequence[str], cfg: Config
) -> tuple[pd.DataFrame, list[dict]]:
    """Add WDI indicators as columns, matched on ISO3 unit ids and the year of each period."""
    keys, _ = panel_keys(raw, unit, time_col)
    if not keys["unit"].str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError(
            "World Bank indicators are matched on ISO3 country codes, and some "
            f"{unit} values are not ISO3 codes"
        )
    years = keys["period"].dt.year
    span = (int(years.min()), int(years.max()))
    out, sources = raw.copy(), []
    for code in codes:
        if code in out.columns:
            raise ValueError(f"A column named {code} already exists")
        long, source = fetch_wdi(code, span, cfg)
        lookup = pd.DataFrame({"iso3": keys["unit"].astype(str), "year": years.astype("int64")})
        merged = lookup.merge(long, on=["iso3", "year"], how="left", validate="many_to_one")
        out[code] = merged["value"].to_numpy()
        sources.append(source)
    return out, sources


# --- Runs and reports ----------------------------------------------------------------------


class Run:
    """One execution: a new output folder, its log file, results and PDF report.

    Use it as a context manager: the log handler is detached, tracemalloc is stopped (only
    if this run started it) and the global NumPy RNG is restored, even if setup fails.
    """

    def __init__(self, cfg: Config, spec: Spec, label: str = "run") -> None:
        self.cfg, self.spec = cfg, spec
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")[:40] or "run"
        self.path = Path(cfg.output_dir) / f"{stamp}_{slug}_{uuid.uuid4().hex[:6]}"
        self.sections: list[tuple[str, str, Any]] = []
        self.issues: list[dict[str, str]] = []
        self.stages: list[dict[str, Any]] = []
        self.incomplete = False
        self._titles: Counter[str] = Counter()
        self._handler: logging.Handler | None = None
        self._rng_state: Any = None
        self._started_tracing = False

    def __enter__(self) -> Run:
        self.path.mkdir(parents=True, exist_ok=False)
        try:
            for part in ("tables", "figures"):
                (self.path / part).mkdir()
            self._handler = logging.FileHandler(self.path / "run.log", encoding="utf-8")
            self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOG.addHandler(self._handler)
            # csdid's serial bootstrap draws from the global NumPy RNG: seed it for this
            # run only and restore the caller's state afterwards.
            self._rng_state = np.random.get_state()
            np.random.seed(self.cfg.seed)
            if self.cfg.profile and not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started_tracing = True
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._release()

    def _release(self) -> None:
        if self._started_tracing:
            tracemalloc.stop()
            self._started_tracing = False
        if self._rng_state is not None:
            np.random.set_state(self._rng_state)
            self._rng_state = None
        if self._handler is not None:
            LOG.removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    @property
    def exit_code(self) -> int:
        return 1 if self.incomplete else 0

    def issue(self, level: str, stage: str, message: str) -> None:
        self.issues.append({"level": level, "stage": stage, "message": message})
        log_level = {"ERROR": logging.ERROR, "NOTE": logging.INFO}.get(level, logging.WARNING)
        LOG.log(log_level, "%s %s: %s", level, stage, message)

    def skip(self, name: str, reason: str, kind: str = "not_applicable") -> None:
        self.stages.append({"name": name, "status": "skipped", "skip_kind": kind, "reason": reason})
        self.incomplete = True
        self.issue("SKIPPED", name, reason)

    def stage(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run one stage and record its status, timing, warnings and errors.

        A stage that cannot run raises StageSkipped; any other exception marks it failed.
        Either makes the run incomplete, because every stage a run attempts was asked
        for. Full tracebacks go to run.log; the report shows one line per error.
        """
        record: dict[str, Any] = {"name": name}
        start, cpu_start = time.perf_counter(), time.process_time()
        if self.cfg.profile and tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        LOG.info("START %s", name)
        counts: Counter[tuple[type, str]] = Counter()

        def count_warning(message: Any, category: type, *_args: Any, **_kwargs: Any) -> None:
            counts[category, f"{category.__name__}: {message}"] += 1

        value = None
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = count_warning  # restored when the block exits
            try:
                value = fn()
                record["status"] = "success"
            except StageSkipped as skip:
                record.update(status="skipped", skip_kind=skip.kind, reason=skip.reason)
                self.incomplete = True
                self.issue("SKIPPED", name, skip.reason)
            except Exception as exc:  # one failed stage must not end the run
                record["status"] = "failed"
                self.incomplete = True
                LOG.error("%s failed\n%s", name, traceback.format_exc())
                self.issue("ERROR", name, f"{type(exc).__name__}: {exc}")
        notices = 0
        for (category, message), count in counts.items():
            text = message if count == 1 else f"{message} (x{count})"
            if issubclass(category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)):
                # API-change notices from installed packages: logged, kept out of the report.
                notices += count
                LOG.info("%s: %s", name, text)
            else:
                self.issue("WARNING", name, text)
        if notices:
            self.issue(
                "NOTE",
                name,
                f"{notices} deprecation notices from installed packages (listed in run.log)",
            )
        record["seconds"] = round(time.perf_counter() - start, 3)
        record["cpu_seconds"] = round(time.process_time() - cpu_start, 3)
        if self.cfg.profile:
            if tracemalloc.is_tracing():
                record["python_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            if resource is not None:
                record["process_maxrss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.stages.append(record)
        return value

    def add(self, title: str, item: Any) -> None:
        """Add a table, figure or text to the report; tables and figures also get files."""
        self._titles[title] += 1
        if self._titles[title] > 1:
            title = f"{title} ({self._titles[title]})"
        name = safe_component(title)
        if isinstance(item, pd.DataFrame):
            atomic_write(self.path / "tables" / f"{name}.csv", item.to_csv)
            if self.cfg.spreadsheet_safe:
                safe = spreadsheet_safe(item)
                atomic_write(self.path / "tables" / f"{name}_spreadsheet.csv", safe.to_csv)
            self.sections.append((title, "table", item))
        elif isinstance(item, Figure):
            atomic_write(
                self.path / "figures" / f"{name}.png",
                lambda p: item.savefig(p, dpi=150, facecolor=SURFACE),
            )
            self.sections.append((title, "figure", item))
        else:
            self.sections.append((title, "text", str(item)))

    def _write_status(self) -> None:
        write_json(self.path / "stages.json", self.stages)
        write_json(self.path / "issues.json", self.issues)

    def finish(self, info: dict[str, Any]) -> None:
        """Write settings, provenance, status files and the PDF report."""
        file = display_path(self.spec.file) if self.spec.file else None
        settings = spec_to_json(replace(self.spec, file=file))
        write_json(self.path / "settings.json", settings)
        write_json(self.path / "provenance.json", info)
        self._write_status()  # first, so a crash while drawing the report keeps the status
        self.stage("PDF report", lambda: build_report(self, info, settings))
        self._write_status()  # again: the report stage adds its own record and warnings


def table_text(table: pd.DataFrame) -> str:
    """A table as text with numbers rounded to 4 decimals (other columns untouched)."""
    shown = table.copy()
    numeric = shown.select_dtypes("number").columns
    shown[numeric] = shown[numeric].round(4)
    return shown.to_string()


def _text_pages(pdf: PdfPages, title: str, text: str) -> None:
    lines: list[str] = []
    for line in str(text).expandtabs(4).splitlines() or [""]:
        lines += textwrap.wrap(
            line, REPORT_WIDTH, replace_whitespace=False, drop_whitespace=False
        ) or [""]
    for start in range(0, len(lines), REPORT_LINES):
        page = Figure(figsize=(8.5, 11))
        # parse_math=False: "$" in data, warnings or paths must not be parsed as TeX.
        page.text(
            0.05, 0.97, title, fontsize=11, weight="bold", va="top", color=INK, parse_math=False
        )
        page.text(
            0.05,
            0.93,
            "\n".join(lines[start : start + REPORT_LINES]),
            family="monospace",
            fontsize=7,
            va="top",
            linespacing=1.2,
            color=INK,
            parse_math=False,
        )
        pdf.savefig(page)


def build_report(run: Run, info: dict[str, Any], settings: dict[str, Any]) -> None:
    def build(path: Path) -> None:
        with PdfPages(path) as pdf:
            unfinished = [s["name"] for s in run.stages if s["status"] != "success"]
            status = "COMPLETE" if not run.incomplete else "INCOMPLETE: " + ", ".join(unfinished)
            stage_table = pd.DataFrame(run.stages).to_string(index=False)
            _text_pages(
                pdf,
                "Run status",
                f"{status}\n\n{stage_table}\n\nNotes, warnings and "
                "errors are listed at the end; full tracebacks are in run.log.",
            )
            if run.cfg.profile:
                _text_pages(pdf, "Profiling", PROFILE_NOTE)
            _text_pages(pdf, "Settings", json.dumps(settings, indent=2, default=str))
            _text_pages(pdf, "Provenance", json.dumps(info, indent=2, default=str))
            for title, kind, item in run.sections:
                if kind == "figure":
                    pdf.savefig(item)  # vector graphics, not a re-embedded PNG
                elif kind == "table":
                    _text_pages(pdf, title, table_text(item))
                else:
                    _text_pages(pdf, title, item)
            issues = "\n".join(f"{x['level']:8} {x['stage']}: {x['message']}" for x in run.issues)
            _text_pages(pdf, "Notes, warnings and errors", issues or "None")

    atomic_write(run.path / "report.pdf", build)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_info() -> dict[str, Any]:
    git = shutil.which("git")  # an explicit PATH lookup, never the working directory
    if git is None:
        return {"commit": None, "reason": "git not found"}

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [git, *args], capture_output=True, text=True, timeout=5, cwd=PROJECT_ROOT, check=False
        )

    try:
        head = run_git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "reason": "not a git repository"}
        # Untracked outputs and caches must not count as changes to the code.
        status = run_git("status", "--porcelain", "--untracked-files=no")
        tracked = run_git("ls-files", "--error-unmatch", Path(__file__).name)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "reason": type(exc).__name__}
    return {
        "commit": head.stdout.strip(),
        "tracked_files_modified": bool(status.stdout.strip()),
        "script_committed": tracked.returncode == 0,
    }


# --- Estimation helpers ---------------------------------------------------------------------

LABELS = {"_y": "outcome", "_d": "treatment (D)"}


def label(name: str) -> str:
    return LABELS.get(name, name)


def complete_cases(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["_unit", *columns]))
    d = data.dropna(subset=cols)
    for col in cols:
        if pd.api.types.is_numeric_dtype(d[col]) and not np.isfinite(d[col].astype(float)).all():
            raise ValueError(f"Nonfinite values in {label(col)}")
    if d.empty or d["_unit"].nunique() < 2:
        raise ValueError("Estimation needs observations from at least two units")
    return d


def drop_absorbed(
    d: pd.DataFrame, columns: Sequence[str], *, unit_fe: bool, period_fe: bool
) -> tuple[list[str], list[str]]:
    """Split columns into those a fixed-effects model can use and those it absorbs."""
    kept, dropped = [], []
    for col in columns:
        if d[col].nunique() <= 1:
            dropped.append(f"{col} (constant in this sample)")
        elif period_fe and d.groupby("_t")[col].nunique().max() <= 1:
            dropped.append(f"{col} (same for every unit in a period: absorbed by period FE)")
        elif unit_fe and d.groupby("_unit")[col].nunique().max() <= 1:
            dropped.append(f"{col} (time-invariant: absorbed by unit FE)")
        else:
            kept.append(col)
    return kept, dropped


def design_matrix(
    d: pd.DataFrame,
    columns: Sequence[str],
    *,
    budget: int,
    unit_fe: bool = False,
    period_fe: bool = False,
    extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Regressor matrix built directly from data, never from formula strings.

    Returns the matrix and its structural columns (constant and fixed-effect dummies).
    Text columns become dummies. Building from columns, not a formula, means column
    names are only ever labels: nothing in a name is evaluated as code.
    """
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(d[c])]
    text = [c for c in columns if c not in numeric]
    n_cols = (
        1
        + len(numeric)
        + sum(d[c].nunique() - 1 for c in text)
        + (d["_unit"].nunique() - 1 if unit_fe else 0)
        + (d["_t"].nunique() - 1 if period_fe else 0)
        + (extra.shape[1] if extra is not None else 0)
    )
    if len(d) * n_cols * 8 > budget:
        raise MemoryError(
            f"A dense {len(d)} x {n_cols} design exceeds the memory budget; "
            "use an estimator that absorbs fixed effects"
        )
    structural = [pd.Series(1.0, index=d.index, name="const")]
    if unit_fe:
        structural.append(
            pd.get_dummies(d["_unit"], prefix="unit", prefix_sep="=", drop_first=True, dtype=float)
        )
    if period_fe:
        structural.append(
            pd.get_dummies(
                d["_period"].astype(str),
                prefix="period",
                prefix_sep="=",
                drop_first=True,
                dtype=float,
            )
        )
    parts = [*structural]
    if extra is not None:
        parts.append(extra)
    parts += [d[c].astype(float).rename(label(c)) for c in numeric]
    parts += [
        pd.get_dummies(
            d[c].astype("string"), prefix=c, prefix_sep="=", drop_first=True, dtype=float
        )
        for c in text
    ]
    x = pd.concat(parts, axis=1)
    names = [
        n
        for part in structural
        for n in (part.columns if isinstance(part, pd.DataFrame) else [part.name])
    ]
    return x, names


def collinear_columns(x: pd.DataFrame, structural: Sequence[str]) -> list[str]:
    """Columns that add no rank beyond the structural columns and the columns before them."""
    if np.linalg.matrix_rank(x.to_numpy(dtype=float)) == x.shape[1]:
        return []
    base = x[list(structural)].to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(base)
    if rank < base.shape[1]:
        return ["the fixed effects themselves"]
    culprits = []
    for col in x.columns:
        if col in structural:
            continue
        trial = np.column_stack([base, x[col].to_numpy(dtype=float)])
        trial_rank = np.linalg.matrix_rank(trial)
        if trial_rank > rank:
            base, rank = trial, trial_rank
        else:
            culprits.append(col)
    return culprits


def fit_ols(
    y: pd.Series, x: pd.DataFrame, groups: pd.Series, structural: Sequence[str], title: str
) -> Any:
    """OLS with unit-clustered SEs; refuses rank-deficient designs instead of using pinv."""
    culprits = collinear_columns(x, structural)
    if culprits:
        raise ValueError(
            f"{title}: coefficients not identified, collinear with the other "
            f"regressors: {short_list([label(c) for c in culprits])}"
        )
    model = sm.OLS(y, x, missing="raise")
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def few_clusters_note(run: Run, stage: str, d: pd.DataFrame) -> None:
    n = d["_unit"].nunique()
    if n < MIN_RELIABLE_CLUSTERS:
        run.issue(
            "WARNING",
            stage,
            f"{n} clusters: cluster-robust standard errors are "
            f"unreliable below about {MIN_RELIABLE_CLUSTERS}; consider a wild cluster "
            "bootstrap",
        )


def require_adoption(panel: Panel) -> None:
    if panel.timing is None:
        raise StageSkipped("set a treatment variable first")
    if not panel.adoption_ok:
        raise StageSkipped(f"needs absorbing binary treatment: {panel.adoption_problem}")


# --- Figures -------------------------------------------------------------------------------


def new_figure(title: str) -> tuple[Figure, Any]:
    """A figure outside pyplot's global registry: no backend switch, nothing to close."""
    fig = Figure(figsize=(8, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    return fig, ax


def event_figure(
    k: Sequence[float],
    estimate: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    title: str,
    xlabel: str,
    reference: int | None = REFERENCE_EVENT_TIME,
) -> Figure:
    k, estimate = np.asarray(k, dtype=float), np.asarray(estimate, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    fig, ax = new_figure(title)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    below, above = np.clip(estimate - lower, 0, None), np.clip(upper - estimate, 0, None)
    ax.errorbar(
        k,
        estimate,
        yerr=[below, above],
        fmt="o",
        color=SERIES_COLORS[0],
        elinewidth=1.5,
        capsize=0,
        markersize=6,
    )
    if reference is not None:
        ax.plot(
            [reference],
            [0],
            "o",
            markersize=6,
            markerfacecolor=SURFACE,
            markeredgecolor=SERIES_COLORS[0],
        )
        ax.axvline(reference + 0.5, color=GRID, linewidth=1)
        ax.annotate(
            "reference period",
            (reference, 0),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7,
            color=INK_SECONDARY,
        )
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel("Estimate with 95% interval", color=INK_SECONDARY)
    return fig


def _period_ticks(ax: Any, t: np.ndarray, labels: Sequence[str]) -> None:
    step = max(1, len(t) // 10)
    ax.set_xticks(t[::step], [str(x) for x in labels][::step], rotation=0)


def trend_figure(panel: Panel) -> Figure:
    d = panel.data
    fig, ax = new_figure(f"Mean {panel.outcome_label} by period")
    periods = d.groupby("_t")["_period"].first()
    if panel.timing is None:
        means = d.groupby("_t")["_y"].mean()
        ax.plot(means.index, means.to_numpy(), color=SERIES_COLORS[0], linewidth=2)
    else:
        means = d.groupby(["_t", "_treated"])["_y"].mean().unstack()
        names = {1: "ever treated", 0: "never treated"}
        for color, group in zip(SERIES_COLORS, (1, 0), strict=True):
            if group in means:
                series = means[group].dropna()
                ax.plot(
                    series.index, series.to_numpy(), color=color, linewidth=2, label=names[group]
                )
                ax.annotate(
                    names[group],
                    (series.index[-1], series.iloc[-1]),
                    textcoords="offset points",
                    xytext=(4, 0),
                    va="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _period_ticks(ax, periods.index.to_numpy(), periods.to_numpy())
    return fig


def rollout_figure(panel: Panel) -> Figure:
    d = panel.data
    fig, ax = new_figure("Share of units treated")
    share = d.groupby("_t")["_d"].mean()
    periods = d.groupby("_t")["_period"].first()
    ax.step(share.index, share.to_numpy(), where="post", color=SERIES_COLORS[0], linewidth=2)
    ax.set_ylim(0, 1)
    _period_ticks(ax, periods.index.to_numpy(), periods.to_numpy())
    return fig


# --- Stages ----------------------------------------------------------------------------------


@dataclass
class MatchResult:
    signature: str
    pairs: pd.DataFrame  # pair, treated, control, distance

    @property
    def units(self) -> list[str]:
        return [*self.pairs["treated"], *self.pairs["control"]]

    @property
    def pair_of_unit(self) -> dict[str, int]:
        out = dict(zip(self.pairs["treated"], self.pairs["pair"], strict=True))
        out.update(zip(self.pairs["control"], self.pairs["pair"], strict=True))
        return out


@dataclass
class EventStudy:
    signature: str
    k: np.ndarray
    beta: np.ndarray
    cov: np.ndarray


@dataclass
class SessionState:
    """Results reused by later stages (matched pairs, event-study coefficients)."""

    matched: MatchResult | None = None
    event_study: EventStudy | None = None


@dataclass
class Context:
    run: Run
    panel: Panel
    spec: Spec
    cfg: Config
    state: SessionState


def stage_descriptives(ctx: Context) -> None:
    p, run = ctx.panel, ctx.run
    d = p.data
    overview = pd.DataFrame(
        {
            "value": {
                "units": d["_unit"].nunique(),
                "periods": int(d["_t"].max()),
                "frequency": FREQ_NAMES[p.freq],
                "first period": str(d["_period"].min()),
                "last period": str(d["_period"].max()),
                "rows": len(d),
                "rows with outcome": int(d["_y"].notna().sum()),
            }
        }
    )
    run.add("Panel structure", overview)
    if p.notes:
        run.add("Panel construction notes", "\n".join(p.notes))
    if p.timing is not None:
        first = d.groupby("_unit")["_first"].first()
        shown = first.map(lambda t: "never" if pd.isna(t) else str(p.period_at(t)))
        run.add("Treatment timing by unit", shown.rename("first treated").to_frame())
    numeric = [
        c
        for c in ["_y", "_d", *ctx.spec.regressors, *ctx.spec.controls]
        if c in d and pd.api.types.is_numeric_dtype(d[c])
    ]
    numeric = list(dict.fromkeys(numeric))
    stats = d[numeric].describe().T.rename(index=label)
    run.add("Summary statistics", stats)
    if len(numeric) > 1:
        run.add("Correlations", d[numeric].corr().rename(index=label, columns=label))
    run.add(f"Mean {p.outcome_label} by period", trend_figure(p))
    if p.timing is not None and p.timing.binary:
        run.add("Share of units treated", rollout_figure(p))


def stage_ols(ctx: Context) -> None:
    p, spec, run = ctx.panel, ctx.spec, ctx.run
    regressors = list(spec.regressors) or (["_d"] if p.timing is not None else [])
    if not regressors:
        raise StageSkipped("choose OLS regressors or a treatment variable")
    controls = [c for c in spec.controls if c not in regressors]
    d = complete_cases(p.data, ["_y", *regressors, *controls])
    few_clusters_note(run, "OLS", d)
    budget = ctx.cfg.max_design_bytes
    models: dict[str, Any] = {}
    x, structural = design_matrix(d, [*regressors, *controls], budget=budget)
    y = d["_y"].rename(p.outcome_label)  # names the dependent variable in summaries
    pooled = fit_ols(y, x, d["_unit"], structural, "Pooled OLS")
    models["Pooled OLS"] = (pooled, d)
    failed = []

    def fit_more(name: str, fit: Callable[[], Any], sample: pd.DataFrame) -> None:
        """Further models are reported even if one of them cannot be estimated."""
        try:
            models[name] = (fit(), sample)
        except (ValueError, MemoryError) as exc:
            failed.append(name)
            run.issue("ERROR", "OLS", f"{name}: {exc}")

    fe_regressors, absorbed = drop_absorbed(d, regressors, unit_fe=True, period_fe=True)
    fe_controls, dropped = drop_absorbed(d, controls, unit_fe=True, period_fe=True)
    for message in [*absorbed, *dropped]:
        run.issue("NOTE", "OLS", f"Two-way FE model omits {message}")
    if fe_regressors:

        def fit_fe() -> Any:
            x_fe, structural_fe = design_matrix(
                d, [*fe_regressors, *fe_controls], budget=budget, unit_fe=True, period_fe=True
            )
            return fit_ols(y, x_fe, d["_unit"], structural_fe, "Two-way FE")

        fit_more("Two-way FE", fit_fe, d)
    cooks = pooled.get_influence().cooks_distance[0]
    top = d.assign(cooks=cooks).nlargest(min(COOK_TOP_N, len(d)), "cooks")
    rest = d.drop(top.index)
    if len(rest) > x.shape[1] + 2 and rest["_unit"].nunique() >= 2:
        fit_more(
            f"Pooled OLS without the {len(top)} highest Cook's D",
            lambda: fit_ols(
                y.loc[rest.index],
                x.loc[rest.index],
                rest["_unit"],
                structural,
                "Pooled OLS without influential rows",
            ),
            rest,
        )
    else:
        run.issue("NOTE", "OLS", "Cook's distance check skipped: too few observations would remain")
    rows = []
    for name, (model, sample) in models.items():
        ci = model.conf_int()
        for reg in [
            c
            for c in model.params.index
            if c not in ("const",) and not c.startswith(("unit=", "period="))
        ]:
            rows.append(
                {
                    "model": name,
                    "term": label(reg),
                    "beta": model.params[reg],
                    "se": model.bse[reg],
                    "ci_lo": ci.loc[reg, 0],
                    "ci_hi": ci.loc[reg, 1],
                    "p": model.pvalues[reg],
                    "N": int(model.nobs),
                    "units": sample["_unit"].nunique(),
                    "R2": model.rsquared,
                }
            )
    table = pd.DataFrame(rows).set_index(["model", "term"])
    run.add(
        f"OLS: {p.outcome_label} (SEs clustered by unit; FE R2 includes the fixed effects)", table
    )
    run.add("Pooled OLS summary", str(pooled.summary()))
    shown = top[["_unit", "_period", "cooks"]].rename(
        columns={"_unit": "unit", "_period": "period"}
    )
    run.add("Most influential observations (Cook's distance, pooled OLS)", shown)
    bp = het_breuschpagan(pooled.resid, pooled.model.exog)[1]
    run.add(
        "Breusch-Pagan diagnostic",
        f"Conventional Breusch-Pagan p-value: {bp:.6g}\n"
        "Diagnostic only: the test ignores panel dependence, and OLS of the outcome on "
        "the treatment is not a causal estimate under staggered adoption.",
    )
    if failed:
        raise RuntimeError(f"not estimated: {', '.join(failed)} (the other models are reported)")


def baseline_window(panel: Panel, spec: Spec) -> tuple[pd.Period, pd.Period]:
    """The matching baseline: the given window, or the year before the earliest adoption."""
    if spec.baseline is not None:
        lo, hi = (pd.Period(label, freq=panel.freq) for label in spec.baseline)
        if lo > hi:
            raise ValueError("baseline start is after its end")
        return lo, hi
    first_t = panel.data["_first"].min()
    if pd.isna(first_t):
        raise ValueError("no unit is treated in the sample")
    hi = panel.period_at(first_t) - 1
    if hi < panel.start_period:
        raise ValueError("there is no period before the earliest adoption")
    length = max(3, PERIODS_PER_YEAR[panel.freq])
    return max(hi - (length - 1), panel.start_period), hi


def baseline_means(panel: Panel, spec: Spec, covariates: Sequence[str]) -> pd.DataFrame:
    d = panel.data
    lo, hi = baseline_window(panel, spec)
    length = (hi - lo).n + 1
    need = spec.min_baseline_obs or length
    if need > length:
        raise ValueError(
            f"minimum baseline observations ({need}) exceeds the baseline "
            f"length ({length} periods, {lo} to {hi})"
        )
    treated_first = d.loc[d["_treated"] == 1].groupby("_unit")["_first"].first()
    adoption = treated_first.map(panel.period_at)
    early = adoption.index[adoption <= hi]
    if len(early):
        raise ValueError(
            f"baseline {lo}-{hi} is not pre-treatment for {short_list(early)}; "
            "choose an earlier baseline"
        )
    window = d[(d["_period"] >= lo) & (d["_period"] <= hi)]
    counts = window.groupby("_unit")[list(covariates)].count()
    eligible = counts.index[counts.ge(need).all(axis=1)]
    means = (
        window[window["_unit"].isin(eligible)]
        .groupby("_unit")[[*covariates, "_treated"]]
        .mean()
        .dropna()
    )
    if means.empty or means["_treated"].nunique() != 2:
        raise ValueError(
            f"matching needs treated and never-treated units with at least "
            f"{need} baseline observations of every covariate ({lo} to {hi})"
        )
    return means


def optimal_pairs(t: np.ndarray, c: np.ndarray, caliper: float, budget: int):
    """Match without replacement: maximize valid pairs, then minimize total distance.

    Every assignment has min(nt, nc) edges. Edges outside the caliper cost more than the
    sum of all possible valid costs, so the solver first maximizes the number of valid
    pairs; invalid edges are removed afterwards. The cost matrix is dense.
    """
    t, c = np.asarray(t, dtype=float), np.asarray(c, dtype=float)
    if not len(t) or not len(c) or not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError("Matching needs nonempty finite scores")
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("Invalid caliper")
    if t.size * c.size * 24 > budget:  # cost matrix, solver copy and masks
        raise MemoryError("Dense matching exceeds the allocation budget")
    cost = np.subtract(t[:, None], c[None, :])
    np.abs(cost, out=cost)
    penalty = (min(len(t), len(c)) + 1) * (caliper + 1)
    if not np.isfinite(penalty):
        raise ValueError("Matching penalty overflow")
    cost[cost > caliper] = penalty
    i, j = linear_sum_assignment(cost)
    valid = cost[i, j] < penalty
    return i[valid], j[valid], cost[i[valid], j[valid]]


def stage_matching(ctx: Context) -> None:
    p, spec, run = ctx.panel, ctx.spec, ctx.run
    require_adoption(p)
    if not p.has_never_treated():
        raise StageSkipped("matching needs never-treated units")
    covariates = list(spec.covariates)
    if not covariates:
        raise StageSkipped("choose matching covariates first")
    text = [c for c in covariates if not pd.api.types.is_numeric_dtype(p.data[c])]
    if text:
        raise ValueError(f"matching covariates must be numeric: {text}")
    # Rank correlation also catches the outcome under another name or after a log.
    copies = [
        c
        for c in covariates
        if c == spec.outcome or abs(p.data[c].corr(p.data["_y"], method="spearman")) > 0.999
    ]
    if copies:
        run.issue(
            "WARNING",
            "Matching",
            f"{short_list(copies)} is (a monotone copy of) the outcome: matching on "
            "pre-treatment outcome levels can bias DiD through regression to the mean "
            "(Daw and Hatfield 2018)",
        )
    means = baseline_means(p, spec, covariates)
    x = sm.add_constant(means[covariates], has_constant="add")
    fit = sm.Logit(means["_treated"], x, missing="raise").fit(disp=0, maxiter=200)
    if not fit.mle_retvals.get("converged", False):
        raise ValueError("propensity logit did not converge")
    score = fit.fittedvalues  # linear predictor x'b, so no probability saturates at 0 or 1
    if not np.isfinite(score).all():
        raise ValueError("nonfinite propensity scores")
    caliper = spec.caliper_sd * score.std()
    treated, control = means[means["_treated"] == 1], means[means["_treated"] == 0]
    i, j, dist = optimal_pairs(
        score[treated.index].to_numpy(),
        score[control.index].to_numpy(),
        caliper,
        ctx.cfg.max_match_bytes,
    )
    if not len(i):
        raise ValueError("no treated unit has a control within the caliper")
    pairs = pd.DataFrame(
        {
            "pair": np.arange(1, len(i) + 1),
            "treated": treated.index[i],
            "control": control.index[j],
            "distance": dist,
        }
    )
    matched = means.loc[[*pairs["treated"], *pairs["control"]]]
    rows = []
    for v in covariates:
        sd = np.sqrt((treated[v].var() + control[v].var()) / 2)
        if not np.isfinite(sd) or sd <= 0:
            run.issue("WARNING", "Matching", f"undefined pre-match SD for {v}")
            sd = np.nan
        after = matched.groupby("_treated")[v].mean()
        rows.append(
            {
                "covariate": v,
                "SMD before": (treated[v].mean() - control[v].mean()) / sd,
                "SMD after": (after.get(1, np.nan) - after.get(0, np.nan)) / sd,
            }
        )
    lo, hi = baseline_window(p, spec)
    run.add("Matched pairs", pairs.set_index("pair"))
    run.add(
        "Covariate balance (SMD with the fixed pre-match SD)",
        pd.DataFrame(rows).set_index("covariate"),
    )
    run.add(
        "Propensity model",
        f"Baseline {lo} to {hi}; caliper {spec.caliper_sd} SD of the "
        f"linear score. Matched {len(i)} of {len(treated)} eligible treated units.\n\n"
        f"{fit.summary()}",
    )
    ctx.state.matched = MatchResult(p.signature, pairs)


def _dcdh(
    ctx: Context,
    data: pd.DataFrame,
    title: str,
    *,
    controls: Sequence[str] = (),
    cluster: pd.Series | None = None,
    **options: Any,
) -> None:
    """Run did_multiplegt_dyn on internal-name aliases and report its table and tests."""
    if ctx.panel.timing is None:
        raise StageSkipped("set a treatment variable first")
    pl = optional("polars")
    dyn = optional("did_multiplegt_dyn")
    unknown = data["_d"].isna()
    if unknown.any():
        ctx.run.issue("NOTE", title, f"{int(unknown.sum())} rows with unknown treatment omitted")
        data = data[~unknown]
    est = pd.DataFrame(
        {
            "g": pd.factorize(data["_unit"], sort=True)[0],
            "t": data["_t"],
            "y": data["_y"],
            "d": data["_d"],
        }
    )
    names = {f"x{i}": c for i, c in enumerate(controls)}
    for alias, col in names.items():
        est[alias] = data[col].astype(float)
    if cluster is not None:
        est["cl"] = cluster.loc[data.index].to_numpy()
    for col in est:
        if np.isinf(est[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"infinite values in dCDH input {names.get(col, col)}")
    opts = {**asdict(ctx.spec.dcdh), **options}
    if names:
        opts["controls"] = list(names)
    if cluster is not None:
        opts["cluster"] = "cl"
    with contextlib.redirect_stdout(io.StringIO()):  # the package prints its own tables
        model = dyn.DidMultiplegtDyn(
            df=pl.from_pandas(est), outcome="y", group="g", time="t", treatment="d", **opts
        )
        model.fit()
        table = model.summary()
    if not isinstance(table, pd.DataFrame) or not {"Block", "Estimate", "SE"} <= set(table):
        raise RuntimeError(
            "Unsupported did_multiplegt_dyn version: summary() should return a "
            "table with Block, Estimate and SE"
        )
    table = table.set_index("Block")
    result = getattr(model, "result", {})
    raw = result.get("did_multiplegt_dyn", {}) if isinstance(result, dict) else {}
    ctx.run.add(f"{title}: {ctx.panel.outcome_label}", table)
    tests = {
        k: raw[k] for k in ("p_jointplacebo", "p_jointeffects", "p_equality_effects") if k in raw
    }
    if tests:
        ctx.run.add(f"{title}: joint tests", pd.Series(tests, name="p-value").to_frame())
    counts = [
        f"{est['g'].nunique()} units",
        f"{len(est)} rows",
        f"{int(est['y'].notna().sum())} with the outcome",
    ]
    if names:
        counts.append(f"{int(est[list(names)].notna().all(axis=1).sum())} with every control")
    level = "matched pair" if cluster is not None else "unit"
    ctx.run.add(f"{title}: sample", f"{', '.join(counts)}. Standard errors clustered by {level}.")
    events = table[table.index.astype(str).str.fullmatch(r"(?:Effect|Placebo)_\d+")]
    if not events.empty and {"LB CI", "UB CI"} <= set(events):
        blocks = events.index.astype(str)
        k = blocks.str.extract(r"(\d+)")[0].astype(int).to_numpy()
        k = np.where(blocks.str.startswith("Placebo"), -k, k)
        ctx.run.add(
            f"{title}: horizons",
            event_figure(
                k,
                events["Estimate"],
                events["LB CI"],
                events["UB CI"],
                f"{title} by horizon",
                "dCDH horizon as reported: Effect_l at +l, Placebo_l at -l (not TWFE event time)",
                reference=None,
            ),
        )


def stage_dcdh(ctx: Context) -> None:
    _dcdh(ctx, ctx.panel.data, "dCDH")


def stage_dcdh_never(ctx: Context) -> None:
    title = "dCDH (never-switchers as controls)"
    try:
        _dcdh(ctx, ctx.panel.data, title, only_never_switchers=True)
    except Exception as exc:
        # py-did-multiplegt-dyn 0.1.9 crashes (missing column T_max_XX) when this option is
        # combined with same_switchers; retry without it rather than lose the estimate.
        if "T_max_XX" not in str(exc) or not ctx.spec.dcdh.same_switchers:
            raise
        ctx.run.issue(
            "NOTE",
            title,
            "did_multiplegt_dyn fails when only_never_switchers is "
            "combined with same_switchers, so this estimate uses same_switchers=False: "
            "the switchers behind each horizon may differ",
        )
        _dcdh(ctx, ctx.panel.data, title, only_never_switchers=True, same_switchers=False)


def stage_dcdh_matched(ctx: Context) -> None:
    matched = ctx.state.matched
    if matched is None or matched.signature != ctx.panel.signature:
        raise StageSkipped(
            "run propensity score matching with the current settings first", "upstream_failed"
        )
    d = ctx.panel.data
    d = d[d["_unit"].isin(matched.units)]
    # Matching without replacement: cluster on the matched pair (Abadie and Spiess 2022).
    _dcdh(ctx, d, "dCDH (matched sample)", cluster=d["_unit"].map(matched.pair_of_unit))


def stage_dcdh_controls(ctx: Context) -> None:
    d = ctx.panel.data
    numeric = [c for c in ctx.spec.controls if pd.api.types.is_numeric_dtype(d[c])]
    skipped = [c for c in ctx.spec.controls if c not in numeric]
    usable, absorbed = drop_absorbed(d, numeric, unit_fe=True, period_fe=True)
    for message in [*absorbed, *(f"{c} (not numeric)" for c in skipped)]:
        ctx.run.issue("NOTE", "dCDH with controls", f"omits {message}")
    if not usable:
        raise StageSkipped("no control varies across units and over time")
    _dcdh(ctx, d, "dCDH (with controls)", controls=usable)


def stage_csdid(ctx: Context) -> None:
    p, spec, run = ctx.panel, ctx.spec, ctx.run
    require_adoption(p)
    optional("csdid")
    from csdid.att_gt import ATTgt

    d = p.data.dropna(subset=["_y"])
    n_periods = p.data["_t"].nunique()
    complete = d.groupby("_unit")["_t"].nunique()
    balanced = complete.index[complete == n_periods]
    dropped = sorted(set(p.data["_unit"]) - set(balanced))
    if dropped:
        run.issue(
            "NOTE",
            "Callaway-Sant'Anna",
            f"needs a balanced panel: omitted units without "
            f"the outcome in every period: {short_list(dropped)}",
        )
    d = d[d["_unit"].isin(balanced)]
    if d.empty:
        raise ValueError(
            "no unit has the outcome in every period; for a lower-frequency "
            "outcome set Options > frequency to annual"
        )
    est = pd.DataFrame(
        {
            "id": pd.factorize(d["_unit"], sort=True)[0] + 1,
            "t": d["_t"],
            "y": d["_y"],
            "g": d["_first"].fillna(0).astype(int),
        }
    )
    if spec.control_group == "nevertreated" and not (est["g"] == 0).any():
        raise StageSkipped("no never-treated units for the never-treated control group")
    few_clusters_note(run, "Callaway-Sant'Anna", d)
    lo, hi = p.window
    with contextlib.redirect_stdout(io.StringIO()):
        att = ATTgt(
            yname="y", tname="t", idname="id", gname="g", data=est, control_group=spec.control_group
        )
        att.fit(est_method="dr")
        att.aggte("dynamic", min_e=lo, max_e=hi)
    res = getattr(att, "atte", None)
    if not isinstance(res, dict) or not {"egt", "att_egt", "se_egt"} <= set(res):
        raise RuntimeError(
            "Unsupported csdid version: expected ATTgt.atte with egt, att_egt "
            "and se_egt after aggte('dynamic')"
        )

    def scalar(value: Any) -> float:
        return float(np.ravel([value])[0])

    crit = res.get("crit_val_egt")
    crit, band = (1.96, "pointwise 95%") if crit is None else (scalar(crit), "simultaneous 95%")
    table = pd.DataFrame(
        {
            "event time": np.ravel(res["egt"]).astype(int),
            "ATT": np.ravel(res["att_egt"]),
            "se": np.ravel(res["se_egt"]),
        }
    )
    table["band_lo"] = table["ATT"] - crit * table["se"]
    table["band_hi"] = table["ATT"] + crit * table["se"]
    run.add(f"Callaway-Sant'Anna dynamic effects: {p.outcome_label}", table.set_index("event time"))
    run.add(
        "Callaway-Sant'Anna summary",
        f"Overall ATT {scalar(res['overall_att']):.6g} (se {scalar(res['overall_se']):.6g}); "
        f"control group {spec.control_group}; doubly robust; {est['id'].nunique()} units, "
        f"{len(est)} rows. Bands are {band} (critical value {crit:.3f}).",
    )
    run.add(
        "Callaway-Sant'Anna event study",
        event_figure(
            table["event time"],
            table["ATT"],
            table["band_lo"],
            table["band_hi"],
            "Callaway-Sant'Anna dynamic ATT",
            "Periods relative to adoption",
            reference=None,
        ),
    )


def stage_did2s(ctx: Context) -> None:
    p, run = ctx.panel, ctx.run
    require_adoption(p)
    pf = optional("pyfixest")
    d = complete_cases(p.data, ["_y", "_d", "_rel"])
    few_clusters_note(run, "did2s", d)
    est = pd.DataFrame(
        {
            "y": d["_y"],
            "d": d["_d"],
            "rel": d["_rel"],
            "unit": pd.factorize(d["_unit"], sort=True)[0],
            "t": d["_t"],
        }
    )
    fit = pf.did2s(
        est,
        yname="y",
        first_stage="~ 0 | unit + t",
        second_stage=f"~ i(rel, ref={REFERENCE_EVENT_TIME})",
        treatment="d",
        cluster="unit",
    )
    tidy = fit.tidy()
    if not isinstance(tidy, pd.DataFrame) or not {"Estimate", "2.5%", "97.5%"} <= set(tidy):
        raise RuntimeError("Unsupported pyfixest version: tidy() should give Estimate, 2.5%, 97.5%")
    k = tidy.index.astype(str).str.extract(r"rel::(-?\d+)")[0].astype(float).to_numpy()
    run.add(f"Gardner did2s event study: {p.outcome_label}", tidy.assign(event_time=k))
    run.add(
        "Gardner did2s sample",
        f"{est['unit'].nunique()} units, {len(est)} rows; endpoints "
        f"{p.window} pool all more distant periods; SEs clustered by unit.",
    )
    run.add(
        "Gardner did2s figure",
        event_figure(
            k,
            tidy["Estimate"],
            tidy["2.5%"],
            tidy["97.5%"],
            "Gardner two-stage DiD",
            "Periods relative to adoption (endpoints pooled)",
        ),
    )


def stage_twfe_es(ctx: Context) -> None:
    p, run = ctx.panel, ctx.run
    require_adoption(p)
    d = complete_cases(p.data, ["_y", "_rel"])
    few_clusters_note(run, "TWFE event study", d)
    lo, hi = p.window
    present = set(d["_rel"].unique())
    events = pd.DataFrame(
        {
            f"rel={k}": (d["_rel"] == k).astype(float)
            for k in range(lo, hi + 1)
            if k != REFERENCE_EVENT_TIME and k in present
        },
        index=d.index,
    )
    if events.empty:
        raise ValueError("no treated observations away from the reference period")
    x, structural = design_matrix(
        d, [], budget=ctx.cfg.max_design_bytes, unit_fe=True, period_fe=True, extra=events
    )
    model = fit_ols(d["_y"], x, d["_unit"], structural, "TWFE event study")
    names = list(events.columns)
    k = np.array([int(n.split("=")[1]) for n in names])
    ci = model.conf_int().loc[names]
    table = pd.DataFrame(
        {
            "event time": k,
            "beta": model.params[names].to_numpy(),
            "se": model.bse[names].to_numpy(),
            "ci_lo": ci[0].to_numpy(),
            "ci_hi": ci[1].to_numpy(),
        }
    ).set_index("event time")
    run.add(f"TWFE event study (diagnostic): {p.outcome_label}", table)
    run.add(
        "TWFE event study interpretation",
        "Naive TWFE is a diagnostic: with staggered "
        "adoption and heterogeneous effects its coefficients mix comparisons (Sun and "
        f"Abraham 2021). Endpoints {p.window} pool all more distant periods; the reference "
        f"is {REFERENCE_EVENT_TIME}; event time 0 is the first treated period.",
    )
    run.add(
        "TWFE event study figure",
        event_figure(
            k,
            table["beta"],
            table["ci_lo"],
            table["ci_hi"],
            "TWFE event study (diagnostic)",
            "Periods relative to adoption (endpoints pooled)",
        ),
    )
    cov = model.cov_params().loc[names, names].to_numpy()
    ctx.state.event_study = EventStudy(p.signature, k, model.params[names].to_numpy(), cov)


def sensitivity_layout(k: np.ndarray, beta: np.ndarray, cov: np.ndarray):
    """Order event-study coefficients for HonestDiD and check they form a usable layout."""
    order = np.argsort(k)
    k, beta, cov = np.asarray(k)[order], np.asarray(beta)[order], np.asarray(cov)[order][:, order]
    pre, post = k[k < 0], k[k >= 0]
    if not len(pre) or not len(post) or post[0] != 0:
        raise ValueError("sensitivity needs pre-period coefficients and event time 0")
    expected_pre = np.arange(pre[0], REFERENCE_EVENT_TIME)
    if not np.array_equal(pre, expected_pre) or not np.array_equal(post, np.arange(post[-1] + 1)):
        raise ValueError(
            f"event times must be consecutive (apart from the reference "
            f"{REFERENCE_EVENT_TIME}); got {k.tolist()}"
        )
    if cov.shape != (len(beta), len(beta)) or not (
        np.isfinite(beta).all() and np.isfinite(cov).all()
    ):
        raise ValueError("invalid or nonfinite sensitivity inputs")
    target = np.zeros(len(post))
    target[0] = 1
    return beta, cov, len(pre), len(post), target


def stage_honestdid(ctx: Context) -> None:
    p, run = ctx.panel, ctx.run
    require_adoption(p)
    hd = optional("honestdid")
    es = ctx.state.event_study
    if es is None or es.signature != p.signature:
        raise StageSkipped(
            "run the TWFE event study with the current settings first", "upstream_failed"
        )
    beta, cov, n_pre, n_post, target = sensitivity_layout(es.k, es.beta, es.cov)
    run.issue(
        "WARNING",
        "HonestDiD",
        "Bounds use naive TWFE coefficients with pooled endpoint "
        "bins; they are not bounds on the dCDH or Callaway-Sant'Anna estimands",
    )
    args = {
        "betahat": beta,
        "sigma": cov,
        "numPrePeriods": n_pre,
        "numPostPeriods": n_post,
        "l_vec": target,
    }
    try:
        robust = hd.createSensitivityResults_relativeMagnitudes(
            **args,
            Mbarvec=np.asarray(ctx.spec.sensitivity_mbar, dtype=float),
            gridPoints=ctx.spec.sensitivity_grid,
        )
        original = hd.constructOriginalCS(**args)
    except TypeError as exc:
        if "0-dimensional" in str(exc):
            raise RuntimeError(
                "honestdid 0.1.1 fails on NumPy >= 2.4; install the pinned versions "
                "in requirements-honestdid.txt (NumPy 2.3)"
            ) from exc
        raise
    run.add(
        "HonestDiD relative-magnitudes bounds for event time 0",
        pd.concat([pd.DataFrame(original), pd.DataFrame(robust)], ignore_index=True),
    )


@dataclass(frozen=True)
class StageDef:
    key: str
    title: str
    fn: Callable[[Context], Any]
    needs: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    did: bool = False


STAGES = (
    StageDef("descriptives", "Descriptives", stage_descriptives),
    StageDef("ols", "OLS with controls (pooled and two-way FE)", stage_ols),
    StageDef(
        "matching",
        "Propensity score matching",
        stage_matching,
        needs=("adoption", "never_treated", "covariates"),
    ),
    StageDef(
        "dcdh",
        "dCDH (de Chaisemartin-D'Haultfoeuille)",
        stage_dcdh,
        needs=("treatment",),
        packages=("did_multiplegt_dyn", "polars"),
        did=True,
    ),
    StageDef(
        "dcdh_never",
        "dCDH, never-switchers as controls",
        stage_dcdh_never,
        needs=("treatment",),
        packages=("did_multiplegt_dyn", "polars"),
        did=True,
    ),
    StageDef(
        "dcdh_matched",
        "dCDH on the matched sample",
        stage_dcdh_matched,
        needs=("treatment", "matched"),
        packages=("did_multiplegt_dyn", "polars"),
        did=True,
    ),
    StageDef(
        "dcdh_controls",
        "dCDH with controls",
        stage_dcdh_controls,
        needs=("treatment", "controls"),
        packages=("did_multiplegt_dyn", "polars"),
        did=True,
    ),
    StageDef(
        "csdid",
        "Callaway-Sant'Anna",
        stage_csdid,
        needs=("adoption",),
        packages=("csdid",),
        did=True,
    ),
    StageDef(
        "did2s",
        "Gardner two-stage DiD (did2s)",
        stage_did2s,
        needs=("adoption",),
        packages=("pyfixest",),
        did=True,
    ),
    StageDef(
        "twfe_es", "TWFE event study (diagnostic)", stage_twfe_es, needs=("adoption",), did=True
    ),
    StageDef(
        "honestdid",
        "HonestDiD sensitivity (on the TWFE event study)",
        stage_honestdid,
        needs=("adoption", "event_study"),
        packages=("honestdid",),
        did=True,
    ),
)
STAGE_BY_KEY = {s.key: s for s in STAGES}
PREREQUISITES = {"dcdh_matched": "matching", "honestdid": "twfe_es"}


def with_prerequisites(keys: Sequence[str]) -> list[str]:
    """Requested stages plus the stages they need, in registry order."""
    wanted = set(STAGE_BY_KEY) if "all" in keys else set(keys)
    for key in list(wanted):
        if key in PREREQUISITES:
            wanted.add(PREREQUISITES[key])
    return [s.key for s in STAGES if s.key in wanted]


def availability(
    stage: StageDef,
    spec: Spec,
    panel: Panel | Exception,
    state: SessionState,
    planned: Sequence[str] = (),
) -> str | None:
    """Why ``stage`` cannot run now (None if it can)."""
    missing = [name for name in stage.packages if not has_package(name)]
    if missing:
        return f"install {', '.join(missing)}"
    if isinstance(panel, Exception):
        return f"panel not ready: {panel}"
    for need in stage.needs:
        if need == "treatment" and panel.timing is None:
            return "set a treatment variable"
        if need == "adoption" and not panel.adoption_ok:
            return f"needs absorbing binary treatment: {panel.adoption_problem}"
        if need == "never_treated" and not panel.has_never_treated():
            return "needs never-treated units"
        if need == "covariates" and not spec.covariates:
            return "choose matching covariates"
        if need == "controls" and not spec.controls:
            return "choose controls"
        if (
            need == "matched"
            and "matching" not in planned
            and not (state.matched and state.matched.signature == panel.signature)
        ):
            return "run propensity score matching first"
        if (
            need == "event_study"
            and "twfe_es" not in planned
            and not (state.event_study and state.event_study.signature == panel.signature)
        ):
            return "run the TWFE event study first"
    return None


# --- Session ---------------------------------------------------------------------------------


def provenance(session: Session) -> dict[str, Any]:
    cfg, spec = session.cfg, session.spec
    script = Path(__file__).resolve()
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
        "packages": package_versions(),
        "script": {"file": display_path(script), "sha256": fingerprint(script)},
        "data": {
            "file": display_path(spec.file) if spec.file else None,
            "sheet": spec.sheet,
            "sha256": session.data_hash or None,
            "rows": None if session.raw is None else len(session.raw),
        },
        "world_bank_sources": session.wdi_sources,
        "git": git_info(),
        "seed": cfg.seed,
        "seed_scope": "Global NumPy RNG, seeded at the start of each run and restored after "
        "(csdid's serial bootstrap draws from it); honestdid seeds itself.",
        "config": {
            "output_dir": display_path(cfg.output_dir),
            "cache_dir": display_path(cfg.cache_dir),
            "max_match_bytes": cfg.max_match_bytes,
            "max_design_bytes": cfg.max_design_bytes,
            "profile": cfg.profile,
            "spreadsheet_safe": cfg.spreadsheet_safe,
        },
    }


class Session:
    """The loaded file, its analysis, the current Spec, and results shared across runs."""

    def __init__(self, cfg: Config, spec: Spec | None = None) -> None:
        self.cfg = cfg
        self.spec = validate_spec(spec or Spec())
        self.raw: pd.DataFrame | None = None
        self.data_hash = ""
        self.report: DatasetReport | None = None
        self.state = SessionState()
        self.wdi_sources: list[dict] = []
        self._panel: tuple[str, Panel | Exception] | None = None

    def load(
        self, path: str | Path, sheet: str | None = None, *, reset_roles: bool = False
    ) -> DatasetReport:
        """Read and analyze a file. Roles already set are kept unless ``reset_roles``;
        unset unit, time, treatment and column families are filled from the analysis."""
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        raw = read_table(path, sheet)
        self.raw, self.data_hash = raw, fingerprint(path)
        base = Spec(dcdh=self.spec.dcdh) if reset_roles else self.spec
        self.spec = validate_spec(replace(base, file=str(path), sheet=sheet))
        self.state, self.wdi_sources, self._panel = SessionState(), [], None
        if self.spec.wdi and self.spec.unit and self.spec.time:
            self.raw, self.wdi_sources = add_wdi_columns(
                raw, self.spec.unit, self.spec.time, self.spec.wdi, self.cfg
            )
        self.analyze()
        self._suggest_roles()
        return self.report

    def analyze(self) -> DatasetReport:
        if self.raw is None:
            raise ValueError("Choose a data file first")
        unit = self.spec.unit if self.spec.unit in self.raw.columns else None
        time_col = self.spec.time if self.spec.time in self.raw.columns else None
        name = Path(self.spec.file).name if self.spec.file else "data"
        if self.spec.sheet:
            name += f" [{self.spec.sheet}]"
        self.report = analyze_dataset(self.raw, name, unit, time_col, self.spec.adoption_rule)
        return self.report

    def _suggest_roles(self) -> None:
        r, s = self.report, self.spec
        changes: dict[str, Any] = {}
        if s.unit is None and r.unit:
            changes["unit"] = r.unit
        if s.time is None and r.time:
            changes["time"] = r.time
        if not s.combine and r.families:
            changes["combine"] = tuple(r.families)
        if s.treatment is None:
            usable = [
                (c, k)
                for c, k, text in r.treatment_candidates
                if k in ("date", "indicator") and not text.startswith("not usable")
            ]
            usable.sort(key=lambda x: x[1] != "date")  # prefer policy dates
            if usable:
                changes.update(treatment=usable[0][0], treatment_kind=usable[0][1])
        if changes:
            self.update(**changes)

    def update(self, **changes: Any) -> None:
        new = validate_spec(replace(self.spec, **changes))
        if new == self.spec:
            return
        keys_changed = self.report is None or (new.unit, new.time) != (
            self.report.unit,
            self.report.time,
        )
        reanalyze = keys_changed or new.adoption_rule != self.spec.adoption_rule
        self.spec, self._panel = new, None
        if reanalyze and self.raw is not None:
            self.analyze()

    def panel(self) -> Panel:
        """The panel for the current Spec (cached; a build error is cached and re-raised)."""
        if self.raw is None:
            raise ValueError("Choose a data file first")
        key = json.dumps(spec_to_json(self.spec), sort_keys=True, default=str)
        if self._panel is None or self._panel[0] != key:
            try:
                value: Panel | Exception = build_panel(self.raw, self.spec, self.data_hash)
            except (ValueError, KeyError, TypeError) as exc:
                value = exc
            self._panel = (key, value)
        value = self._panel[1]
        if isinstance(value, Exception):
            raise value
        return value

    def panel_or_error(self) -> Panel | Exception:
        try:
            return self.panel()
        except (ValueError, KeyError, TypeError) as exc:
            return exc

    def add_wdi(self, codes: Sequence[str]) -> None:
        if self.raw is None or not (self.spec.unit and self.spec.time):
            raise ValueError("Choose a file and set the unit and time columns first")
        self.raw, sources = add_wdi_columns(
            self.raw, self.spec.unit, self.spec.time, codes, self.cfg
        )
        self.wdi_sources += sources
        self.update(wdi=(*self.spec.wdi, *codes))
        self._panel = None
        self.analyze()

    def run(self, keys: Sequence[str], label: str | None = None) -> Run:
        """Run stages in a new output folder and return the finished Run."""
        keys = with_prerequisites(keys)
        with Run(self.cfg, self.spec, label or "-".join(keys) or "run") as run:
            panel = run.stage("Build panel", self.panel)
            for key in keys:
                stage = STAGE_BY_KEY[key]
                if panel is None:
                    run.skip(stage.title, "the panel could not be built", "upstream_failed")
                    continue
                reason = availability(stage, self.spec, panel, self.state, planned=keys)
                if reason is not None:
                    kind = "missing_package" if reason.startswith("install") else "not_applicable"
                    run.skip(stage.title, reason, kind)
                    continue
                ctx = Context(run, panel, self.spec, self.cfg, self.state)
                run.stage(stage.title, lambda stage=stage, ctx=ctx: stage.fn(ctx))
            run.finish(provenance(self))
        return run

    def settings_json(self) -> dict[str, Any]:
        return spec_to_json(self.spec)


# --- Interactive menu --------------------------------------------------------------------------


class Console:
    """Prompt helpers over injectable input and output functions (scriptable in tests)."""

    def __init__(
        self, read: Callable[[str], str] = input, write: Callable[[str], None] = print
    ) -> None:
        self.read, self.write = read, write

    def ask(self, prompt: str, default: str | None = None) -> str:
        shown = f" [{default}]" if default not in (None, "") else ""
        answer = self.read(f"{prompt}{shown}: ").strip()
        return answer or (default or "")

    def ask_bool(self, prompt: str, default: bool) -> bool:
        while True:
            answer = self.ask(f"{prompt} (y/n)", "y" if default else "n").lower()
            if answer in ("y", "yes", "n", "no"):
                return answer.startswith("y")
            self.write("Please answer y or n.")

    def ask_choice(self, prompt: str, choices: Sequence[str], default: str) -> str:
        while True:
            answer = self.ask(f"{prompt} ({'/'.join(choices)})", default)
            if answer in choices:
                return answer
            self.write(f"Choose one of: {', '.join(choices)}")

    def ask_int(self, prompt: str, default: int | None, minimum: int | None = None) -> int | None:
        while True:
            answer = self.ask(prompt, "" if default is None else str(default))
            if answer == "" or answer.lower() == "auto":
                return None
            try:
                value = int(answer)
            except ValueError:
                self.write("Please enter a whole number (or 'auto').")
                continue
            if minimum is not None and value < minimum:
                self.write(f"Must be at least {minimum}.")
                continue
            return value

    def ask_columns(
        self, prompt: str, columns: Sequence[str], current: Sequence[str], multi: bool = True
    ) -> tuple[str, ...]:
        """Columns by number, range (3-7) or name; Enter keeps the current choice, '-' clears."""
        while True:
            answer = self.ask(prompt, ", ".join(current) if current else "")
            if answer == "-":
                return ()
            if answer == ", ".join(current):
                return tuple(current)
            chosen, problems = [], []
            for token in [t.strip() for t in answer.split(",") if t.strip()]:
                found = self._resolve(token, columns)
                if found is None:
                    problems.append(token)
                else:
                    chosen += found
            chosen = list(dict.fromkeys(chosen))
            if problems:
                self.write(f"Not recognised: {', '.join(problems)}")
            elif not multi and len(chosen) > 1:
                self.write("Choose a single column.")
            else:
                return tuple(chosen)

    @staticmethod
    def _resolve(token: str, columns: Sequence[str]) -> list[str] | None:
        if re.fullmatch(r"\d+", token) and 1 <= int(token) <= len(columns):
            return [columns[int(token) - 1]]
        if match := re.fullmatch(r"(\d+)\s*-\s*(\d+)", token):
            lo, hi = int(match[1]), int(match[2])
            if 1 <= lo <= hi <= len(columns):
                return list(columns[lo - 1 : hi])
        if token in columns:
            return [token]
        folded = [c for c in columns if c.lower() == token.lower()]
        return folded if len(folded) == 1 else None


class Menu:
    """Numbered terminal menus over a Session."""

    def __init__(self, session: Session, console: Console | None = None) -> None:
        self.session, self.console = session, console or Console()

    @property
    def spec(self) -> Spec:
        return self.session.spec

    def say(self, text: str = "") -> None:
        self.console.write(text)

    def loop(self) -> int:
        actions = {
            "1": self.choose_file,
            "2": self.show_analysis,
            "3": self.roles_menu,
            "4": self.options_menu,
            "5": lambda: self.run_keys(["descriptives"]),
            "6": lambda: self.run_keys(["ols"]),
            "7": lambda: self.run_keys(["matching"]),
            "8": self.did_menu,
            "9": self.run_several,
            "w": self.add_world_bank,
            "s": self.save_settings,
        }
        while True:
            self.header()
            try:
                choice = self.console.ask("Choice").lower()
            except (EOFError, KeyboardInterrupt):
                self.say("")
                return 0
            if choice in ("q", "quit", "exit"):
                return 0
            action = actions.get(choice)
            if action is None:
                self.say("Unknown choice.")
                continue
            try:
                action()
            except (EOFError, KeyboardInterrupt):
                self.say("\n(cancelled)")
            except Exception as exc:  # report the problem and keep the menu alive
                self.say(f"Error: {type(exc).__name__}: {exc}")

    def header(self) -> None:
        s, session = self.spec, self.session
        self.say("\n" + "=" * 72)
        self.say("Panel DiD toolkit")
        if session.raw is None:
            self.say("File      : (none) - choose a data file first")
        else:
            sheet = f" [sheet {s.sheet}]" if s.sheet else ""
            self.say(
                f"File      : {display_path(s.file)}{sheet} "
                f"({len(session.raw)} rows, {session.raw.shape[1]} columns)"
            )
            panel = session.panel_or_error()
            status = panel.describe() if isinstance(panel, Panel) else f"not ready - {panel}"
            self.say(f"Panel     : {status}")
            outcome = f"log({s.outcome})" if s.outcome and s.log_outcome else s.outcome
            treatment = f"{s.treatment} ({s.treatment_kind or 'auto'})" if s.treatment else "-"
            self.say(f"Outcome   : {outcome or '-'}   Treatment: {treatment}")
            self.say(
                f"Controls  : {short_list(s.controls, 6) or '-'}   "
                f"Covariates: {short_list(s.covariates, 4) or '-'}"
            )
        self.say("-" * 72)
        panel = session.panel_or_error() if session.raw is not None else ValueError("no file")
        lines = [
            ("1", "Choose data file"),
            ("2", "Show file analysis"),
            ("3", "Set variable roles"),
            ("4", "Options"),
        ]
        for number, key in (("5", "descriptives"), ("6", "ols"), ("7", "matching")):
            lines.append((number, self._stage_line(STAGE_BY_KEY[key], panel)))
        lines += [
            ("8", "Staggered DiD estimators ..."),
            ("9", "Run several stages ..."),
            ("w", "Add World Bank indicators (country panels)"),
            ("s", "Save settings"),
            ("q", "Quit"),
        ]
        for number, text in lines:
            self.say(f" {number:>2}  {text}")

    def _stage_line(
        self, stage: StageDef, panel: Panel | Exception, planned: Sequence[str] = ()
    ) -> str:
        reason = availability(stage, self.spec, panel, self.session.state, planned)
        return stage.title if reason is None else f"{stage.title}  (unavailable: {reason})"

    @staticmethod
    def _pick(answer: str, options: Sequence[Any]) -> Any | None:
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        return None

    def choose_file(self) -> None:
        data_dir = PROJECT_ROOT / "data"
        found = (
            [
                p
                for p in data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in DATA_SUFFIXES and "cache" not in p.parts
            ]
            if data_dir.is_dir()
            else []
        )
        candidates = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        for i, path in enumerate(candidates, 1):
            self.say(f" {i:>2}  {display_path(path)}")
        answer = self.console.ask("File number or path")
        if not answer:
            return
        path = self._pick(answer, candidates) or Path(answer).expanduser()
        sheet = None
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            sheets = list_sheets(path)
            for i, name in enumerate(sheets, 1):
                self.say(f" {i:>2}  {name}")
            pick = self.console.ask("Sheet", "1")
            sheet = self._pick(pick, sheets) or pick
        current = Path(self.spec.file).expanduser().resolve() if self.spec.file else None
        same = current == path.resolve() and sheet == self.spec.sheet
        report = self.session.load(path, sheet, reset_roles=not same)
        self.say("")
        self.say(report.render())
        self.say(
            "\nUnit, time, treatment and column families were suggested from the analysis; "
            "choose the outcome under 3 (Set variable roles)."
        )

    def show_analysis(self) -> None:
        self.say(self.session.analyze().render())
        panel = self.session.panel_or_error()
        if isinstance(panel, Panel) and panel.notes:
            self.say("Panel notes:\n" + "\n".join(f"  - {n}" for n in panel.notes))

    def _columns(self) -> list[str]:
        raw = self.session.raw
        if raw is None:
            raise ValueError("Choose a data file first")
        return [*raw.columns, *[f for f in self.spec.combine if f not in raw.columns]]

    def _show_columns(self, columns: Sequence[str]) -> None:
        report = self.session.report
        for i, col in enumerate(columns, 1):
            tags = report.tags(col) if report and col in report.kinds else "combined family"
            self.say(f" {i:>3}  {col}  [{tags}]")

    def roles_menu(self) -> None:
        while True:
            s = self.spec
            self.say("\nVariable roles")
            items = [
                ("1", "Unit", s.unit),
                ("2", "Time", s.time),
                ("3", "Combine column families", short_list(s.combine) or "-"),
                ("4", "Outcome", f"{s.outcome or '-'}{' (log)' if s.log_outcome else ''}"),
                ("5", "Treatment", f"{s.treatment or '-'} ({s.treatment_kind or 'auto'})"),
                ("6", "OLS regressors", short_list(s.regressors) or "(the treatment)"),
                ("7", "Controls", short_list(s.controls) or "-"),
                ("8", "Matching covariates", short_list(s.covariates) or "-"),
            ]
            for number, name, value in items:
                self.say(f" {number}  {name:<26} {value}")
            choice = self.console.ask("Edit which (b = back)", "b").lower()
            if choice == "b":
                return
            columns = self._columns()
            if choice in ("1", "2", "4", "5", "6", "7", "8"):
                self._show_columns(columns)
            if choice == "1":
                self.session.update(unit=self._one("Unit column", columns, s.unit))
            elif choice == "2":
                self.session.update(time=self._one("Time column", columns, s.time))
            elif choice == "3":
                families = self.session.report.families if self.session.report else {}
                for f in families.values():
                    self.say(f"  {f.describe()}")
                if not families:
                    self.say("  No column families detected.")
                chosen = self.console.ask_columns("Families to combine", list(families), s.combine)
                self.session.update(combine=chosen)
            elif choice == "4":
                outcome = self._one("Outcome", columns, s.outcome)
                self.session.update(
                    outcome=outcome,
                    log_outcome=self.console.ask_bool("Use log(outcome)", s.log_outcome),
                )
            elif choice == "5":
                treatment = self._one(
                    "Treatment column ('-' for none)", columns, s.treatment, allow_none=True
                )
                kind = None
                if treatment:
                    if treatment == s.treatment and s.treatment_kind:
                        guess = s.treatment_kind
                    elif treatment in self.session.raw:
                        guess = infer_treatment_kind(self.session.raw[treatment])
                    else:
                        guess = "dose"
                    kind = self.console.ask_choice("Treatment kind", TREATMENT_KINDS, guess)
                self.session.update(treatment=treatment, treatment_kind=kind)
            elif choice == "6":
                self.session.update(
                    regressors=self.console.ask_columns(
                        "OLS regressors (Enter keeps; '-' = the treatment)", columns, s.regressors
                    )
                )
            elif choice == "7":
                chosen = self.console.ask_columns("Controls ('-' clears)", columns, s.controls)
                self._warn_controls(chosen)
                self.session.update(controls=chosen)
            elif choice == "8":
                chosen = self.console.ask_columns(
                    "Matching covariates ('-' clears)", columns, s.covariates
                )
                if s.outcome in chosen:
                    self.say(
                        "Note: matching on the outcome's own baseline level can bias DiD "
                        "through regression to the mean (Daw and Hatfield 2018)."
                    )
                self.session.update(covariates=chosen)

    def _one(
        self, prompt: str, columns: Sequence[str], current: str | None, allow_none: bool = False
    ) -> str | None:
        chosen = self.console.ask_columns(
            prompt, columns, [current] if current else [], multi=False
        )
        if not chosen and not allow_none:
            raise ValueError(f"{prompt}: a column is required")
        return chosen[0] if chosen else None

    def _warn_controls(self, chosen: Sequence[str]) -> None:
        report, s = self.session.report, self.spec
        if report is None:
            return
        groups = (
            ("absorbed by period fixed effects (kept only in pooled OLS)", report.time_only),
            ("absorbed by unit fixed effects", report.unit_only),
            ("defined only for treated units: post-treatment, a bad control", report.treated_only),
            (
                "treatment intensity (zero before adoption): a bad control",
                [c for c, k, _ in report.treatment_candidates if k == "dose"],
            ),
            ("the outcome or the treatment itself", [s.outcome, s.treatment]),
        )
        for text, cols in groups:
            hit = [c for c in chosen if c in cols]
            if hit:
                self.say(f"Note: {short_list(hit)} - {text}.")
        for col, parts in report.dependencies:
            if col in chosen and set(parts) & set(chosen):
                self.say(f"Note: {col} is (almost) a linear combination of {short_list(parts)}.")

    def options_menu(self) -> None:
        while True:
            s, cfg = self.spec, self.session.cfg
            panel = self.session.panel_or_error()
            window = s.event_window or (panel.window if isinstance(panel, Panel) else "auto")
            self.say("\nOptions")
            items = [
                ("1", "Event window (periods)", window),
                (
                    "2",
                    "Matching baseline",
                    " to ".join(s.baseline) if s.baseline else "auto (year before first adoption)",
                ),
                ("3", "Min. baseline observations", s.min_baseline_obs or "every baseline period"),
                ("4", "Caliper (SD of linear score)", s.caliper_sd),
                ("5", "dCDH effects / placebos", f"{s.dcdh.effects} / {s.dcdh.placebo}"),
                ("6", "Callaway-Sant'Anna controls", s.control_group),
                ("7", "Sample period", " to ".join(s.sample) if s.sample else "all"),
                (
                    "8",
                    "Frequency",
                    s.frequency
                    + (f" (summed: {short_list(s.sum_columns)})" if s.sum_columns else ""),
                ),
                ("9", "Policy date -> first treated period", s.adoption_rule),
                ("10", "Left-censored units", s.left_censored),
                (
                    "11",
                    "Units without treatment records",
                    "never treated" if s.missing_treatment_zero else "error",
                ),
                ("12", "Spreadsheet-safe CSV copies", "on" if cfg.spreadsheet_safe else "off"),
                ("13", "Profiling", "on" if cfg.profile else "off"),
                (
                    "14",
                    "HonestDiD M-bar values / grid points",
                    f"{', '.join(map(str, s.sensitivity_mbar))} / {s.sensitivity_grid}",
                ),
            ]
            for number, name, value in items:
                self.say(f" {number:>2}  {name:<38} {value}")
            choice = self.console.ask("Change which (b = back)", "b").lower()
            if choice == "b":
                return
            self._change_option(choice)

    def _ask_bounds(self, prompt: str, current: tuple[str, str] | None) -> tuple[str, str] | None:
        answer = self.console.ask(
            f"{prompt} as 'start end' (e.g. 2015Q1 2016Q4), or 'all'",
            " ".join(current) if current else "all",
        )
        if answer.lower() in ("all", "auto", ""):
            return None
        parts = answer.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("enter two period labels")
        return parts[0], parts[1]

    def _change_option(self, choice: str) -> None:
        s, session, ask = self.spec, self.session, self.console
        if choice == "1":
            answer = ask.ask(
                "Event window 'first last' (e.g. -8 12), or 'auto'",
                " ".join(map(str, s.event_window)) if s.event_window else "auto",
            )
            window = (
                None
                if answer.lower() == "auto"
                else tuple(int(x) for x in answer.replace(",", " ").split())
            )
            session.update(event_window=window)
        elif choice == "2":
            session.update(baseline=self._ask_bounds("Baseline", s.baseline))
        elif choice == "3":
            session.update(
                min_baseline_obs=ask.ask_int(
                    "Minimum observations ('auto' = every period)", s.min_baseline_obs, minimum=1
                )
            )
        elif choice == "4":
            session.update(caliper_sd=float(ask.ask("Caliper in SDs", str(s.caliper_sd))))
        elif choice == "5":
            effects = ask.ask_int("dCDH effects", s.dcdh.effects, minimum=1) or s.dcdh.effects
            placebo = ask.ask_int("dCDH placebos", s.dcdh.placebo, minimum=0)
            session.update(
                dcdh=replace(
                    s.dcdh, effects=effects, placebo=s.dcdh.placebo if placebo is None else placebo
                )
            )
        elif choice == "6":
            session.update(
                control_group=ask.ask_choice(
                    "Control group", SPEC_CHOICES["control_group"], s.control_group
                )
            )
        elif choice == "7":
            session.update(sample=self._ask_bounds("Sample period", s.sample))
        elif choice == "8":
            frequency = ask.ask_choice("Frequency", SPEC_CHOICES["frequency"], s.frequency)
            sums: tuple[str, ...] = ()
            if frequency == "annual":
                candidates = [
                    c for c in [s.outcome, *s.regressors, *s.controls, *s.covariates] if c
                ]
                self.say(
                    "Flows such as starts or permits should be summed; levels and rates averaged."
                )
                self._show_columns(candidates)
                sums = ask.ask_columns("Columns to sum ('-' = none)", candidates, s.sum_columns)
            session.update(frequency=frequency, sum_columns=sums)
        elif choice == "9":
            session.update(
                adoption_rule=ask.ask_choice(
                    "First treated period", SPEC_CHOICES["adoption_rule"], s.adoption_rule
                )
            )
        elif choice == "10":
            session.update(
                left_censored=ask.ask_choice(
                    "Left-censored units", SPEC_CHOICES["left_censored"], s.left_censored
                )
            )
        elif choice == "11":
            session.update(
                missing_treatment_zero=ask.ask_bool(
                    "Treat units with no treatment records at all as never treated",
                    s.missing_treatment_zero,
                )
            )
        elif choice == "12":
            session.cfg = replace(session.cfg, spreadsheet_safe=not session.cfg.spreadsheet_safe)
        elif choice == "13":
            session.cfg = replace(session.cfg, profile=not session.cfg.profile)
        elif choice == "14":
            values = ask.ask(
                "M-bar values, comma-separated", ", ".join(map(str, s.sensitivity_mbar))
            )
            grid = ask.ask_int(
                "Grid points (fewer is faster, coarser)", s.sensitivity_grid, minimum=10
            )
            session.update(
                sensitivity_mbar=tuple(float(v) for v in values.split(",")),
                sensitivity_grid=grid or s.sensitivity_grid,
            )
        else:
            self.say("Unknown option.")

    def did_menu(self) -> None:
        stages = [s for s in STAGES if s.did]
        self._pick_and_run(stages, "Staggered DiD estimators")

    def run_several(self) -> None:
        self._pick_and_run(list(STAGES), "All stages")

    def _pick_and_run(self, stages: Sequence[StageDef], title: str) -> None:
        panel = (
            self.session.panel_or_error() if self.session.raw is not None else ValueError("no file")
        )
        self.say(f"\n{title}")
        for i, stage in enumerate(stages, 1):
            self.say(f" {i:>2}  {self._stage_line(stage, panel)}")
        answer = self.console.ask(
            "Numbers to run (e.g. 1,3), 'a' = all available, b = back", "b"
        ).lower()
        if answer == "b":
            return
        if answer == "a":
            chosen = [
                s.key
                for s in stages
                if availability(s, self.spec, panel, self.session.state, [x.key for x in stages])
                is None
            ]
        else:
            numbers = [int(x) for x in re.findall(r"\d+", answer)]
            chosen = [stages[n - 1].key for n in numbers if 1 <= n <= len(stages)]
        if chosen:
            self.run_keys(chosen)

    def run_keys(self, keys: Sequence[str]) -> None:
        if self.session.raw is None:
            raise ValueError("Choose a data file first")
        keys = with_prerequisites(keys)
        self.say(f"\nRunning: {', '.join(STAGE_BY_KEY[k].title for k in keys)}")
        if "honestdid" in keys:
            self.say(
                f"HonestDiD can take about a minute per M-bar value "
                f"({len(self.spec.sensitivity_mbar)} set; change under Options > 14)."
            )
        run = self.session.run(keys)
        for title, kind, item in run.sections:
            if kind == "figure":
                continue
            text = table_text(item) if kind == "table" else str(item)
            lines = text.splitlines()
            more = (
                f"\n  ... ({len(lines) - CONSOLE_TABLE_LINES} more lines in the report)"
                if len(lines) > CONSOLE_TABLE_LINES
                else ""
            )
            self.say(f"\n== {title} ==\n" + "\n".join(lines[:CONSOLE_TABLE_LINES]) + more)
        problems = run.issues
        if problems:
            self.say("\nIssues:")
            for x in problems:
                self.say(f"  {x['level']:8} {x['stage']}: {x['message'][:300]}")
        status = "INCOMPLETE" if run.incomplete else "COMPLETE"
        self.say(f"\n{status}: {display_path(run.path)} (report.pdf, tables/, figures/)")

    def add_world_bank(self) -> None:
        codes = self.console.ask("WDI indicator codes, comma-separated (e.g. NY.GDP.PCAP.KD)")
        codes = [c.strip() for c in codes.split(",") if c.strip()]
        if codes:
            self.session.add_wdi(codes)
            self.say(f"Added {', '.join(codes)}.")

    def save_settings(self) -> None:
        target = Path(self.console.ask("Save settings to", "settings.json")).expanduser()
        write_json(target, self.session.settings_json())
        self.say(f"Saved. Replay with: python {Path(__file__).name} --settings {target} --run all")


# --- Demo data -------------------------------------------------------------------------------


def demo_panel(seed: int = 0) -> pd.DataFrame:
    """Synthetic quarterly provincial housing panel with staggered policy dates.

    It has the features the analysis looks for: national series, a province family with an
    Atlantic aggregate, an annual series, a mean score that duplicates its components, a
    post-treatment category and a late-entering unit. The policy lowers log starts.
    """
    rng = np.random.default_rng(seed)
    policies = {
        "BC": ("2016-08-02", "tax"),
        "ON": ("2017-04-21", "tax"),
        "NS": ("2018-03-15", "zoning"),
        "QC": ("2019-11-10", "supply"),
        "NB": ("2020-01-15", "supply"),
        "MB": ("2020-06-01", "zoning"),
    }
    quarters = pd.period_range("2012Q1", "2021Q4", freq="Q")
    n = len(quarters)
    national = {
        "BD.CDN.10YR.DQ.YLD": 2.0 + rng.normal(0, 0.15, n).cumsum(),
        "V39079": np.clip(1 + rng.normal(0, 0.2, n).cumsum(), 0.25, 5),
        "M.BCPI": 500 + rng.normal(0, 15, n).cumsum(),
    }
    regional = {
        f"FVI_CSCE_{r}": rng.normal(0, 1, n).cumsum()
        for r in ("AB", "ATL", "BC", "MB", "ON", "QC", "SK")
    }
    rows = []
    for province in ("AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"):
        size, area = rng.uniform(0.15, 14) * 1e6, rng.uniform(5e3, 1.5e6)
        date, kind = policies.get(province, (None, None))
        adopt = pd.Period(date, freq="Q") if date else None
        for t, quarter in enumerate(quarters):
            treated = adopt is not None and quarter >= adopt
            effect = -0.08 - 0.01 * min((quarter - adopt).n, 6) if treated else 0.0
            scores = rng.uniform(2, 9, 3).round(1) if treated else np.zeros(3)
            population = size * 1.004**t
            level = np.log(size) - 6 + 0.01 * t + 0.1 * np.sin(quarter.quarter)
            rows.append(
                {
                    "province": province,
                    "quarter": str(quarter),
                    "policy_date": date,
                    "policy_type": kind,
                    "score_supply": scores[0],
                    "score_demand": scores[1],
                    "score_tenant": scores[2],
                    "score_mean": round(float(scores.mean()), 4),
                    "starts": float(np.exp(level + effect + rng.normal(0, 0.05))),
                    "permits_annual": float(np.exp(level + 1.3 + rng.normal(0, 0.05)))
                    if quarter.quarter == 4
                    else np.nan,
                    "population": round(population),
                    "pop_growth": round(0.4 + rng.normal(0, 0.05), 4),
                    "pop_density": round(population / area, 3),
                    "unemployment": round(6 + rng.normal(0, 1), 2),
                    **{k: v[t] for k, v in national.items()},
                    **{k: v[t] for k, v in regional.items()},
                }
            )
    data = pd.DataFrame(rows)
    late = (data["province"] == "PE") & (data["quarter"] < "2013Q1")
    return data[~late].reset_index(drop=True)


DEMO_FILE = PROJECT_ROOT / "data" / "demo" / "demo_panel.csv"


def write_demo_file() -> Path:
    atomic_write(DEMO_FILE, lambda p: demo_panel().to_csv(p, index=False))
    return DEMO_FILE


# --- Self-test -------------------------------------------------------------------------------


def self_test() -> int:
    """Offline regression tests (synthetic data; optional estimators skipped if absent)."""
    import unittest
    from unittest.mock import patch

    def demo_spec(**changes: Any) -> Spec:
        base = Spec(
            unit="province",
            time="quarter",
            outcome="starts",
            log_outcome=True,
            treatment="policy_date",
            treatment_kind="date",
            combine=("FVI_CSCE",),
        )
        return replace(base, **changes)

    def quarterly(units: dict[str, list[Any]], start: str = "2010Q1") -> pd.DataFrame:
        rows = []
        for unit, values in units.items():
            for i, value in enumerate(values):
                rows.append({"u": unit, "q": str(pd.Period(start, freq="Q") + i), "d": value})
        return pd.DataFrame(rows)

    class Tests(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls.demo = demo_panel()
            cls.tmp = tempfile.TemporaryDirectory()
            cls.cfg = Config(
                output_dir=Path(cls.tmp.name) / "output", cache_dir=Path(cls.tmp.name) / "cache"
            )

        @classmethod
        def tearDownClass(cls) -> None:
            cls.tmp.cleanup()

        # Time parsing and analysis
        def test_parse_periods(self) -> None:
            cases = {
                "Q": (["2015Q1", "Q2 2015", "2015-Q3"]),
                "M": (["2015-01", "2015M02"]),
                "Y": ([2014, 2015]),
            }
            for freq, values in cases.items():
                _, found = parse_periods(pd.Series(values))
                self.assertEqual(found, freq)
            dates = pd.Series(pd.to_datetime(["2015-03-31", "2015-06-30", "2015-12-31"]))
            self.assertEqual(parse_periods(dates)[1], "Q")
            with self.assertRaises(ValueError):
                parse_periods(pd.Series(pd.to_datetime(["2015-01-01", "2015-01-02"])))

        def test_analysis_finds_structure(self) -> None:
            report = analyze_dataset(self.demo, "demo")
            self.assertEqual((report.unit, report.time, report.freq), ("province", "quarter", "Q"))
            self.assertIn("M.BCPI", report.time_only)
            self.assertIn("permits_annual", report.low_frequency)
            self.assertIn("policy_type", report.treated_only)
            self.assertIn("score_mean", [c for c, _ in report.dependencies])
            family = report.families["FVI_CSCE"]
            self.assertEqual(family.sources["NS"], "FVI_CSCE_ATL")
            self.assertEqual(family.sources["AB"], "FVI_CSCE_AB")
            self.assertEqual(report.treatment_candidates[0][:2], ("policy_date", "date"))

        def test_family_values(self) -> None:
            keys, _ = panel_keys(self.demo, "province", "quarter")
            family = find_families(self.demo, keys)["FVI_CSCE"]
            combined = combine_family(self.demo, keys, family)
            row = self.demo.index[self.demo["province"] == "NL"][0]
            self.assertEqual(combined[row], self.demo.loc[row, "FVI_CSCE_ATL"])

        # Review fix 1: treatment timing
        def test_adoption_ignores_outcome_coverage(self) -> None:
            data = self.demo.copy()
            bc = data["province"] == "BC"
            data.loc[bc & data["quarter"].between("2015Q4", "2017Q2"), "starts"] = np.nan
            data = data[~(bc & data["quarter"].between("2016Q2", "2016Q4"))]  # rows absent too
            panel = build_panel(data, demo_spec())
            first = panel.data.loc[panel.data["_unit"] == "BC", "_first"].iloc[0]
            self.assertEqual(str(panel.period_at(first)), "2016Q3")

        def test_missing_zero_only_fills_absent_units(self) -> None:
            data = quarterly({"A": [np.nan, np.nan, 1, 1], "B": [np.nan] * 4, "C": [0, 0, 1, 1]})
            keys, freq = panel_keys(data, "u", "q")
            timing = treatment_timing(keys, data["d"], "indicator", freq, missing_zero=True)
            self.assertIn("A", timing.left_censored)  # not a made-up adoption in period 3
            self.assertEqual(timing.absent, ["B"])
            self.assertTrue((timing.d[data["u"] == "B"] == 0).all())
            self.assertTrue(timing.d[data["u"] == "A"].iloc[:2].isna().all())

        def test_ambiguous_and_filled_indicator(self) -> None:
            data = quarterly({"A": [0, np.nan, 1, 1], "B": [0, np.nan, 0, 1, np.nan, 1]})
            keys, freq = panel_keys(data, "u", "q")
            timing = treatment_timing(keys, data["d"], "indicator", freq)
            self.assertEqual(timing.ambiguous, ["A"])
            self.assertEqual(timing.d[data["u"] == "B"].tolist(), [0, 0, 0, 1, 1, 1])
            data["y"] = 1.0
            with self.assertRaises(ValueError):
                build_panel(
                    data,
                    Spec(
                        unit="u", time="q", outcome="y", treatment="d", treatment_kind="indicator"
                    ),
                )

        def test_left_censoring_policy(self) -> None:
            data = self.demo.copy()
            data.loc[data["province"] == "AB", "policy_date"] = "2011-05-01"
            with self.assertRaises(ValueError):
                build_panel(data, demo_spec())
            self.assertNotIn(
                "AB", build_panel(data, demo_spec(left_censored="drop")).data["_unit"].unique()
            )
            self.assertFalse(build_panel(data, demo_spec(left_censored="keep")).adoption_ok)

        def test_rank_deficiency_is_an_error(self) -> None:
            d = build_panel(self.demo, demo_spec()).data.dropna(subset=["_y"])
            codes = {u: float(i) for i, u in enumerate(sorted(d["_unit"].unique()))}
            d = d.assign(size=d["_unit"].map(codes))  # time-invariant: absorbed by unit FE
            x, structural = design_matrix(d, ["size"], budget=10**9, unit_fe=True, period_fe=True)
            with self.assertRaisesRegex(ValueError, "size"):
                fit_ols(d["_y"], x, d["_unit"], structural, "test")

        def test_stage_status_is_explicit(self) -> None:
            with Run(self.cfg, Spec()) as run:
                run.stage("ok", lambda: run.issue("NOTE", "ok", "sub-step skipped"))

                def skipped() -> None:
                    raise StageSkipped("not applicable")

                run.stage("skip", skipped)
                run.stage("fail", lambda: 1 / 0)
            self.assertEqual([s["status"] for s in run.stages], ["success", "skipped", "failed"])
            self.assertTrue(run.incomplete)

        def test_report_with_dollar_text(self) -> None:
            with Run(self.cfg, Spec()) as run:
                run.add("text", "bad value $x^$ and $5")
                run.add("figure", event_figure([0, 1], [1, 2], [0, 1], [2, 3], "t", "x"))
                run.finish({"test": True})
            self.assertEqual(run.stages[-1]["status"], "success")
            self.assertTrue((run.path / "report.pdf").read_bytes().startswith(b"%PDF"))

        def test_baseline_observation_check(self) -> None:
            spec = demo_spec(
                covariates=("population",), baseline=("2015Q1", "2015Q4"), min_baseline_obs=5
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                baseline_means(build_panel(self.demo, spec), spec, ["population"])

        def test_sensitivity_needs_consecutive_event_times(self) -> None:
            with self.assertRaises(ValueError):
                sensitivity_layout(np.array([-4, -2, 0, 1]), np.ones(4), np.eye(4))
            beta, _, pre, post, target = sensitivity_layout(
                np.array([1, -2, 0, -3]), np.arange(4.0), np.eye(4)
            )
            self.assertEqual((pre, post), (2, 2))
            np.testing.assert_array_equal(beta, [3, 1, 2, 0])
            np.testing.assert_array_equal(target, [1, 0])

        # Review fix 2: resources
        def test_run_releases_resources(self) -> None:
            loggers = len(logging.Logger.manager.loggerDict)
            state = np.random.get_state()[1].copy()
            tracemalloc.start()
            try:
                with self.assertRaises(RuntimeError), Run(replace(self.cfg, profile=True), Spec()):
                    raise RuntimeError("boom")
                self.assertTrue(tracemalloc.is_tracing())  # the caller's tracing survives
            finally:
                tracemalloc.stop()
            self.assertEqual(LOG.handlers, [])
            self.assertEqual(len(logging.Logger.manager.loggerDict), loggers)
            np.testing.assert_array_equal(np.random.get_state()[1], state)

        def test_warnings_are_counted(self) -> None:
            def noisy() -> None:
                for _ in range(500):
                    warnings.warn("repeat", RuntimeWarning, stacklevel=1)

            with Run(self.cfg, Spec()) as run:
                run.stage("noisy", noisy)
            self.assertEqual([x["message"] for x in run.issues], ["RuntimeWarning: repeat (x500)"])

        # Review fix 4: security and privacy
        def test_atomic_write(self) -> None:
            path = Path(self.tmp.name) / "a.txt"
            path.write_text("old")

            def fail(p: Path) -> None:
                p.write_text("partial")
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(path.read_text(), "old")
            atomic_write(path, lambda p: p.write_text("new"))
            self.assertEqual(path.stat().st_mode & 0o777, FILE_MODE)

        def test_column_names_are_not_code(self) -> None:
            name = "x + I(print('CODE RAN') or x)"
            data = self.demo.assign(**{name: self.demo["unemployment"]})
            panel = build_panel(data, demo_spec(controls=(name,)))
            buffer = io.StringIO()
            with (
                Run(self.cfg, demo_spec(controls=(name,))) as run,
                contextlib.redirect_stdout(buffer),
            ):
                stage_ols(
                    Context(run, panel, demo_spec(controls=(name,)), self.cfg, SessionState())
                )
            self.assertNotIn("CODE RAN", buffer.getvalue())

        def test_private_paths(self) -> None:
            self.assertEqual(display_path("/home/someone/secret/panel.csv"), "panel.csv")
            with patch("shutil.which", return_value=None):
                self.assertIsNone(git_info()["commit"])

        # Settings, matching helpers, collapse
        def test_matching_flags_outcome_copies(self) -> None:
            data = self.demo.assign(income=2 * np.log(self.demo["starts"]))  # renamed, rescaled
            spec = demo_spec(covariates=("income", "pop_density"), caliper_sd=2.0)
            with Run(self.cfg, spec) as run:
                ctx = Context(run, build_panel(data, spec), spec, self.cfg, SessionState())
                run.stage("matching", lambda: stage_matching(ctx))
            self.assertTrue(any("copy of) the outcome" in x["message"] for x in run.issues))

        def test_settings_roundtrip(self) -> None:
            spec = demo_spec(controls=("unemployment",), event_window=(-6, 8))
            self.assertEqual(spec_from_json(json.loads(json.dumps(spec_to_json(spec)))), spec)
            for bad in ({"outcom": "y"}, {"event_window": [-8]}, {"caliper_sd": -1}):
                with self.assertRaises(ValueError):
                    spec_from_json(bad)

        def test_collapse_annual(self) -> None:
            data = quarterly({"A": [1, 2, 3, 4, 5, 6, 7]})
            data["flow"], data["rate"] = data["d"], data["d"] * 10.0
            spec = Spec(
                unit="u",
                time="q",
                outcome="flow",
                frequency="annual",
                sum_columns=("flow",),
                controls=("rate",),
            )
            panel = build_panel(data, spec)
            self.assertEqual(panel.data["_y"].tolist()[0], 10)
            self.assertTrue(np.isnan(panel.data["_y"].tolist()[1]))  # incomplete year
            self.assertEqual(panel.data["rate"].tolist(), [25.0, 60.0])

        def test_optimal_pairs(self) -> None:
            self.assertEqual(
                len(optimal_pairs(np.array([0.0]), np.array([10.0]), 0.1, 10000)[0]), 0
            )
            i, _, dist = optimal_pairs(np.array([0.0, 0.09]), np.array([0.08, 0.18]), 0.1, 10000)
            self.assertEqual(len(i), 2)
            self.assertTrue((dist <= 0.1).all())
            with self.assertRaises(MemoryError):
                optimal_pairs(np.zeros(10), np.zeros(10), 0.1, 1)

        def test_spreadsheet_text(self) -> None:
            self.assertEqual(spreadsheet_text(" =SUM(A1)"), "' =SUM(A1)")
            self.assertEqual(spreadsheet_text(-2.5), -2.5)

        # End to end
        def test_menu_session(self) -> None:
            path = Path(self.tmp.name) / "demo.csv"
            self.demo.to_csv(path, index=False)
            answers = iter(
                [
                    "1",
                    str(path),
                    "3",
                    "4",
                    "starts",
                    "y",
                    "7",
                    "unemployment, pop_density",
                    "b",
                    "6",
                    "q",
                ]
            )
            output: list[str] = []
            session = Session(self.cfg)
            code = Menu(session, Console(lambda _: next(answers), output.append)).loop()
            self.assertEqual(code, 0)
            text = "\n".join(output)
            self.assertIn("COMPLETE:", text)
            self.assertIn("treatment (D)", text)

        def test_all_stages(self) -> None:
            spec = demo_spec(
                controls=("unemployment", "FVI_CSCE"),
                covariates=("pop_density", "unemployment"),
                caliper_sd=2.0,
                event_window=(-4, 4),
                sensitivity_mbar=(1.0,),
                sensitivity_grid=50,
            )
            session = Session(self.cfg, spec)
            session.raw, session.data_hash = self.demo, "demo"
            run = session.run(["all"])
            statuses = {s["name"]: (s["status"], s.get("skip_kind")) for s in run.stages}
            for stage in STAGES:
                status, kind = statuses[stage.title]
                if status != "success":
                    self.assertEqual(kind, "missing_package", f"{stage.title}: {status}")
            self.assertTrue((run.path / "report.pdf").exists())

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    return 0 if result.wasSuccessful() else 1


# --- Command line ------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--self-test", action="store_true", help="run the offline tests")
    parser.add_argument("--demo", action="store_true", help="use a synthetic demo panel")
    parser.add_argument("--settings", type=Path, help="settings JSON saved from the menu")
    parser.add_argument("--file", help="CSV or Excel data file")
    parser.add_argument("--sheet", help="Excel sheet (default: the first)")
    parser.add_argument("--analyze", action="store_true", help="print the file analysis and exit")
    parser.add_argument(
        "--run",
        nargs="+",
        choices=[*STAGE_BY_KEY, "all"],
        metavar="STAGE",
        help=f"run without the menu: {', '.join(STAGE_BY_KEY)} or all",
    )
    roles = parser.add_argument_group("variable roles (override --settings)")
    for name in ("unit", "time", "outcome", "treatment"):
        roles.add_argument(f"--{name}")
    roles.add_argument("--treatment-kind", choices=TREATMENT_KINDS)
    roles.add_argument("--log-outcome", action=argparse.BooleanOptionalAction, default=None)
    for name in ("regressors", "controls", "covariates", "combine"):
        roles.add_argument(f"--{name}", nargs="*")
    env = parser.add_argument_group("environment")
    env.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    env.add_argument("--cache-dir", type=Path, default=cfg.cache_dir)
    env.add_argument("--profile", action="store_true", default=cfg.profile)
    env.add_argument("--spreadsheet-safe", action="store_true", default=cfg.spreadsheet_safe)
    return parser.parse_args(argv)


def spec_from_args(args: argparse.Namespace) -> Spec:
    spec = Spec()
    if args.settings is not None:
        spec = spec_from_json(json.loads(args.settings.read_text(encoding="utf-8")))
    overrides = {
        name: getattr(args, name)
        for name in (
            "file",
            "sheet",
            "unit",
            "time",
            "outcome",
            "treatment",
            "treatment_kind",
            "log_outcome",
        )
        if getattr(args, name) is not None
    }
    for name in ("regressors", "controls", "covariates", "combine"):
        if getattr(args, name) is not None:
            overrides[name] = tuple(getattr(args, name))
    return validate_spec(replace(spec, **overrides))


def resolve_data_path(file: str, settings: Path | None) -> Path:
    """A relative data path is tried from the working directory, the settings file's folder,
    then the project root."""
    path = Path(file).expanduser()
    if path.is_absolute() or path.exists():
        return path
    for base in ([settings.parent] if settings else []) + [PROJECT_ROOT]:
        if (base / path).exists():
            return base / path
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    cfg = Config(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        profile=args.profile,
        spreadsheet_safe=args.spreadsheet_safe,
    )
    try:
        spec = spec_from_args(args)
        if args.demo and spec.file is None:
            spec = replace(spec, file=str(write_demo_file()))
        if (args.analyze or args.run) and spec.file is None:
            raise ValueError("--analyze and --run need --file, --settings or --demo")
        session = Session(cfg, spec)
        if spec.file is not None:
            session.load(resolve_data_path(spec.file, args.settings), spec.sheet)
            if args.demo and session.spec.outcome is None:
                session.update(outcome="starts", log_outcome=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None
    if args.analyze:
        print(session.report.render())
        return 0
    if args.run:
        run = session.run(args.run)
        print(f"{'INCOMPLETE' if run.incomplete else 'COMPLETE'}: {display_path(run.path)}")
        return run.exit_code
    return Menu(session).loop()


if __name__ == "__main__":
    raise SystemExit(main())
