"""Validated panel DiD research pipeline.

Run:
    python project.py --treatment data/raw/treatment.csv
    python project.py --self-test
    python project.py --refresh --profile

Required: numpy pandas scipy statsmodels matplotlib pyarrow wbgapi
Main DiD: py-did-multiplegt-dyn (import: did_multiplegt_dyn)
Optional: pyfixest csdid honestdid

Install a tested environment and freeze its versions before production research.
Optional package adapters validate their outputs but do not promise cross-version API
compatibility. Dose/reversible treatment is supported ONLY by the generalized dCDH
branch; adoption-design matching and event studies require absorbing binary treatment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
import traceback
import tracemalloc
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    years: tuple[int, int] = (2000, 2024)  # End exclusive.
    baseline: tuple[int, int] = (2005, 2007)  # Both endpoints inclusive.
    covariates: tuple[str, ...] = ("pop_growth", "income")
    controls: tuple[str, ...] = ()
    rel_range: tuple[int, int] = (-4, 6)  # Pooled endpoint tails, not single periods.
    seed: int = 20261003
    caliper_sd: float = 0.2
    treatment_mode: str = "binary"
    missing_treatment_zero: bool = False
    allow_left_censored: bool = False
    min_baseline_observations: int = 3
    max_match_bytes: int = 512 * 1024**2
    max_design_bytes: int = 512 * 1024**2
    refresh: bool = False
    profile: bool = False
    spreadsheet_safe: bool = False
    treatment: str = "data/raw/treatment.csv"
    cache_dir: str = "data/cache"
    output_dir: str = "output"
    wdi_max_age_days: int = 30
    dcdh: dict[str, Any] = field(default_factory=lambda: {
        "effects": 5, "placebo": 3, "same_switchers": True, "effects_equal": True,
    })


VARIABLES = (
    {"name": "Y", "source": "wdi", "code": "NY.GDP.PCAP.KD", "transform": "log"},
    {"name": "X", "source": "wdi", "code": "SP.POP.GROW"},
    {"name": "pop_growth", "source": "wdi", "code": "SP.POP.GROW"},
    {"name": "income", "source": "wdi", "code": "NY.GDP.PCAP.KD", "transform": "log"},
)
CACHE_SCHEMA = 2


def atomic_write(path: Path, writer: Callable[[Path], Any]) -> None:
    """Replace a file only after its sibling temporary file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, lambda p: p.write_text(
        json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8"))


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_component(name: str) -> str:
    stem = "".join(c if c.isalnum() else "_" for c in name).strip("_")[:90]
    return f"{stem or 'item'}_{hashlib.sha256(name.encode()).hexdigest()[:10]}"


def spreadsheet_text(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


class Run:
    """All mutable state belongs to a single run, never to module globals."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(cfg.output_dir) / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        for part in ("tables", "figures", "processed"):
            (self.path / part).mkdir()
        self.sections: list[tuple[str, str, Any]] = []
        self.issues: list[dict[str, str]] = []
        self.stages: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []
        self.incomplete = False
        self.log = logging.getLogger(f"project.{stamp}.{uuid.uuid4().hex}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        handler = logging.FileHandler(self.path / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.log.addHandler(handler)

    def issue(self, level: str, stage: str, message: str) -> None:
        self.issues.append({"level": level, "stage": stage, "message": message})
        self.log.log(logging.ERROR if level == "ERROR" else logging.WARNING,
                     "%s: %s", stage, message)

    def skip(self, name: str, reason: str, required: bool = False) -> None:
        self.stages.append({"name": name, "status": "skipped_dependency", "reason": reason})
        self.issue("SKIPPED", name, reason)
        self.incomplete |= required

    def call(self, name: str, fn: Callable[[], Any], required: bool = False) -> Any:
        record: dict[str, Any] = {"name": name, "required": required}
        start = time.perf_counter()
        cpu_start = time.process_time()
        prior_stage_count = len(self.stages)
        import matplotlib.pyplot as plt
        prior_figures = set(plt.get_fignums())
        if self.cfg.profile:
            tracemalloc.reset_peak()
        self.log.info("START %s", name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = fn()
                new_skips = [s for s in self.stages[prior_stage_count:]
                     if s.get("status") == "skipped_dependency"]
                record["status"] = "skipped_dependency" if new_skips and value is None else "success"
                if new_skips and value is not None:
                    record["nested_skips"] = [s["name"] for s in new_skips]
            except Exception:
                value = None
                record["status"] = "failed"
                self.incomplete |= required
                self.issue("ERROR", name, traceback.format_exc())
            finally:
                for warning in caught:
                    message = f"{warning.category.__name__}: {warning.message}"
                    if not any(i["stage"] == name and i["message"] == message for i in self.issues):
                        self.issue("WARNING", name, message)
                record["seconds"] = round(time.perf_counter() - start, 6)
                record["cpu_seconds"] = round(time.process_time() - cpu_start, 6)
                for number in set(plt.get_fignums()) - prior_figures:
                    plt.close(number)
                if self.cfg.profile:
                    record["python_traced_peak_bytes"] = tracemalloc.get_traced_memory()[1]
                    try:
                        import resource
                        record["process_maxrss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    except ImportError:
                        pass
                self.stages.append(record)
        return value

    def add(self, title: str, item: Any) -> None:
        import matplotlib.pyplot as plt
        name = safe_component(title)
        if isinstance(item, pd.DataFrame):
            atomic_write(self.path / "tables" / f"{name}.csv", lambda p: item.to_csv(p))
            if self.cfg.spreadsheet_safe:
                safe = item.copy()
                for col in safe.columns:
                    if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
                        safe[col] = safe[col].map(spreadsheet_text)
                if isinstance(safe.index, pd.MultiIndex):
                    safe.index = pd.MultiIndex.from_tuples(
                        [tuple(spreadsheet_text(x) for x in row) for row in safe.index],
                        names=safe.index.names)
                else:
                    safe.index = safe.index.map(spreadsheet_text)
                if isinstance(safe.columns, pd.MultiIndex):
                    safe.columns = pd.MultiIndex.from_tuples(
                        [tuple(spreadsheet_text(x) for x in row) for row in safe.columns],
                        names=safe.columns.names)
                else:
                    safe.columns = safe.columns.map(spreadsheet_text)
                atomic_write(self.path / "tables" / f"{name}_spreadsheet.csv", lambda p: safe.to_csv(p))
            self.sections.append((title, "text", item.round(4).to_string()))
        elif isinstance(item, plt.Figure):
            path = self.path / "figures" / f"{name}.png"
            try:
                atomic_write(path, lambda p: item.savefig(p, dpi=150, bbox_inches="tight"))
                self.sections.append((title, "image", path))
            finally:
                plt.close(item)
        else:
            self.sections.append((title, "text", str(item)))

    def close(self) -> None:
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)


def optional(name: str, run: Run) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            run.skip(name, f"Optional package {name} is not installed.")
            return None
        raise ImportError(f"{name} is installed but dependency {exc.name} is missing") from exc


def validate_long(data: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"id", "year", "value"}
    if not required.issubset(data.columns):
        raise ValueError(f"{label}: missing columns {sorted(required - set(data.columns))}")
    data = data.loc[:, ["id", "year", "value"]].copy()
    if data["id"].isna().any():
        raise ValueError(f"{label}: missing IDs")
    data["id"] = data["id"].astype("string").str.strip()
    if data["id"].eq("").any():
        raise ValueError(f"{label}: empty IDs")
    year = pd.to_numeric(data["year"], errors="raise")
    if year.isna().any() or not np.isfinite(year).all() or (year % 1 != 0).any():
        raise ValueError(f"{label}: invalid years")
    data["year"] = year.astype("int64")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if np.isinf(data["value"].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError(f"{label}: infinite values")
    if data.duplicated(["id", "year"]).any():
        raise ValueError(f"{label}: duplicate unit-years")
    return data.sort_values(["id", "year"]).reset_index(drop=True)


def cache_identity(source: str, code: str, years: tuple[int, int], source_hash: str | None = None) -> dict:
    return {"schema": CACHE_SCHEMA, "source": source, "code": code,
            "years": list(years), "source_sha256": source_hash}


class Loader:
    def __init__(self, run: Run):
        self.run = run
        self.memo: dict[str, pd.DataFrame] = {}
        self.cache = Path(run.cfg.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def load(self, source: str, code: str) -> pd.DataFrame:
        cfg = self.run.cfg
        source_path = Path(code).resolve() if source == "csv" else None
        source_hash = fingerprint(source_path) if source_path else None
        identity = cache_identity(source, str(source_path) if source_path else code,
                                  cfg.years, source_hash)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in self.memo:
            return self.memo[key].copy()
        path = self.cache / f"{key}.parquet"
        meta_path = self.cache / f"{key}.json"
        cached = False
        meta = None
        if not cfg.refresh and path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(meta["extracted_at"])).total_seconds()
                fresh = source != "wdi" or age <= cfg.wdi_max_age_days * 86400
                if meta["identity"] == identity and fresh and meta["parquet_sha256"] == fingerprint(path):
                    data = validate_long(pd.read_parquet(path), code)
                    cached = True
            except Exception as exc:
                self.run.issue("WARNING", "cache", f"Ignoring invalid cache {key}: {exc}")
        if not cached:
            if source == "csv":
                # Paths are deliberate local CLI/config inputs, not remotely supplied strings.
                data = pd.read_csv(source_path, dtype={"id": "string"})
            elif source == "wdi":
                wb = importlib.import_module("wbgapi")
                # Exclude regional/income aggregates; treatment IDs must use the same economy codes.
                economies = [
                    row["id"] for row in wb.economy.list() if not row.get("aggregate", False)]
                wide = wb.data.DataFrame(code, economy=economies, time=range(*cfg.years),
                                         numericTimeKeys=True, skipBlanks=True)
                data = wide.stack().rename("value").rename_axis(["id", "year"]).reset_index()
            else:
                raise ValueError(f"Unknown source {source}")
            data = validate_long(data, code)
            data = data[data.year.between(cfg.years[0], cfg.years[1] - 1)].copy()
            if data.empty:
                raise ValueError(f"{code}: no data in requested period")
            atomic_write(path, lambda p: data.to_parquet(p, index=False))
            meta = {"identity": identity, "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "parquet_sha256": fingerprint(path)}
            write_json(meta_path, meta)
        self.run.sources.append({**meta, "cache_hit": cached})
        self.memo[key] = data
        return data.copy()


def transform(series: pd.Series, kind: str | None, label: str) -> pd.Series:
    values = series.dropna()
    if kind == "log":
        if (values <= 0).any():
            raise ValueError(f"{label}: log requires positive values")
        return np.log(series)
    if kind == "log1p":
        if (values <= -1).any():
            raise ValueError(f"{label}: log1p requires values greater than -1")
        return np.log1p(series)
    if kind is not None:
        raise ValueError(f"Unknown transformation {kind}")
    return series


def treatment_history(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = panel.sort_values(["id", "year"]).reset_index(drop=True).copy()
    if p["D"].isna().any():
        if not cfg.missing_treatment_zero:
            raise ValueError("Missing treatment: supply coverage or explicitly use --missing-treatment-zero")
        p["D"] = p["D"].fillna(0)
    if not np.isfinite(p["D"]).all() or (p["D"] < 0).any():
        raise ValueError("Treatment must be finite and nonnegative")
    if cfg.treatment_mode == "binary":
        if not p["D"].isin([0, 1]).all():
            raise ValueError("Binary treatment requires 0/1 values")
        if p.groupby("id").D.diff().lt(0).any():
            raise ValueError("Adoption estimators require absorbing treatment: reversal detected")
    first_rows = p.groupby("id", sort=False).head(1)
    censored = first_rows.loc[first_rows["D"].gt(0), "id"].tolist()
    if censored and not cfg.allow_left_censored:
        raise ValueError(f"Already treated at panel entry; true adoption unknown: {censored[:15]}")
    p["treated_group"] = p["D"].gt(0).groupby(p["id"]).transform("any").astype("int8")
    p["first"] = p["id"].map(p.loc[p["D"].gt(0)].groupby("id").year.min())
    p["rel"] = (p.year - p["first"]).fillna(-1).clip(*cfg.rel_range).astype("int16")
    return p


def build_panel(run: Run) -> pd.DataFrame:
    loader = Loader(run)
    cols = []
    for var in VARIABLES:
        raw = loader.load(var["source"], var["code"]).set_index(["id", "year"]).value
        cols.append(transform(raw, var.get("transform"), var["name"]).rename(var["name"]))
    p = pd.concat(cols, axis=1).reset_index()
    treatment = loader.load("csv", run.cfg.treatment).rename(columns={"value": "D"})
    unknown = sorted(set(treatment["id"]) - set(p["id"]))
    if unknown:
        raise ValueError(f"Treatment IDs absent from outcome/covariates: {unknown[:20]}")
    p = p.merge(treatment, on=["id", "year"], how="left", validate="one_to_one")
    missing = int(p["D"].isna().sum())
    if missing and run.cfg.missing_treatment_zero:
        run.issue("WARNING", "treatment", f"Explicit sparse-treatment assumption: {missing} missing records become zero")
    p = treatment_history(p, run.cfg)
    if run.cfg.allow_left_censored:
        run.issue("WARNING", "treatment", "Left-censoring override enabled; adoption estimates may not be identified")
    span = p.groupby("id").year.agg(["min", "max", "count"])
    gaps = span.index[span["count"] != span["max"] - span["min"] + 1]
    if len(gaps):
        run.issue("WARNING", "coverage", f"Year gaps for {len(gaps)} units: {list(gaps)[:20]}")
    run.add("Data coverage", span.describe())
    run.add("Missing shares", p.isna().mean().to_frame("share_missing"))
    atomic_write(run.path / "processed" / "panel.parquet", lambda path: p.to_parquet(path, index=False))
    return p


def complete_sample(p: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["id", *columns]))
    d = p.dropna(subset=cols).reset_index(drop=True).copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(d[col]) and not np.isfinite(d[col]).all():
            raise ValueError(f"Nonfinite estimation variable {col}")
    if d.empty or d["id"].nunique() < 2:
        raise ValueError("Estimation requires observations and at least two clusters")
    return d


def clustered(formula: str, p: pd.DataFrame, required: list[str], cfg: Config):
    import statsmodels.formula.api as smf
    d = complete_sample(p, required)
    estimated_cols = 2 + (d["id"].nunique() if "C(id)" in formula else 0) + (
        d.year.nunique() if "C(year)" in formula else 0) + (20 if "C(rel" in formula else 0)
    # Guard is an estimate for the input design only, not total solver memory.
    if len(d) * estimated_cols * 8 > cfg.max_design_bytes:
        raise MemoryError("Dense design exceeds budget; use an absorbed-FE estimator")
    model = smf.ols(formula, d, missing="raise")
    return model.fit(cov_type="cluster", cov_kwds={"groups": d["id"].to_numpy()})


def descriptives(p: pd.DataFrame, run: Run, label: str) -> None:
    import matplotlib.pyplot as plt
    cols = list(dict.fromkeys(["Y", "X", *run.cfg.covariates]))
    run.add(f"Summary {label}", p[cols].describe().T)
    run.add(f"Correlations {label}", p[cols].corr())
    fig, ax = plt.subplots(figsize=(8, 4))
    p.groupby(["year", "treated_group"]).Y.mean().unstack().plot(ax=ax)
    ax.set_title(f"Mean log outcome: {label}")
    run.add(f"Trends {label}", fig)


def ols_stage(p: pd.DataFrame, run: Run) -> None:
    from statsmodels.stats.diagnostic import het_breuschpagan
    d = complete_sample(p, ["Y", "X", "year"])
    pooled = clustered("Y ~ X", d, ["Y", "X"], run.cfg)
    fe = clustered("Y ~ X + C(id) + C(year)", d, ["Y", "X", "year"], run.cfg)
    influence = pooled.get_influence().cooks_distance[0]
    top = d.assign(cooks=influence).nlargest(min(10, len(d)), "cooks")
    models = {"pooled OLS": pooled, "TWFE": fe}
    remaining = d.drop(top.index)
    if len(remaining) > 3 and remaining["id"].nunique() >= 2:
        models["OLS minus top Cook observations"] = clustered("Y ~ X", remaining, ["Y", "X"], run.cfg)
    else:
        run.skip("Cook exclusion", "Too few observations or clusters after exclusion")
    run.add("OLS coefficients", pd.DataFrame({name: {
        "beta": model.params["X"], "se": model.bse["X"],
        "ci_lo": model.conf_int().loc["X", 0], "ci_hi": model.conf_int().loc["X", 1],
        "N": model.nobs, "R2": model.rsquared,
    } for name, model in models.items()}).T)
    run.add("OLS summary", pooled.summary())
    run.add("Cook distances", top[["id", "year", "cooks"]])
    run.add("BP diagnostic", f"Conventional Breusch-Pagan p: {het_breuschpagan(pooled.resid, pooled.model.exog)[1]:.6g}\n"
            "Diagnostic only: this test is not adjusted for panel dependence; OLS is not a treatment-effect estimate.")


def baseline_sample(p: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    treated = p.loc[p["treated_group"].eq(1)].groupby("id")["first"].first()
    bad = treated[treated <= cfg.baseline[1]]
    if len(bad):
        raise ValueError(f"Baseline not entirely pre-treatment for {list(bad.index)[:20]}; choose a valid baseline")
    b = p[p.year.between(*cfg.baseline)]
    counts = b.groupby("id")[list(cfg.covariates)].count()
    eligible = counts.index[counts.ge(cfg.min_baseline_observations).all(axis=1)]
    result = b[b["id"].isin(eligible)].groupby("id")[[*cfg.covariates, "treated_group"]].mean().dropna()
    if result.empty or result["treated_group"].nunique() != 2:
        raise ValueError("Matching needs both groups with sufficient baseline covariates")
    return result


def optimal_pairs(t: np.ndarray, c: np.ndarray, caliper: float, budget: int):
    """No-replacement matching: maximize valid cardinality, then minimize distance.

    Every assignment has min(nt,nc) edges. Invalid edges get a penalty greater
    than the sum of all possible valid costs, so cardinality has priority.
    Invalid assignments are removed after optimization. This remains dense.
    """
    from scipy.optimize import linear_sum_assignment
    t, c = np.asarray(t, dtype=float), np.asarray(c, dtype=float)
    if not len(t) or not len(c) or not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError("Matching needs nonempty finite scores")
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("Invalid caliper")
    if t.size * c.size * 24 > budget:
        raise MemoryError("Dense matching exceeds conservative allocation budget")
    cost = np.subtract(t[:, None], c[None, :])
    np.abs(cost, out=cost)
    penalty = (min(len(t), len(c)) + 1) * (caliper + 1)
    if not np.isfinite(penalty):
        raise ValueError("Matching penalty overflow")
    cost[cost > caliper] = penalty
    i, j = linear_sum_assignment(cost)
    valid = cost[i, j] < penalty
    return i[valid], j[valid], cost[i[valid], j[valid]]


def matching(p: pd.DataFrame, run: Run) -> list[str]:
    import statsmodels.formula.api as smf
    b = baseline_sample(p, run.cfg)
    fit = smf.logit("treated_group ~ " + " + ".join(run.cfg.covariates), b, missing="raise").fit(disp=0)
    if not fit.mle_retvals.get("converged", False):
        raise ValueError("Propensity logit did not converge")
    # Compute the linear predictor directly: no log(p/(1-p)) saturation.
    b["lps"] = np.asarray(fit.model.exog @ fit.params)
    if not np.isfinite(b["lps"]).all():
        raise ValueError("Nonfinite logit scores")
    t, c = b[b["treated_group"].eq(1)], b[b["treated_group"].eq(0)]
    caliper = run.cfg.caliper_sd * b["lps"].std()
    i, j, distances = optimal_pairs(t["lps"].to_numpy(), c["lps"].to_numpy(), caliper, run.cfg.max_match_bytes)
    if not len(i):
        raise ValueError("No pairs within caliper")
    pairs = pd.DataFrame({"treated_id": t.index[i], "control_id": c.index[j], "distance": distances})
    keep = list(pairs["treated_id"]) + list(pairs["control_id"])
    matched = b.loc[keep]
    rows = []
    for v in run.cfg.covariates:
        denom = np.sqrt((t[v].var() + c[v].var()) / 2)
        if not np.isfinite(denom) or denom <= 0:
            run.issue("WARNING", "balance", f"Undefined reference SD for {v}")
            denom = np.nan
        rows.append({"variable": v,
                     "SMD before": (t[v].mean() - c[v].mean()) / denom,
                     "SMD after": (matched.loc[matched["treated_group"].eq(1), v].mean()
                                   - matched.loc[matched["treated_group"].eq(0), v].mean()) / denom})
    run.add("Matched pairs", pairs)
    run.add("Balance fixed pre-match SD", pd.DataFrame(rows).set_index("variable"))
    run.add("Propensity diagnostic", f"Matched {len(i)} / {len(t)} eligible treated units.\n{fit.summary()}")
    return keep


def dcdh(p: pd.DataFrame, run: Run, **opts):
    import polars as pl
    from did_multiplegt_dyn import DidMultiplegtDyn
    controls = list(opts.get("controls", []))
    columns = list(dict.fromkeys(["id", "year", "Y", "D", *controls]))
    est = p.loc[:, columns].copy()
    # Preserve missing outcomes for the estimator's panel handling; forbid infinities.
    for col in ["Y", "D", *controls]:
        if np.isinf(est[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Infinite dCDH input {col}")
    est["gid"] = pd.factorize(est.pop("id"), sort=True)[0]
    options = {**run.cfg.dcdh, **opts}
    model = DidMultiplegtDyn(df=pl.from_pandas(est), outcome="Y", group="gid",
                             time="year", treatment="D", **options)
    model.fit()
    tab = model.summary()
    if not isinstance(tab, pd.DataFrame) or not {"Block", "Estimate", "SE"}.issubset(tab.columns):
        raise RuntimeError("Unsupported dCDH summary API: expected Block, Estimate, SE")
    raw = model.result["did_multiplegt_dyn"]
    return tab.set_index("Block"), raw


def dcdh_main(p: pd.DataFrame, run: Run):
    tab, raw = dcdh(p, run)
    run.add("dCDH main", tab)
    keys = ("p_jointplacebo", "p_jointeffects", "p_equality_effects")
    present = {key: raw[key] for key in keys if key in raw}
    absent = set(keys) - set(present)
    if absent:
        run.issue("WARNING", "dCDH tests", f"Unavailable API keys: {sorted(absent)}")
    if present:
        run.add("dCDH tests", pd.Series(present, name="p_value").to_frame())
    # Separate figure: dCDH horizons are intentionally NOT overlaid on adoption event times.
    import matplotlib.pyplot as plt
    events = tab[tab.index.astype(str).str.fullmatch(r"(?:Effect|Placebo)_\d+")]
    if not events.empty:
        k = events.index.astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()
        k *= np.where(events.index.astype(str).str.startswith("Placebo"), -1, 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(k, events.Estimate, yerr=1.96 * events.SE, fmt="o", capsize=3)
        ax.axhline(0, color="black")
        ax.set_xlabel("dCDH reported horizon (not TWFE relative year)")
        run.add("dCDH reported horizons", fig)
    return tab


def twfe_event(p: pd.DataFrame, run: Run):
    import re
    import matplotlib.pyplot as plt
    model = clustered("Y ~ C(rel, Treatment(reference=-1)) + C(id) + C(year)",
                      p, ["Y", "rel", "year"], run.cfg)
    names = [n for n in model.params.index if n.startswith("C(rel")]
    if not names:
        raise ValueError("No estimable event-time coefficients")
    k = np.array([int(re.search(r"\[T\.(-?\d+)\]", n).group(1)) for n in names])
    run.add("Naive TWFE event study", pd.DataFrame({"rel": k, "beta": model.params[names].values,
                                                  "se": model.bse[names].values}))
    run.add("TWFE interpretation", "Naive TWFE is diagnostic, not the preferred causal estimate under heterogeneous staggered effects.\n"
            f"Endpoints {run.cfg.rel_range} pool all more distant periods; reference is -1; first treated year is 0.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(k, model.params[names], yerr=1.96 * model.bse[names], fmt="s", capsize=3)
    ax.axhline(0, color="black")
    ax.set_xlabel("Years relative to first treatment; endpoints are pooled tails")
    run.add("TWFE event figure", fig)
    return model, names, k


def did2s_stage(p: pd.DataFrame, run: Run):
    pf = optional("pyfixest", run)
    if pf is None:
        return
    d = complete_sample(p, ["Y", "D", "rel", "year"])
    model = pf.did2s(d, yname="Y", first_stage="~ 0 | id + year",
                    second_stage="~ i(rel, ref=-1)", treatment="D", cluster="id")
    tab = model.tidy()
    if not isinstance(tab, pd.DataFrame):
        raise RuntimeError("Unsupported pyfixest tidy API")
    run.add("Gardner did2s", tab)


def csdid_stage(p: pd.DataFrame, run: Run):
    if optional("csdid", run) is None:
        return
    from csdid.att_gt import ATTgt
    d = complete_sample(p, ["Y", "D", "year"])
    # Python csdid commonly expects numeric unit IDs.
    d["gid"] = pd.factorize(d["id"], sort=True)[0] + 1
    d["cohort"] = d["first"].fillna(0).astype(int)
    att = ATTgt(yname="Y", tname="year", idname="gid", gname="cohort", data=d,
                control_group="notyettreated")
    att.fit(est_method="dr")  # Do not assume fit returns self.
    dynamic = att.aggte("dynamic")
    if isinstance(dynamic, pd.DataFrame):
        tab = dynamic
    elif isinstance(dynamic, dict):
        # Some versions return an AGGTE result object rather than a table/dict.
        tab = pd.DataFrame(dynamic)
    else:
        obj = dynamic if dynamic is not None else getattr(att, "aggte_results", None)
        fields = ["egt", "att_egt", "se_egt"]
        if obj is None or not all(hasattr(obj, f) for f in fields):
            raise RuntimeError("Unsupported csdid dynamic API; update adapter for installed version")
        tab = pd.DataFrame({f: getattr(obj, f) for f in fields})
    run.add("Callaway Sant Anna dynamic", tab)


def sensitivity_layout(model, names, k):
    order = np.argsort(k)
    ordered_k = np.asarray(k)[order]
    ordered_names = [names[i] for i in order]
    pre = int((ordered_k < 0).sum())
    post = int((ordered_k >= 0).sum())
    if pre == 0 or post == 0 or 0 not in ordered_k:
        raise ValueError("Sensitivity requires pre-period coefficients and treatment-year zero")
    beta = model.params[ordered_names].to_numpy()
    covariance = model.cov_params().loc[ordered_names, ordered_names].to_numpy()
    if beta.shape != (pre + post,) or covariance.shape != (len(beta), len(beta)):
        raise ValueError("Invalid sensitivity dimensions")
    if not np.isfinite(beta).all() or not np.isfinite(covariance).all():
        raise ValueError("Nonfinite sensitivity inputs")
    target = np.zeros(post)
    target[0] = 1
    return beta, covariance, pre, post, target


def sensitivity_stage(event, run: Run):
    hd = optional("honestdid", run)
    if hd is None:
        return
    model, names, k = event
    beta, covariance, pre, post, target = sensitivity_layout(model, names, k)
    function = getattr(hd, "createSensitivityResults_relativeMagnitudes", None)
    if function is None:
        raise RuntimeError("Installed honestdid lacks expected sensitivity API")
    run.issue("WARNING", "sensitivity", "Bounds use naive TWFE and pooled tail coefficients; not bounds on the preferred dCDH estimand")
    result = function(beta, covariance, pre, post, l_vec=target, Mbarvec=np.arange(0, 2.01, .25))
    run.add("TWFE sensitivity treatment-year-zero target", pd.DataFrame(result))


def provenance(cfg: Config) -> dict:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                            "config": asdict(cfg), "variables": VARIABLES,
                            "seed_scope": "NumPy global RNG only; package bootstrap RNGs are not guaranteed"}
    packages = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pyarrow", "wbgapi",
                "polars", "py-did-multiplegt-dyn", "pyfixest", "csdid", "honestdid"]
    info["packages"] = {}
    for package in packages:
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    info["script_sha256"] = fingerprint(Path(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=Path(__file__).resolve().parent, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                timeout=5, cwd=Path(__file__).resolve().parent, check=True)
        info["git"] = {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError) as exc:
        info["git"] = {"commit": None, "reason": str(exc)}
    return info


def report(run: Run, info: dict) -> None:
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    def text_pages(pdf, title, text):
        wrapped = []
        for line in str(text).expandtabs(4).splitlines() or [""]:
            wrapped.extend(textwrap.wrap(line, width=105, replace_whitespace=False,
                                         drop_whitespace=False) or [""])
        for i in range(0, len(wrapped), 68):
            fig = plt.figure(figsize=(8.5, 11))
            try:
                fig.text(.05, .97, title, fontsize=11, weight="bold", va="top")
                fig.text(.05, .93, "\n".join(wrapped[i:i + 68]), family="monospace",
                         fontsize=7, va="top", linespacing=1.2)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
    def build(path):
        with PdfPages(path) as pdf:
            status = "INCOMPLETE: required stage failed" if run.incomplete else "Required stages completed"
            text_pages(pdf, "Run status", status + "\n\n" + pd.DataFrame(run.stages).to_string(index=False)
                       + "\n\nOptional failures/skips and warnings are listed at the end.")
            text_pages(pdf, "Provenance", json.dumps(info, indent=2, default=str))
            for title, kind, item in run.sections:
                if kind == "text":
                    text_pages(pdf, title, item)
                else:
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    try:
                        ax.imshow(plt.imread(item))
                        ax.axis("off")
                        ax.set_title(title, fontsize=10)
                        pdf.savefig(fig, bbox_inches="tight")
                    finally:
                        plt.close(fig)
            text_pages(pdf, "Issues", "\n\n".join(
                f"{x['level']} {x['stage']}: {x['message']}" for x in run.issues) or "None")
    atomic_write(run.path / "report.pdf", build)


def execute(cfg: Config) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Run(cfg)
    np.random.seed(cfg.seed)
    if cfg.profile:
        tracemalloc.start()
    info = provenance(cfg)
    run.add("Profiling interpretation", "Stage timings use wall-clock time. Python traced peaks do not reliably cover all native allocations.\n"
            "process_maxrss_raw is cumulative process peak RSS (KiB on Linux; bytes on macOS), not a per-stage delta.\n"
            "Dense-design/matching budgets are conservative guards, not measured total-memory guarantees.")
    try:
        panel = run.call("Build and validate panel", lambda: build_panel(run), required=True)
        if panel is None:
            for name in ("Descriptives", "OLS", "Matching", "dCDH main", "TWFE", "did2s", "csdid", "Sensitivity"):
                run.skip(name, "Panel construction failed")
        else:
            run.call("Descriptives", lambda: descriptives(panel, run, "full"))
            run.call("OLS", lambda: ols_stage(panel, run), required=True)
            keep = None
            if cfg.treatment_mode == "binary" and not cfg.allow_left_censored:
                keep = run.call("Matching", lambda: matching(panel, run), required=True)
                if keep is not None:
                    matched = panel[panel.id.isin(keep)].copy()
                    run.call("Matched descriptives", lambda: descriptives(matched, run, "matched"))
            else:
                run.skip("Matching", "Adoption branch disabled for dose or left-censoring override")
            main = run.call("dCDH main", lambda: dcdh_main(panel, run), required=True)
            if main is not None:
                run.call("dCDH never-switchers", lambda: run.add("dCDH never-switchers",
                         dcdh(panel, run, only_never_switchers=True)[0]))
                if keep is not None:
                    run.call("dCDH matched", lambda: run.add("dCDH matched", dcdh(panel[panel.id.isin(keep)], run)[0]))
                else:
                    run.skip("dCDH matched", "No valid matched sample")
                if cfg.controls:
                    run.call("dCDH controls", lambda: run.add("dCDH controls", dcdh(panel, run, controls=list(cfg.controls))[0]))
            else:
                run.skip("dCDH robustness", "Main dCDH failed")
            if cfg.treatment_mode == "binary" and not cfg.allow_left_censored:
                event = run.call("TWFE event study", lambda: twfe_event(panel, run))
                run.call("did2s", lambda: did2s_stage(panel, run))
                run.call("csdid", lambda: csdid_stage(panel, run))
                if event is not None:
                    run.call("Sensitivity", lambda: sensitivity_stage(event, run))
                else:
                    run.skip("Sensitivity", "TWFE coefficients unavailable")
            else:
                for name in ("TWFE", "did2s", "csdid", "Sensitivity"):
                    run.skip(name, "Adoption branch disabled for dose or left-censoring override")
        info["sources"] = run.sources
        write_json(run.path / "provenance.json", info)
        write_json(run.path / "stages.json", run.stages)
        write_json(run.path / "issues.json", run.issues)
        run.call("PDF report", lambda: report(run, info), required=True)
        # call returns None for a successful side-effect stage: inspect status, not its return.
        write_json(run.path / "stages.json", run.stages)
        write_json(run.path / "issues.json", run.issues)
        print(f"{'INCOMPLETE' if run.incomplete else 'COMPLETE'}: {run.path}")
        return 1 if run.incomplete else 0
    finally:
        plt.close("all")
        if cfg.profile:
            tracemalloc.stop()
        run.close()


def self_test() -> int:
    """Offline regression tests. No WDI, estimator downloads, or treatment file needed."""
    import unittest
    from unittest.mock import patch
    from dataclasses import replace
    class Tests(unittest.TestCase):
        def test_year_cache_identity(self):
            self.assertNotEqual(cache_identity("wdi", "x", (2000, 2024)), cache_identity("wdi", "x", (2001, 2024)))
        def test_edited_source_fingerprint(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "t.csv"
                path.write_text("id,year,value\nA,2000,0\n")
                one = fingerprint(path)
                path.write_text("id,year,value\nA,2000,1\n")
                self.assertNotEqual(one, fingerprint(path))
                self.assertNotEqual(cache_identity("csv", str(path), (2000, 2024), one),
                                    cache_identity("csv", str(path), (2000, 2024), fingerprint(path)))
        def test_duplicate_keys(self):
            with self.assertRaises(ValueError):
                validate_long(pd.DataFrame({"id": ["a", "a"], "year": [2000, 2000], "value": [1, 2]}), "test")
        def test_invalid_log(self):
            with self.assertRaises(ValueError):
                transform(pd.Series([0., 1.]), "log", "test")
        def test_missing_outcome_cluster_alignment(self):
            d = pd.DataFrame({"id": np.repeat(["a", "b", "c", "d"], 5),
                              "year": np.tile(np.arange(2000, 2005), 4),
                              "X": np.tile([0, 1, 2, 3, 4], 4), "Y": np.arange(20, dtype=float)})
            d.loc[1, "Y"] = np.nan
            model = clustered("Y ~ X", d, ["Y", "X"], Config())
            self.assertEqual(model.nobs, 19)
        def test_missing_treatment(self):
            p = pd.DataFrame({"id": ["a", "a"], "year": [2000, 2001], "D": [0, np.nan]})
            with self.assertRaises(ValueError):
                treatment_history(p, Config())
        def test_reversal(self):
            p = pd.DataFrame({"id": ["a"] * 3, "year": [2000, 2001, 2002], "D": [0, 1, 0]})
            with self.assertRaises(ValueError):
                treatment_history(p, Config())
        def test_early_baseline(self):
            p = pd.DataFrame({"id": ["a", "b"], "year": [2005, 2005], "first": [2006, np.nan],
                              "treated_group": [1, 0], "pop_growth": [1., 2.], "income": [3., 4.]})
            with self.assertRaises(ValueError):
                baseline_sample(p, Config())
        def test_left_censoring(self):
            p = pd.DataFrame({"id": ["a", "a"], "year": [2000, 2001], "D": [1, 1]})
            with self.assertRaises(ValueError):
                treatment_history(p, Config())
        def test_empty_matches(self):
            i, j, d = optimal_pairs(np.array([0.]), np.array([10.]), .1, 10000)
            self.assertEqual(len(i), 0)
        def test_matching_cardinality(self):
            i, j, distances = optimal_pairs(np.array([0., .09]), np.array([.08, .18]), .1, 10000)
            self.assertEqual(len(i), 2)
            self.assertTrue((distances <= .1).all())
        def test_matching_budget(self):
            with self.assertRaises(MemoryError):
                optimal_pairs(np.zeros(10), np.zeros(10), .1, 1)
        def test_sensitivity_zero(self):
            class Fake:
                params = pd.Series([1., 2., 3.], index=["pre", "zero", "post"])
                def cov_params(self):
                    return pd.DataFrame(np.eye(3), index=self.params.index, columns=self.params.index)
            beta, cov, pre, post, target = sensitivity_layout(Fake(), ["pre", "zero", "post"], np.array([-2, 0, 1]))
            self.assertEqual((pre, post), (1, 2))
            self.assertEqual(len(beta), pre + post)
            np.testing.assert_array_equal(target, [1, 0])
        def test_missing_git(self):
            with patch("subprocess.run", side_effect=FileNotFoundError("git")):
                self.assertIsNone(provenance(Config())["git"]["commit"])
        def test_optional_failure_isolation(self):
            with tempfile.TemporaryDirectory() as tmp:
                run = Run(replace(Config(), output_dir=tmp))
                try:
                    def fail():
                        raise RuntimeError("optional failure")
                    self.assertIsNone(run.call("optional", fail))
                    self.assertEqual(run.call("next", lambda: 42), 42)
                    self.assertFalse(run.incomplete)
                    run.call("required", fail, required=True)
                    self.assertTrue(run.incomplete)
                finally:
                    run.close()
        def test_atomic_failure_preserves_output(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "a.txt"
                path.write_text("old")
                def fail(p):
                    p.write_text("partial")
                    raise RuntimeError("interrupted")
                with self.assertRaises(RuntimeError):
                    atomic_write(path, fail)
                self.assertEqual(path.read_text(), "old")
        def test_optional_missing_package(self):
            with tempfile.TemporaryDirectory() as tmp:
                run = Run(replace(Config(), output_dir=tmp))
                try:
                    error = ModuleNotFoundError("absent", name="absent_test_package")
                    with patch("importlib.import_module", side_effect=error):
                        self.assertIsNone(run.call("optional adapter", lambda: optional("absent_test_package", run)))
                    self.assertEqual(run.stages[-1]["status"], "skipped_dependency")
                    self.assertFalse(run.incomplete)
                finally:
                    run.close()
        def test_report_and_figure_lifetime(self):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            with tempfile.TemporaryDirectory() as tmp:
                run = Run(replace(Config(), output_dir=tmp))
                try:
                    fig, ax = plt.subplots()
                    number = fig.number
                    ax.plot([0, 1], [0, 1])
                    run.add("test figure", fig)
                    self.assertNotIn(number, plt.get_fignums())
                    self.assertIsInstance(run.sections[0][2], Path)
                    run.add("wide text", "x" * 500 + "\n" + "row\n" * 150)
                    report(run, {"test": True})
                    pdf = run.path / "report.pdf"
                    self.assertGreater(pdf.stat().st_size, 0)
                    self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
                finally:
                    run.close()
        def test_spreadsheet_text(self):
            self.assertEqual(spreadsheet_text(" =SUM(A1)"), "' =SUM(A1)")
            self.assertEqual(spreadsheet_text(-2.5), -2.5)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--treatment", default="data/raw/treatment.csv")
    parser.add_argument("--years", nargs=2, type=int, default=(2000, 2024), metavar=("START", "END_EXCLUSIVE"))
    parser.add_argument("--baseline", nargs=2, type=int, default=(2005, 2007))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--spreadsheet-safe", action="store_true")
    parser.add_argument("--missing-treatment-zero", action="store_true", help="Explicitly treat unrecorded treatment as zero")
    parser.add_argument("--allow-left-censored", action="store_true", help="Disable adoption branches and allow already-treated entrants in dCDH")
    parser.add_argument("--treatment-mode", choices=("binary", "dose"), default="binary")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--wdi-max-age-days", type=int, default=30)
    parser.add_argument("--min-baseline-observations", type=int, default=3,
                        help="Required non-missing baseline observations per covariate")
    args = parser.parse_args()
    if args.years[0] >= args.years[1] or args.baseline[0] > args.baseline[1]:
        parser.error("Invalid period bounds")
    if args.wdi_max_age_days < 0:
        parser.error("Cache age must be nonnegative")
    if args.min_baseline_observations < 1:
        parser.error("Minimum baseline observations must be positive")
    if not args.years[0] <= args.baseline[0] <= args.baseline[1] < args.years[1]:
        parser.error("Baseline must be inside the analysis period")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    cfg = Config(years=tuple(args.years), baseline=tuple(args.baseline),
                 min_baseline_observations=args.min_baseline_observations,
                 treatment=args.treatment, refresh=args.refresh, profile=args.profile,
                 spreadsheet_safe=args.spreadsheet_safe, missing_treatment_zero=args.missing_treatment_zero,
                 allow_left_censored=args.allow_left_censored, treatment_mode=args.treatment_mode,
                 output_dir=args.output_dir, cache_dir=args.cache_dir, wdi_max_age_days=args.wdi_max_age_days)
    return execute(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
