"""Write made-up data in the layout config.toml expects, to try project.py.

    python project.py --demo     writes the data below, then opens the menu on it
    python demo.py               only writes the data (to data/demo/)

60 municipalities in 10 provinces, 2010-2024. 40 adopt a zoning bylaw: 38 between 2015
and 2022, and 2 before the data start (so they are left out of the DiD). One more bylaw
passed council in late 2024 and is not in effect yet. Housing starts and permits rise by
about 3-17% once a bylaw takes effect, more for higher bylaw scores. Every number is
simulated; the files only mimic the layout of StatCan and Bank of Canada downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
PROVINCES = (
    ("ON",) * 19
    + ("QC",) * 12
    + ("BC",) * 9
    + ("AB",) * 8
    + ("MB", "SK") * 3
    + (
        "NS",
        "NS",
        "NB",
        "NB",
        "NL",
        "PE",
    )
)
ADOPTERS = 40  # the first two adopted before the data start
REGION = {"NB": "ATL", "NL": "ATL", "NS": "ATL", "PE": "ATL"}
SEASON = np.array([-0.30, 0.10, 0.15, 0.05])  # quarterly pattern in starts
# A policy-rate path like the Bank of Canada's: (first month, rate in percent)
POLICY_RATE = [
    ("2010-01", 0.25),
    ("2010-06", 0.50),
    ("2010-07", 0.75),
    ("2010-09", 1.00),
    ("2015-01", 0.75),
    ("2015-07", 0.50),
    ("2017-07", 0.75),
    ("2017-09", 1.00),
    ("2018-01", 1.25),
    ("2018-07", 1.50),
    ("2018-10", 1.75),
    ("2020-03", 0.25),
    ("2022-03", 0.50),
    ("2022-04", 1.00),
    ("2022-06", 1.50),
    ("2022-07", 2.50),
    ("2022-09", 3.25),
    ("2022-10", 3.75),
    ("2022-12", 4.25),
    ("2023-01", 4.50),
    ("2023-06", 4.75),
    ("2023-07", 5.00),
    ("2024-06", 4.75),
    ("2024-07", 4.50),
    ("2024-09", 4.25),
    ("2024-10", 3.75),
    ("2024-12", 3.25),
]

CONFIG = """\
# Settings for the demo data written by demo.py. config.toml explains every setting.

[data]
files = [
    "starts_quarterly.csv",
    "permits_annual.csv",
    "municipal_controls.csv",
    "zoning_bylaws.csv",
    "boc_valet.csv",
    "cpi_shelter.csv",
]
unit = "municipality"
region = "province"

[rename]
V39079 = "policy_rate"

[outcomes]
starts = "starts"
permits = "permits"
transform = "log"

[treatment]
vote_date = "council_vote_date"
effective_date = "effective_date"
use = "effective"
score = "bylaw_score"
zoning_type = "zoning_type"
initiative = "initiative_type"

[controls]
use = [
    "pop_growth",
    "population",
    "pop_density",
    "median_income",
    "market_barometer",
    "cpi_shelter",
    "policy_rate",
]

[market_barometer]
series = ["fvi_csce_{region}"]

[regions]
NB = "ATL"
NL = "ATL"
NS = "ATL"
PE = "ATL"

[ols]
models = [
    "permits ~ population",
    "starts ~ population",
]
logs = true

[did]
window_quarterly = [-8, 12]
window_annual = [-4, 6]
placebo_shift_quarterly = 8
placebo_shift_annual = 2

[output]
folder = "../../output"
"""


def write_demo(folder: Path, seed: int = SEED) -> Path:
    """Write the demo files and their config.toml into folder; return the config path."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(PROVINCES)
    names = [f"Municipality {i:02d}" for i in range(1, n + 1)]
    years = np.arange(2010, 2025)
    quarters = pd.period_range("2010Q1", "2024Q4", freq="Q")
    months = pd.period_range("2010-01", "2024-12", freq="M")

    # National and regional series: monthly policy rate and CPI shelter, and a quarterly
    # regional indicator published in the first month of each quarter.
    steps = pd.Series(dict(POLICY_RATE)).rename(index=pd.Period)
    rate = steps.reindex(months).ffill().to_numpy()
    growth = np.where(months.year >= 2022, 0.005, 0.002) + rng.normal(0, 0.001, len(months))
    cpi = 120 * np.exp(np.cumsum(growth))
    regions = sorted(set(REGION.get(p, p) for p in PROVINCES))
    cycle = np.cumsum(rng.normal(0, 0.3, len(quarters)))
    fvi = {r: 50 + 4 * cycle + rng.normal(0, 2, len(quarters)).cumsum() * 0.5 for r in regions}
    valet = pd.DataFrame({"date": months.to_timestamp().strftime("%Y-%m-%d"), "V39079": rate})
    first_month = (months.month - 1) % 3 == 0
    for r in regions:
        valet[f"FVI_CSCE_{r}"] = np.nan
        valet.loc[first_month, f"FVI_CSCE_{r}"] = fvi[r].round(2)
    header = [
        '"TERMS AND CONDITIONS"',
        '"https://www.bankofcanada.ca/terms/"',
        "",
        '"SERIES"',
        '"id","label","description"',
        '"V39079","Policy rate","Made-up policy rate for the demo"',
        *(f'"FVI_CSCE_{r}","Indicator {r}","Made-up regional indicator"' for r in regions),
        "",
        '"OBSERVATIONS"',
    ]
    text = "\n".join(header) + "\n" + valet.to_csv(index=False, float_format="%.2f")
    (folder / "boc_valet.csv").write_text(text, encoding="utf-8")
    shelter = pd.DataFrame({"date": months.to_timestamp().strftime("%Y-%m-%d"), "cpi_shelter": cpi})
    shelter.to_csv(folder / "cpi_shelter.csv", index=False, float_format="%.1f")

    # Municipalities: annual population, density and income
    pop0 = np.exp(rng.normal(np.log(60_000), 0.9, n))
    pop_growth = rng.normal(0.015, 0.006, n)
    area = rng.uniform(40, 600, n)
    income0 = rng.normal(70_000, 9_000, n)
    t = years - years[0]
    population = (
        pop0[:, None] * (1 + pop_growth[:, None]) ** t * np.exp(rng.normal(0, 0.004, (n, len(t))))
    )
    previous = np.column_stack([population[:, 0] / (1 + pop_growth), population[:, :-1]])
    income = income0[:, None] * 1.025**t * np.exp(rng.normal(0, 0.01, (n, len(t))))
    controls = pd.DataFrame(
        {
            "municipality": np.repeat(names, len(years)),
            "year": np.tile(years, n),
            "province": np.repeat(PROVINCES, len(years)),
            "population": population.ravel().round(),
            "pop_growth": (100 * (population / previous - 1)).ravel().round(2),
            "pop_density": (population / area[:, None]).ravel().round(1),
            "median_income": income.ravel().round(-2),
        }
    )
    controls.to_csv(folder / "municipal_controls.csv", index=False)

    # Zoning bylaws: larger municipalities adopt more often; a few pass a second bylaw later
    k = ADOPTERS
    adopters = rng.choice(n, k, replace=False, p=pop0 / pop0.sum() * 0.5 + 0.5 / n)
    days = pd.to_timedelta(rng.integers(0, 8 * 365, k - 2), "D")
    vote = pd.DatetimeIndex(["2008-05-12", "2009-03-03"]).append(pd.Timestamp("2015-01-01") + days)
    effective = vote + pd.to_timedelta(rng.integers(60, 271, k), "D")
    score = rng.uniform(1, 5, k).round(1)
    bylaws = pd.DataFrame(
        {
            "municipality": [names[i] for i in adopters],
            "bylaw": [f"{v.year}-{rng.integers(10, 99)}" for v in vote],
            "council_vote_date": vote.strftime("%Y-%m-%d"),
            "effective_date": effective.strftime("%Y-%m-%d"),
            "bylaw_score": score,
            "zoning_type": rng.choice(
                ["Gentle density", "Mid-rise corridors", "Secondary suites"], k
            ),
            "initiative_type": rng.choice(
                ["Municipal", "Provincial direction", "Federal funding"], k
            ),
        }
    )
    later = bylaws.iloc[[5, 9, 14]].assign(bylaw=lambda b: b["bylaw"] + "A")
    for col in ("council_vote_date", "effective_date"):
        later[col] = (pd.to_datetime(later[col]) + pd.DateOffset(years=2)).dt.strftime("%Y-%m-%d")
    never = sorted(set(range(n)) - set(adopters))
    pending = {
        "municipality": names[never[0]],
        "bylaw": "2024-77",
        "council_vote_date": "2024-10-15",
        "effective_date": "",
        "bylaw_score": 3.0,
        "zoning_type": "Gentle density",
        "initiative_type": "Municipal",
    }
    pd.concat([bylaws, later, pd.DataFrame([pending])]).to_csv(
        folder / "zoning_bylaws.csv", index=False
    )

    # Housing starts (quarterly) and permits (annual), with the bylaw effect after it takes effect
    region_of = [REGION.get(p, p) for p in PROVINCES]
    fvi_z = {r: (v - v.mean()) / v.std() for r, v in fvi.items()}
    level = rng.normal(0, 0.3, n) + 0.1 * np.isin(np.arange(n), adopters)
    adoption = pd.Series(pd.PeriodIndex(effective, freq="Q"), index=adopters)
    tau = pd.Series(0.03 + 0.035 * (score - 1), index=adopters)
    q = np.arange(len(quarters))
    quarter_rate = pd.Series(rate, index=months).groupby(months.asfreq("Q")).mean().to_numpy()
    rows, permits = [], []
    for i in range(n):
        pop_q = population[i, quarters.year - years[0]]
        log_mu = (
            np.log(0.0016 * pop_q)
            + level[i]
            + SEASON[quarters.quarter - 1]
            + 0.10 * fvi_z[region_of[i]]
            - 0.04 * quarter_rate
            + 0.05 * np.sin(q / 6)
        )
        if i in adoption.index:
            since = (quarters - adoption[i]).map(lambda d: d.n).to_numpy()
            log_mu = log_mu + np.where(since >= 0, tau[i] * np.minimum(1, (since + 1) / 4), 0)
        mu = np.exp(log_mu)
        rows.append(rng.poisson(mu * np.exp(rng.normal(0, 0.10, len(q)))))
        yearly = pd.Series(mu).groupby(quarters.year).sum().to_numpy()
        permits.append(rng.poisson(1.15 * yearly * np.exp(rng.normal(0, 0.06, len(years)))))
    starts = pd.DataFrame(
        {
            "municipality": np.repeat(names, len(quarters)),
            "ref_date": np.tile(
                quarters.to_timestamp().strftime("%Y-%m"), n
            ),  # StatCan style: 2010-01 is 2010Q1
            "starts": np.concatenate(rows),
        }
    )
    starts.to_csv(folder / "starts_quarterly.csv", index=False)
    pd.DataFrame(
        {
            "municipality": np.repeat(names, len(years)),
            "year": np.tile(years, n),
            "permits": np.concatenate(permits),
        }
    ).to_csv(folder / "permits_annual.csv", index=False)

    config = folder / "config.toml"
    config.write_text(CONFIG, encoding="utf-8")
    return config


if __name__ == "__main__":
    path = write_demo(Path(__file__).resolve().parent / "data" / "demo")
    print(f"Demo data written to {path.parent}. Run: python project.py --config {path}")
    sys.exit(0)
