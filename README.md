# Zoning bylaws and housing supply

`project.py` measures how municipal zoning bylaws change housing starts (quarterly) and
building permits (annual), using staggered difference-in-differences. It runs from a
terminal menu, and every setting lives in one file, `config.toml`.

| Section | What it does |
|---|---|
| 1. Pre-tests and validity | Data coverage, adoption cohorts, vote-to-effective lags, parallel-trends tests (TWFE, did2s, dCDH), a placebo with adoption moved earlier, robustness checks (vote date, no controls, without the implementation lag, TWFE, dCDH with never-treated comparisons only), and covariate balance |
| 2. Basic OLS | Each model in `[ols] models` (by default `permits ~ population` and `starts ~ population`): pooled, pooled with the controls, and with municipality and period fixed effects, plus a scatter plot |
| 3. Housing starts, quarterly | Event study and average effect from three estimators, and effects by zoning type, initiative type and bylaw score |
| 4. Building permits, annual | The same, on annual data |

## Install

Python 3.11 or newer, in a virtual environment:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins every package with hashes; `requirements.txt` lists only
the direct dependencies.

## Try it

```
python project.py --demo
```

This writes made-up data to `data/demo/` (60 municipalities, 2010-2024, with a known
bylaw effect of about 10%) and opens the menu on it.

## Use your own data

1. Put the files in `data/` and list them under `[data] files` in `config.toml`.
2. Check the column names in `config.toml` (the unit column, outcomes, bylaw dates and
   controls). Names are matched after cleaning, so `FVI_CSCE_AB` and `fvi_csce_ab` both work.
3. Run `python project.py`. The program reads the files and shows a data check before
   anything is estimated:

```
 ✓  starts_quarterly.csv      municipal panel, quarterly, 2010Q1 to 2024Q4, 3,600 rows: starts
 ✓  bylaws                    41 municipalities with a bylaw, 40 with a date in effective_date
 ✓  log(starts) (Section 3)   quarterly, 2010Q1 to 2024Q4: 58 municipalities, 38 adopt a bylaw,
                              20 never do, 2 left out
 !                            Left out of the DiD because their bylaw was already in effect when
                              their data start: Municipality 36, Municipality 50
 !  control policy_rate       boc_valet.csv (time series, monthly); is the same for every
                              municipality in a period, so period fixed effects absorb it
```

Then pick a section from the menu. Each one first says what it will estimate (sample,
window, treatment date, controls) and asks before running.

### What the files can look like

| File | Kind | Columns (example) |
|---|---|---|
| Housing starts | municipal panel | `municipality`, `ref_date` (StatCan style: `2015-01` is 2015Q1), `starts` |
| Building permits | municipal panel | `municipality`, `year`, `permits` |
| Municipal controls | municipal panel | `municipality`, `year`, `province`, `population`, `pop_growth`, `pop_density`, `median_income` |
| Zoning bylaws | a row per bylaw | `municipality`, `council_vote_date`, `effective_date`, `bylaw_score`, `zoning_type`, `initiative_type` |
| Bank of Canada series | time series | a Valet CSV download as is: `date`, `V39079`, `FVI_CSCE_AB`, ..., `FVI_CSCE_SK` |
| CPI shelter | time series | `date`, `cpi_shelter` |

A time column is named `date`, `ref_date`, `period`, `quarter`, `year`, `month` or
`time`, and holds years, quarters (`2015Q1`) or dates. Frequencies are matched
automatically: monthly or daily series are averaged within each quarter or year, and
annual controls repeat within the year. CSV and Excel (`.xlsx`) files both work.

### Controls and the market barometer

Add or delete lines in `[controls] use`; the menu (`s`) can also switch controls on and
off for a session. `market_barometer` averages the z-scores of the series in
`[market_barometer] series`; `fvi_csce_{region}` takes each municipality's own regional
series, with `[regions]` mapping provinces to region codes (NB, NL, NS and PE use ATL).
Put `-` in front of a series that moves against the market. `[rename]` gives coded
columns readable names (`V39079 = "policy_rate"`).

A series that is the same for every municipality in a period (the policy rate, national
CPI shelter) is absorbed by the period fixed effects of the DiD, so it is used in pooled
OLS only; the data check marks it.

## Output

Each run makes a new folder, `output/<date-time>_<sections>/`, with:

- `index.html`: every table, figure and note on one page;
- a CSV file per table and a PNG file per figure;
- `.tex` regression tables (pyfixest `etable`);
- `run_info.json`: the settings, the SHA-256 of each data file and the package versions.

Without the menu: `python project.py --run 3 4` (or `--run all`).

## Method notes

- Treatment starts in the period of each municipality's earliest bylaw, at the effective
  date or the council vote date (`[treatment] use`). Municipalities without a bylaw, or
  whose bylaw takes effect after the data end, are the comparison group. Municipalities
  whose bylaw was already in effect when their data start are left out.
- Event studies use the period before adoption (-1) as the reference, and pool periods
  outside the window into its end points. TWFE is shown as a diagnostic: with staggered
  adoption and changing effects it can be biased. did2s (Gardner 2022) and dCDH
  (de Chaisemartin and D'Haultfœuille 2024) are robust to that.
- Standard errors are clustered by municipality. With few municipalities or adopters,
  they can be too small; Section 1 says so.
- Outcomes are in logs by default (`transform = "log"` drops zeros, `"log1p"` keeps them),
  so effects read as approximate percent changes; the tables add the exact `% change`.

## Tests

```
python -m pytest -q
```

The tests cover the settings, file reading, frequency matching, treatment timing, the
four sections on the demo data (including that the placebo is near zero and the known
effect is recovered), the HTML output and a scripted menu session.
