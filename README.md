# Zoning bylaws and housing supply

Two programs:

- **`baseline.py`**: baseline OLS on one final dataset, written to a single PDF. Start
  here to see whether a DiD or RDD design is worth building.
- **`project.py`**: the full staggered difference-in-differences toolkit, with a
  terminal menu.

## Baseline regressions: `baseline.py`

```
python baseline.py data/final.xlsx              # writes data/final_baseline.pdf
python baseline.py data/final.csv -o report.pdf
```

The dataset is one `.csv` or `.xlsx` table with a row per census division and period
(monthly, quarterly or annual). Set the column names, the frequency and the controls
in the SETTINGS block at the top of `baseline.py`. Names are matched ignoring case,
spaces and punctuation, so `CPI Shelter` matches `cpi_shelter`. If a column is
missing, the program lists the file's columns and the closest match.

| Setting | Default | Meaning |
|---|---|---|
| `UNIT`, `TIME`, `FREQ` | `census_division`, `date`, `"Q"` | the panel: id, period (dates, years or `2015Q1`), frequency |
| `SCORE` | `score` | the internal zoning score |
| `OUTCOMES` | `permits`, `starts` | Table 34-10-0292-01 permits; CMHC starts |
| `TREATED` | `treatment_date` | the date the bylaw took effect; blank if never |
| `CONTROLS` | mortgage lending, development charges, BCPI, NHPI, CPI shelter, unemployment | added in models (3) and (4) |
| `EXTRA` | `completions`, `under_construction` | summary statistics only |
| `LOG`, `CUTOFF`, `WINDOW` | `True`, `None`, `8` | log(1 + outcome); a score threshold to mark (RDD check); pre-trend window |

The models, with standard errors clustered by census division:

```
(1) permits ~ score        (3) permits ~ score + post + controls
(2) starts ~ score         (4) starts ~ score + post + controls
```

`post` is 1 from the period the bylaw took effect. The PDF has five parts:

1. Data and summary statistics: sample, missing values, means before the first bylaw
   (adopters against never-adopters), correlations.
2. Trends and pre-trends: mean outcomes by period for both groups, and the adopter
   gap by period since adoption, measured from the period before (95% band).
3. The score: its distribution, and binned means of each outcome by score with the
   OLS line (split at `CUTOFF` when set).
4. The regression table with diagnostics: R-squared, Breusch-Pagan, Jarque-Bera,
   Durbin-Watson, RESET and the largest variance inflation factor. It also has a
   pre-trend test: the difference in pre-adoption slopes, with division fixed effects.
5. Residual plots for models (3) and (4).

These are associations, not causal effects. `baseline.py` reads only the columns it
needs, refuses files over `MAX_MB`, never turns column names into code (models are
built from arrays, not formula strings), and writes the PDF to a temporary file first.
It needs only numpy, pandas, matplotlib, statsmodels, openpyxl and defusedxml:

```
python -m pip install numpy==2.3.5 pandas==3.0.6 matplotlib==3.11.2 statsmodels==0.15.0 openpyxl==3.1.5 defusedxml==0.7.1
```

## The DiD toolkit: `project.py`

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

`test_project.py` covers the settings, file reading, frequency matching, treatment
timing, the four sections on the demo data (including that the placebo is near zero and
the known effect is recovered), the HTML output and a scripted menu session.
`test_baseline.py` checks that `baseline.py` recovers a known slope, detects a planted
pre-trend, reads CSV and Excel, and still writes a report when nobody has adopted.
