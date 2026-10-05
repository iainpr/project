# Panel DiD toolkit

`project.py` analyzes a panel dataset (CSV or Excel) and runs descriptives, OLS with
controls, propensity score matching, and staggered difference-in-differences estimators,
from an interactive menu or from saved settings.

## Install

Use a virtual environment with Python 3.11+:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt   # exact, hashed versions
```

`requirements.txt` lists the same direct dependencies without the transitive pins.
The HonestDiD stage is optional because `honestdid` depends on PyTorch:

```
python -m pip install -r requirements-honestdid.txt
```

On Linux, install the CPU-only PyTorch first (`--index-url https://download.pytorch.org/whl/cpu`)
or pip downloads several GB of CUDA libraries. A stage whose package is missing is
reported as skipped; the rest still run.

## Use

```
python project.py --demo                       # try the menu on a synthetic panel
python project.py --file data/raw/panel.xlsx   # menu with your file loaded
python project.py --analyze --file data/raw/panel.csv
python project.py --settings settings.json --run all
python project.py --self-test
```

The data must be long: one row per unit and period. The time column can hold years,
quarter labels (`2015Q1`), month labels (`2015-01`) or dates.

1. **Choose data file.** The file is analyzed straight away. The analysis reports:
   - the unit and time columns, frequency and balance;
   - treatment candidates with their adoption timing;
   - series that are the same for every unit in a period (national rates) and so are
     absorbed by period fixed effects;
   - lower-frequency series (annual data in a quarterly file);
   - province families such as `FVI_CSCE_AB ... FVI_CSCE_ATL`, which can become one
     province-matched column;
   - columns that duplicate others (a mean score next to its components);
   - post-treatment columns.
2. **Set variable roles.** Choose the outcome, treatment, controls and matching covariates.
   Treatment can be a policy date per unit, a 0/1 indicator, or a dose (dCDH only).
3. **Options.** Event window, matching baseline and caliper, sample period, annual
   collapse (choose which flows to sum), dCDH horizons, and HonestDiD settings.
4. **Run** any stage on its own, or several at once. Each run creates
   `output/<time>_<stages>_<id>/` with `report.pdf`, `tables/*.csv`, `figures/*.png`,
   `run.log`, `settings.json`, `stages.json`, `issues.json` and `provenance.json`.

Choosing **Save settings** writes the current roles and options to JSON;
`--settings that.json --run ...` replays the analysis without prompts. Keep the settings
file in git next to the code.

## Estimators

| Stage | Package | Needs |
|---|---|---|
| OLS with controls (pooled, two-way FE) | statsmodels | outcome; treatment or regressors |
| Propensity score matching | statsmodels, scipy | absorbing 0/1 treatment, never-treated units, covariates |
| dCDH (all, never-switchers, matched, with controls) | py-did-multiplegt-dyn | any treatment |
| Callaway-Sant'Anna | csdid | absorbing 0/1 treatment |
| Gardner did2s | pyfixest | absorbing 0/1 treatment |
| TWFE event study (diagnostic) | statsmodels | absorbing 0/1 treatment |
| HonestDiD sensitivity | honestdid | TWFE event study |

Standard errors are clustered by unit. The matched dCDH clusters by matched pair. With
fewer than about 30 units, cluster-robust standard errors are unreliable, and every
stage says so.
