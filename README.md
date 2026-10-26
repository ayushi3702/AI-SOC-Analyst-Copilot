# AI-SOC-Analyst-Copilot

A smart assistant for security teams: it filters false alarms, investigates real
threats like a junior analyst, and writes the report for a human to approve.

This repository currently contains the CICIDS2017 data-cleaning and DuckDB
ingestion pipeline, exploratory notebooks, and binary/family generalization
experiments. The alert-clustering module is not implemented yet.

## Project layout

```text
configs/
  features.yaml                  Feature selection settings
  schema.yaml                    Common event-field mapping
  settings.yaml                  Database, split, and family settings
data/
  raw/cicids2017/                 Source CICIDS2017 CSV files
  interim/                       Cleaned per-capture Parquet files and report
  soc.duckdb                     DuckDB database (generated)
ingest/
  clean.py                       CICIDS2017 cleanup and normalization helpers
  load_cicids.py                 CSV-to-Parquet cleaning pipeline
  normalize.py                   Normalizes cleaned rows into DuckDB tables
  to_duckdb.py                   Loads Parquet files into DuckDB
ml/
  data.py                        Loads/configures the modelling frame
  features.py                    Selects numeric model features
  splits.py                      Random, time-block, unseen-day, and family splits
  train_intrusion.py             XGBoost binary/family baseline
  leave_one_family_out.py        XGBoost vs Isolation Forest evaluation
  evaluate.py                    Validation-threshold alert-reduction helpers
notebooks/
  01_explore_cicids.ipynb        Dataset cleaning and exploratory analysis
  02_leakage_demo.ipynb          Split leakage and family-generalization charts
evaluation/results/
  leave_one_family_out.csv       Per-family evaluation results (generated)
  leakage_and_generalization.png Notebook chart output (generated)
```

## Environment

Use Python and install the packages in `requirements.txt`. For example, from
the repository root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands below from the repository root so relative configuration paths
such as `configs/settings.yaml` resolve correctly. The project virtual
environment can be used explicitly with `.\venv\Scripts\python.exe`.

## Data preparation

The source data is the CICIDS2017 TrafficLabelling/GeneratedLabelledFlows
dataset, which includes timestamps and IP addresses. Place its capture CSVs in
`data/raw/cicids2017/`. The cleaner expects the source columns for flow ID,
source/destination IP, timestamp, and label; the MachineLearningCVE variant
does not provide the required schema.

Clean each CSV separately and save cleaned Parquet files:

```powershell
python -m ingest.load_cicids
```

The loader normalizes column names, makes duplicate column names unique,
repairs labels and CICIDS2017 timestamps, drops empty rows, invalid
infinity/NaN values, bad timestamps, and exact duplicate rows, creates
`is_attack`, and reduces numeric storage sizes. It writes the cleaned files and
`data/interim/_cleaning_report.csv`. Existing Parquet outputs are skipped; use
`--force` to redo them:

```powershell
python -m ingest.load_cicids --force
```

Load the cleaned Parquet files into DuckDB:

```powershell
python -m ingest.to_duckdb
```

The database is `data/soc.duckdb`, with aligned `events`, `labels`, and
`flow_features` tables. The script prints row counts, attacks by day, and top
talkers. It refuses to load into a database that already contains tables;
rebuild intentionally with:

```powershell
python -m ingest.to_duckdb --force
```

`--force` deletes and recreates the selected database file. Use it only when
you intend to replace that database.

## Exploratory analysis

Open `notebooks/01_explore_cicids.ipynb` in VS Code or Jupyter and run its cells
from top to bottom. It reads the cleaned Parquet files and cleaning report to
inspect:

1. Rows removed during cleaning.
2. Label counts and class balance.
3. Timestamp spans by capture file.
4. Attack windows and attack types by day.
5. Attack types present only in the Thursday-Friday test days.
6. Near-duplicate feature rows in the Tuesday capture.

The cleaned dataset currently has **2,827,677 flows**. The DuckDB
`events`, `labels`, and `flow_features` tables each contain 2,827,677 rows.
The `to_duckdb` daily flow/attack counts were:

| Day | Flows | Attack flows |
|---|---:|---:|
| Friday | 702,713 | 288,785 |
| Monday | 529,450 | 0 |
| Thursday | 458,483 | 2,216 |
| Tuesday | 445,641 | 13,832 |
| Wednesday | 691,390 | 251,722 |

The notebook's chronological holdout uses Monday-Wednesday for training and
Thursday-Friday for testing. The attack labels occurring only on those test
days were:

```text
['Bot', 'DDoS', 'Infiltration', 'PortScan',
 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS']
```

The notebook's near-duplicate check selected
`tuesday_workinghours.parquet`; **5.4%** of its flows repeat the chosen numeric
feature values of an earlier flow. The check excludes `is_attack` and
`source_port` from the duplicate feature set.

## Model experiments

`configs/settings.yaml` controls the DuckDB path, random seed, sample fraction,
10-minute time-block size, train/test days, and mapping from labels to attack
families. `configs/features.yaml` controls dropped fields and whether
destination port is used. By default, modelling uses all rows and excludes
destination port.

Train the binary XGBoost baseline for each split:

```powershell
python -m ml.train_intrusion --split random --task binary
python -m ml.train_intrusion --split block --task binary
python -m ml.train_intrusion --split unseen --task binary
```

The split modes are:

- `random`: stratified random split; used as a potentially leaky comparison.
- `block`: hold out complete 10-minute time blocks.
- `unseen`: train on Monday-Wednesday and test on Thursday-Friday.

The script logs parameters and metrics to the MLflow `intrusion_baseline`
experiment, and prints the classification report. Binary runs log attack
recall, PR-AUC, and ROC-AUC in addition to macro F1. A family-classification
run is also supported with `--task family`.

Recent no-port binary runs recorded these attack recalls:

| Split | Attack recall |
|---|---:|
| Random | 0.999605 |
| Block | 0.998904 |
| Unseen days | 0.007780 |

The sharp drop for the unseen-day split demonstrates that strong random and
time-block results do not imply generalization to attack types absent from
training.

### Leave-one-family-out comparison

Run the family generalization experiment:

```powershell
python -m ml.leave_one_family_out
```

For each non-benign family, the split excludes that family from training and
tests on its rows plus benign rows from held-out time blocks. It compares a
supervised XGBoost classifier trained on the other attack families with an
Isolation Forest trained on benign traffic. Recall is measured at a 1% false
positive rate, with the threshold selected from benign test scores. Results
are printed and written to `evaluation/results/leave_one_family_out.csv`.

Latest recorded results (recall values rounded to three decimals):

| Held-out family | Attack rows in test | XGBoost recall at 1% FPR | Isolation Forest recall at 1% FPR |
|---|---:|---:|---:|
| BruteForce | 13,832 | 0.289 | 0.000 |
| DoS | 251,711 | 0.684 | 0.271 |
| Web | 2,159 | 0.952 | 0.000 |
| Rare | 68 | 0.441 | 0.279 |
| PortScan | 158,804 | 0.011 | 0.000 |
| Bot | 1,956 | 0.015 | 0.000 |
| DDoS | 128,025 | 0.645 | 0.083 |

For the DDoS holdout, the exact recorded recall values are 0.6446475 for
XGBoost and 0.0825776 for Isolation Forest. These are family-holdout
evaluation results, distinct from the unseen-day binary results above.

## Charts and MLflow

Open `notebooks/02_leakage_demo.ipynb` and run its code cells. The first cell
loads the `intrusion_baseline` experiment, prints attack recall by split,
loads the family-holdout CSV, prints the family results, and draws both charts.
It saves the combined figure to
`evaluation/results/leakage_and_generalization.png`.

The notebook explicitly uses the repository-root `mlflow.db` SQLite tracking
database. Run MLflow-producing commands from the repository root as shown
above. If no baseline runs exist yet, execute the three binary training
commands before running the notebook. Older baseline runs that predate
`metrics.attack_recall` will not supply that chart metric; rerun the training
commands to log it.

## Tests

Run the current test suite from the repository root:

```powershell
python -m pytest
```

The existing tests cover the CICIDS2017 normalization/cleaning behavior.

## Current scope

The repository currently covers dataset preparation, exploratory analysis,
baseline modelling, and offline model evaluation. `ml/cluster_alerts.py` and
an alert-clustering workflow have not been added yet.
