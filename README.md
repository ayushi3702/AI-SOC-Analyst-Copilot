# AI-SOC-Analyst-Copilot

A smart assistant for security teams: it filters false alarms, investigates real
threats like a junior analyst, and writes the report for a human to approve.

This repository currently contains the CICIDS2017 data-cleaning and DuckDB
ingestion pipeline, exploratory notebooks, binary/family generalization
experiments, reusable binary model inference, and deterministic alert clustering.

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
  train_model.py                 Train and persist a validation-calibrated detector
  compare_models.py              Shared-split XGBoost vs neural-network comparison
  robustness_report.py           Offline fixed-threshold feature-noise stress test
  model_artifact.py               Versioned model artifact save/load helpers
  infer.py                        Score DuckDB events with a saved detector
  coverage_report.py             Family/label coverage by train/validation/test
  cluster_alerts.py               Deterministic entity/time alert clustering
  evaluate_incidents.py           Held-out event and incident evaluation
  leave_one_family_out.py        XGBoost vs Isolation Forest evaluation
  evaluate.py                    Validation-threshold alert-reduction helpers
notebooks/
  01_explore_cicids.ipynb        Dataset cleaning and exploratory analysis
  02_leakage_demo.ipynb          Split leakage and family-generalization charts
evaluation/results/
  leave_one_family_out.csv       Per-family evaluation results (generated)
  unseen_robustness.json         Fixed-threshold robustness report (generated)
  leakage_and_generalization.png Notebook chart output (generated)
evaluation/baselines/
  cicids2017_unseen_xgb.json     Frozen unseen-day XGBoost metrics and provenance
  cicids2017_unseen_coverage.json Frozen split coverage snapshot
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

### Split coverage report

Report raw labels and mapped attack-family counts in the model-fit,
validation, and held-out partitions. This uses the same unseen-day split and
block-based validation split as model training, and forces the full dataset
even if `sample_frac` is lowered for development:

```powershell
python -m ml.coverage_report --split unseen
```

The CSV makes it clear which attack families have zero training examples and
lists any raw labels that have not been mapped in `configs/settings.yaml`.
Use `--split block` to inspect the time-block holdout instead. This report
describes data coverage; it does not claim that a family was detected or that
the evaluation split is representative of real-world traffic.

The current full-data unseen-day report and XGBoost result are preserved as
small versioned snapshots in `evaluation/baselines/`. The trained joblib model
and data remain local/ignored; the baseline JSON records the local artifact's
SHA-256 so it can be identified without committing a serialized model. Treat
these Thursday-Friday metrics as a frozen comparison, not a tuning target.

#### Neural-network comparison

Compare the saved, frozen XGBoost artifact to a small feed-forward neural
network using the same fit/validation/test rows and feature list. XGBoost keeps
its original validation-calibrated threshold; the MLP gets its own threshold
from the shared validation partition. Both use class-balanced training weights:

```powershell
python -m ml.compare_models --split unseen
```

The default XGBoost artifact must match the SHA-256 recorded in the frozen
baseline snapshot; comparison stops if it does not. This avoids treating a
non-identical XGBoost retrain as the preserved baseline. The MLP standardizes
numeric features using statistics fit only on the fit partition, sorts fit rows
by stable event ID, and limits BLAS work to one thread during fitting to improve
repeatability. The report
includes recall, FPR, precision, PR-AUC, ROC-AUC, alert volume, per-family
recall, and observed validation FPR for each model. The MLP currently uses a
fixed 25-epoch budget and may not converge within that budget; its result is
an initial comparison, not a tuned neural-network result. The Thursday-Friday
test set remains a one-time comparison set. The JSON report is saved separately
from the frozen baseline, and the MLP artifact is written under ignored
`ml/models/`. This is an experimental comparison, not an expectation that a
neural network will outperform XGBoost on tabular flow data.

#### Controlled robustness report

Measure the saved unseen-day detector's sensitivity to fixed levels of
non-adaptive, zero-mean numeric-feature noise:

```powershell
python -m ml.robustness_report --split unseen
```

The script uses fit-partition feature scales, deterministic random noise, and
the saved model threshold; it does not retune on test data. The report is
written to `evaluation/results/unseen_robustness.json`. This synthetic stress
test is not a valid-traffic or adversarial-evasion simulation and should not be
interpreted as production robustness.

### Leave-one-family-out comparison

Run the family generalization experiment:

```powershell
python -m ml.leave_one_family_out
```

For each non-benign family, the split excludes that family from training and
tests on its rows plus benign rows from held-out time blocks. It compares a
supervised XGBoost classifier trained on the other attack families with an
Isolation Forest trained on benign traffic. The remaining training blocks are
split again by complete time blocks into model-training and validation sets.
Each model's threshold is selected using validation benign scores at a target
1% false-positive rate, then held fixed while evaluating the test set. Results
are printed and written to `evaluation/results/leave_one_family_out.csv`.
The reported test-set FPR may differ from 1%; the target is calibrated on
validation data, not tuned against test labels or scores.

Latest validation-calibrated results (recall values rounded to three decimals).
The test FPR columns are reported separately because validation calibration
does not guarantee a 1% FPR on held-out test data:

| Held-out family | Attack rows in test | XGBoost recall (validation target FPR 1%) | XGBoost test FPR | Isolation Forest recall (validation target FPR 1%) | Isolation Forest test FPR |
|---|---:|---:|---:|---:|---:|
| BruteForce | 13,832 | 0.156 | 0.007 | 0.000 | 0.010 |
| DoS | 251,711 | 0.669 | 0.007 | 0.383 | 0.010 |
| Rare | 68 | 0.426 | 0.007 | 0.279 | 0.010 |
| Web | 2,159 | 0.948 | 0.007 | 0.000 | 0.010 |
| Bot | 1,956 | 0.017 | 0.007 | 0.000 | 0.010 |
| PortScan | 158,804 | 0.011 | 0.010 | 0.000 | 0.010 |
| DDoS | 128,025 | 0.645 | 0.007 | 0.099 | 0.010 |

Rerun the command above to produce validation-calibrated results. The
CSV contains the full-precision metrics, including measured test FPRs.

### Reusable detector and incident clustering

Train a reusable binary model. Training rows are split by complete time blocks
into model-fit and validation portions. The threshold is calibrated from
validation benign scores; the selected split's test data is used only for
metrics and is never used to fit the model or threshold:

```powershell
python -m ml.train_model --split block
```

Use `--split unseen` for the Monday-Wednesday train / Thursday-Friday test.
`--with-dst-port` runs a destination-port feature ablation; compare it to the
default feature set on the unseen-day holdout, especially for PortScan and Bot.
The ablation also reports held-out attack recall by family. It is an
investigation aid, not evidence that destination port is safe from dataset
artifacts or operational shortcuts.
Artifacts are written to `ml/models/intrusion_binary.joblib` by default and
contain the estimator, exact feature list, calibrated threshold, target FPR,
validation fraction, split, and seed. The `ml/models/` directory is ignored by
Git. Only load artifacts you trust: joblib files use Python serialization.

Score the events in the read-only DuckDB store and write a Parquet file:

```powershell
python -m ml.infer
```

This writes event IDs, timestamps, source/destination IPs, model scores, and
the alert decision using the saved validation threshold. Override the artifact,
database, or output path with `--model`, `--db`, or `--output`.

Cluster the scored alerts by source/destination pair with a five-minute
maximum gap between consecutive alerts:

```powershell
python -m ml.cluster_alerts `
  --input data/processed/scored_events_unseen.parquet `
  --output data/processed/incidents_unseen_pair_gap5.parquet `
  --entity-columns src_ip dst_ip `
  --window-minutes 30 `
  --max-gap-minutes 5
```

`--entity-columns` accepts one or more event columns. `--max-gap-minutes`
starts a new incident when consecutive alerts are farther apart than that gap;
it cannot exceed the maximum span set by `--window-minutes`. Each incident is a deterministic,
non-overlapping group whose events share an entity and fit within the time
window measured from that incident's first event (the window boundary is
inclusive):

```powershell
python -m ml.cluster_alerts --input data/processed/scored_events.parquet
```

Use `--entity-columns dst_ip` to correlate on destination IP, or
`--window-minutes 60` to change the maximum incident span. Incident records
retain the ordered source `event_ids`, first/last timestamps, alert count, and
maximum score. Events at or below the saved model threshold are not clustered.
The function rejects invalid event IDs, timestamps, scores, and entity values
instead of silently dropping alerts.

Evaluate the unseen-model output offline against Thursday-Friday labels. This
command joins labels only after scoring and restricts events to configured
`test_days` before re-clustering, so incidents cannot span into training days:

```powershell
python -m ml.evaluate_incidents `
  --scores data/processed/scored_events_unseen.parquet `
  --output-dir data/processed/evaluation_unseen_pair_gap5 `
  --entity-columns src_ip dst_ip `
  --window-minutes 30 `
  --max-gap-minutes 5
```

The output directory contains event metrics, per-family recall, held-out
incidents with label-derived attack/benign counts, incident metrics, and a
sample of larger attack-containing incidents. The labels are for offline
evaluation only and are not inputs to inference or clustering.

#### CICIDS2017 unseen-day clustering and feature ablation

On the current unseen-day scores, changing from source-IP/30-minute grouping
to source-destination/30-minute grouping with a five-minute maximum event gap
kept event metrics unchanged, removed mixed attack/benign incidents, and
increased the incident count from 1,216 to 3,016. Median incident size stayed
at one alert and the largest DDoS incident remained about 71k alerts. With a
one-minute maximum span/gap, the largest incident fell to 7,808 alerts and
attack-containing incidents increased to 59; with a six-second limit, the
largest was 5,084 across 3,582 incidents. CICIDS2017 timestamps are at
minute-level resolution, so a sub-minute gap cannot split events sharing the
same recorded timestamp. These figures show that paired entities and short
windows reduce some over-grouping but do not fully solve burst-sized incidents.

The destination-port ablation on the same unseen-day holdout changed event
metrics as follows:

| Feature set | Recall | Test FPR | PR-AUC | Bot recall | PortScan recall | DDoS recall |
|---|---:|---:|---:|---:|---:|---:|
| Default (no destination port) | 0.254 | 0.0131 | 0.632 | 0.000 | 0.0053 | 0.554 |
| Include destination port | 0.295 | 0.0079 | 0.559 | 0.000 | 0.0144 | 0.636 |

Destination port improves the measured recall and FPR at the calibrated
threshold, but PR-AUC falls and Bot remains undetected. Keep this as an
ablation, not the new default. These test results are for analysis only; do not
select settings based on repeated tuning to the same held-out days.

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

The tests cover normalization, split isolation, model artifact persistence,
inference, clustering, held-out evaluation, and a synthetic end-to-end check
from DuckDB through saved-model scoring and incident evaluation.

## Current scope

The repository currently covers dataset preparation, exploratory analysis,
baseline modelling, validation-calibrated model inference, deterministic alert
clustering, and offline held-out incident evaluation. The investigator agents,
approval workflow, and analyst UI remain future work.
