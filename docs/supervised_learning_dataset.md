# Supervised-Learning Dataset and Feature Engineering

Status: dataset and EDA complete. Model training is the next
step.

> **Single source of truth.** This report owns the M3 dataset methodology, row
> accounting, feature definitions, split rationale and audit interpretation.
> `RUNBOOK.md` only lists commands/outputs and `FILES.md` only maps paths; both
> link here instead of repeating the details.

This document records how the Milestone 3 supervised-learning dataset was built,
why each decision was made, and what the resulting numbers mean. It is the
evidence base for the dataset, target, feature-engineering and split sections of
the final report.

Related documents:

- [`PLAN.md`](../PLAN.md) - assignment specification, especially section 7.
- [`FILES.md`](../FILES.md) - repository map and milestone status.
- [`RUNBOOK.md`](../RUNBOOK.md) - exact commands and expected console output.
- [`docs/fixed_time_baseline.md`](fixed_time_baseline.md) - M2 data definitions
  and the fixed-time reference controller.

## 1. Scope and data source

The dataset is built from the canonical M2 simulation runs:

| Property | Value |
|---|---:|
| Scenarios | 5 (`low`, `medium`, `high`, `uneven`, `changing`) |
| Seeds per scenario | 4 (42, 43, 44, 45) |
| Runs | 20 |
| Simulated duration | 3,600 s per run |
| Sampling interval | 5 s |
| Timestamps per run | 720 |
| Approaches per timestamp | 4 |
| Raw observation rows | 57,600 |

The loader reads `data/raw/run_manifest.csv`, keeps only runs with `status=ok`
and the configured duration/sampling interval, and refuses to continue if a
listed CSV or a required observation column is missing. This prevents stale,
partial or mixed datasets from silently entering the model.

### Long format and one pooled model

Each modelling row is one **`(run_id, simulation_time, direction)`** triple.
The four approaches therefore remain separate rows rather than becoming four
target columns. This is the **pooled long** design:

- one model has 24,000 training rows instead of four per-direction models with
  6,000 each (each direction contributes 9,600 rows across all three splits);
- the approach identity is supplied by four direction indicator columns;
- cross-approach pressure is supplied by the F4 aggregate features;
- predictions can be reassembled into `[north, south, east, west]` by a thin
  inference wrapper for the hybrid stage.

The alternative layouts were rejected for this phase. Four per-direction models
would quadruple tuning and artefacts for a symmetric intersection. A wide
four-target table would have only 9,600 total rows and complicate per-direction
tuning and error reporting.

## 2. Prediction target

The target is the vehicle count on the **same approach** exactly 300 s ahead:

```python
target_vehicle_count = (
    data.sort_values(["run_id", "direction", "simulation_time"])
        .groupby(["run_id", "direction"])["vehicle_count"]
        .shift(-60)
)
```

`60` is the horizon in samples: `300 s / 5 s`.

Rules:

- The shift is computed **within each run and direction**, never across them.
- The target is never imputed, clipped or filled forward. A row without a real
  future value is removed from the modelling set.
- Because the simulation ends at 3,600 s, the last 60 samples per run and
  direction have no target. The last valid input is 3,300 s, whose target is
  exactly 3,600 s.
- The target is not part of any feature set.

Train-split target statistics: mean **10.072**, standard deviation **7.885**
vehicles. This is the scale against which MAE and RMSE should be interpreted.

## 3. Feature sets

Five cumulative sets are defined so the report can isolate the value of each
information source. F0 is the baseline; every later set contains all earlier
columns plus its own.

| Set | Columns | Added information |
|---|---:|---|
| F0 | 7 | Current counts, status flags, approach identity |
| F1 | 19 | Count history: lags and rolling statistics |
| F2 | 26 | Congestion: queue, speed, waiting time, occupancy |
| F3 | 36 | Signal state: phase, elapsed time, programmed duration |
| F4 | 41 | Cross-approach demand and queue imbalance |

The exact ordered column lists are written to
`data/processed/feature_columns.json`; training code must select columns from
that file rather than hard-coding them.

### F0 - current state (7 columns)

| Column | Definition |
|---|---|
| `vehicle_count` | Vehicles currently on the approach |
| `no_vehicles` | 1 when the approach is empty, else 0 |
| `sensor_missing` | 1 when detector data is unavailable, else 0 |
| `direction_north` | 1 for north rows, else 0 |
| `direction_south` | 1 for south rows, else 0 |
| `direction_east` | 1 for east rows, else 0 |
| `direction_west` | 1 for west rows, else 0 |

The direction columns are one-hot because one pooled model serves all four
approaches. Exactly one is 1 on every row.

### F1 - count history (12 additional columns)

| Column | Definition |
|---|---|
| `count_lag_1`, `count_lag_3`, `count_lag_6` | Count 5 s, 15 s, 30 s ago |
| `count_lag_12`, `count_lag_36`, `count_lag_60` | Count 60 s, 180 s, 300 s ago |
| `count_roll_mean_12` | Mean count over the last 60 s |
| `count_roll_mean_36` | Mean count over the last 180 s |
| `count_roll_mean_60` | Mean count over the last 300 s |
| `count_roll_std_12` | Standard deviation over the last 60 s |
| `count_roll_std_36` | Standard deviation over the last 180 s |
| `count_roll_std_60` | Standard deviation over the last 300 s |

All lags and rolling windows operate within `(run_id, direction)` and end at the
current sample. A lag of `k` samples means `k * 5` seconds.

### F2 - congestion (7 additional columns)

| Column | Definition |
|---|---|
| `queue_length` | Halting vehicles on the approach |
| `mean_speed` | Mean speed in m/s; may be empty when no vehicles are present |
| `waiting_time` | Sum of current waiting times in seconds, not a per-vehicle mean |
| `occupancy` | Detector occupancy as a fraction from 0 to 1 |
| `queue_roll_mean_12` | Queue mean over the last 60 s |
| `speed_roll_mean_12` | Speed mean over the last 60 s |
| `waiting_roll_mean_12` | Waiting-time mean over the last 60 s |

Speed rolling uses `min_periods=1` so a single empty approach does not erase a
minute of speed history. The raw `mean_speed` cell itself remains empty in that
case; imputation is a training-pipeline responsibility.

### F3 - signal state (10 additional columns)

| Column | Definition |
|---|---|
| `phase_0` ... `phase_7` | One-hot indicator for the active TLS phase |
| `time_in_phase` | Seconds elapsed in the current phase |
| `phase_duration` | Programmed duration of the current phase |

The intersection uses the eight-phase fixed-time program documented in
`docs/fixed_time_baseline.md`. Exactly one `phase_*` column is 1 per row.

### F4 - cross-approach context (5 additional columns)

These features describe the other three approaches at the **same timestamp**,
without looking into the future:

| Column | Definition |
|---|---|
| `count_total` | Sum of `vehicle_count` over all four approaches |
| `count_range` | Busiest minus quietest approach count |
| `count_share` | This approach's share of `count_total`; 0 when total is 0 |
| `queue_total` | Sum of `queue_length` over all four approaches |
| `queue_range` | Busiest minus quietest approach queue |

This is the information that lets one pooled model distinguish "8 vehicles and
the whole junction is busy" from "8 vehicles and this approach is the only busy
one."

### Deliberate exclusions

- `departed_count` and `arrived_count` remain throughput diagnostics, not
  features, because they describe a completed interval and are not part of the
  controller's observation.
- No target-derived column is available as an input.
- No future window, centred rolling statistic, or full-run aggregate is used.
- No scaling or imputation is applied at build time; both must be fitted on the
  training split only.

## 4. Chronological splits and embargo

### Timeline

```text
seconds
0        300        1800 1805      2100 2105      2700 2705      3000 3005      3300 3305   3600
| warm-up |  train ... |  embargo   | validation |  embargo   |  test ...  |  target tail |
```

### Split table

| Split | Simulation time | Samples per run | Rows | Purpose |
|---|---|---:|---:|---|
| train | 305-1800 s | 300 | 24,000 | Fit parameters and tune with `TimeSeriesSplit` |
| validation | 2105-2700 s | 120 | 9,600 | Rank models, choose feature set, select blend |
| test | 3005-3300 s | 60 | 4,800 | One final unbiased estimate |

`305`, `2105` and `3005` are the first 5 s samples **after** each 300 s gap.
The test block ends at `3,600 - 300 = 3,300 s`, the last input with a real
target.

### Why the embargo exists

Five-second samples share a 300 s target window. Without a gap, a training row
at 1,800 s would predict the count at 2,100 s, while the first validation row at
2,105 s would use almost the same moment as its input. Training would therefore
see part of the validation period, and the reported validation score would be
optimistic.

The builder enforces the stronger condition:

```text
last training target time < first validation input time
last validation target time < first test input time
```

With the 300 s embargo this holds with one sampling step to spare. The same rule
is re-checked in the tests, not merely assumed.

This is **temporal generalisation**: every run contributes to every split, and
the model is tested on a later part of the same runs. It does not by itself prove
generalisation to a new demand scenario; that is the separate cross-distribution
analysis required by PLAN section 7.11.

### Tuning protocol

- Hyperparameters are tuned with `TimeSeriesSplit(gap=60)` inside the training
  split, so no tuning fold shares a target with its validation fold.
- The independent validation block ranks the final candidates and decides
  whether a blend beats the best single model.
- The test block is evaluated once, after all choices are frozen.

## 5. Row accounting

| Stage | Rows | Explanation |
|---|---:|---|
| Raw observations | 57,600 | 20 runs x 720 timestamps x 4 approaches |
| After removing undefined targets | 52,800 | Last 60 samples per run/direction removed |
| After removing warm-up | 48,000 | First 60 samples per run/direction removed |
| After removing embargo | 38,400 | Two 60-sample gaps x 4 approaches x 20 runs |
| Train | 24,000 | 300 samples x 4 x 20 |
| Validation | 9,600 | 120 samples x 4 x 20 |
| Test | 4,800 | 60 samples x 4 x 20 |

The three exclusions are disjoint and serve different purposes:

- **Target tail (4,800):** no real future value exists. Keeping these rows
  would require fabricating a label.
- **Warm-up (4,800):** the 300 s lag and rolling features need 300 s of history.
  Dropping the same warm-up for every feature set keeps the F0-F4 ablation on
  identical rows and therefore fair.
- **Embargo (9,600):** protects the validation and test blocks from target
  overlap, as explained in section 4.

The cost is real: 19,200 of 57,600 rows (33%) are not used for fitting or
scoring. This is largely because a one-hour simulation is short relative to a
300 s horizon, 300 s warm-up and two 300 s gaps. Longer simulations or more runs
would reduce the fraction without weakening the protocol.

## 6. Data-quality audit

The builder writes `data/processed/data_audit.json`. On the canonical dataset:

| Check | Result |
|---|---:|
| Runs | 20 |
| Scenarios | 5 |
| Duplicate `(run_id, simulation_time, direction)` keys | 0 |
| Missing values in retained rows | 0 |
| Timestamp continuity failures | 0 |
| Zero-vehicle rows | 0 |
| Runs represented in each split | 20 |

A zero here means the simulation data is clean, not that validation is disabled.
The builder raises a `ValueError` for:

- missing required observation columns;
- duplicate keys;
- a group that does not contain exactly 720 samples;
- non-contiguous 5 s timestamps;
- a timestamp without all four directions;
- unknown direction labels;
- a duration that is not a multiple of the sampling interval;
- split boundaries that overlap, are misordered, or use an embargo shorter than
  the prediction horizon.

`mean_speed` is nullable by design: it is blank when an approach has no vehicles
or the detector is missing. The `no_vehicles` and `sensor_missing` flags preserve
that reason. This canonical dataset happens to contain no empty approaches, so
the nullable branch is implemented but not exercised. A controlled sensor-loss
test belongs with the model evaluation and must alter validation/test features
only, never fabricate dirty training data.

## 7. Reproduction and determinism

From the repository root, with the virtual environment active:

```powershell
python -m src.preprocessing.build_dataset
python -m pytest -q
```

Expected console output:

```text
Runs loaded:      20
Raw rows:         57,600
Dataset rows:     38,400
Target dropped:   4,800
Warm-up dropped:  4,800
Embargo excluded: 9,600
Splits:           train=24,000, validation=9,600, test=4,800
Feature counts:   F0=7, F1=19, F2=26, F3=36, F4=41
```

Outputs:

```text
data/processed/sl_dataset.csv
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv
data/processed/feature_columns.json
data/processed/data_audit.json
```

All are gitignored and rebuildable. The builder uses stable sorting and no
random sampling; two consecutive rebuilds produced byte-identical datasets
(verified by SHA-256). Test results: the full suite passes (**61** at last
count), of which 9 are the dataset tests listed in §8 below.

## 8. Verification tests

| Test | What it proves |
|---|---|
| Target alignment | The target at time `t` equals the count at `t + 300 s` |
| Last horizon samples | No target exists after 3,300 s |
| Duplicate rejection | Duplicate keys cannot enter the dataset |
| Gap rejection | Missing timestamps fail loudly |
| Feature-set monotonicity | F0 subset F1 subset F2 subset F3 subset F4, no duplicate columns |
| Future leakage | Changing an observation at 1,200 s leaves every feature at 1,195 s unchanged |
| Split chronology | Expected row counts, embargo respected, no row shared between splits |
| Short embargo | An embargo below the 300 s horizon is rejected |
| End-to-end build | Files, split counts and the audit are produced from raw CSVs |

## 9. Exploratory model-readiness check

As a sanity check only, an untuned Ridge regression (median imputer,
standardiser, `alpha=1.0`) was fitted on the training split and scored on
validation:

| Input | Validation MAE | Validation R² |
|---|---:|---:|
| Predict training mean | 8.16 | 0.000 |
| F0 | 4.86 | 0.767 |
| F4 | 2.80 | 0.875 |

Interpretation:

- Even current counts (F0) are predictive because traffic is persistent.
- F4 lowers MAE by about 42% relative to F0, so history, congestion, signal
  state and cross-approach context carry real signal.
- The remaining error is consistent with stochastic arrivals and a 5-minute
  horizon; R² is not expected to reach 1.

This is **not** a tuned result and must not be cited as the M3 model performance.
It only demonstrates that the dataset and split are model-ready.

## 10. Limitations to state in the report

- **One intersection, one network.** Conclusions do not automatically transfer
  to networks with interacting junctions.
- **Temporal, not demand, generalisation.** The headline split is chronological
  within every run. New-scenario generalisation needs the separate held-out
  scenario/seed analysis.
- **Short runs.** A 3,600 s run makes warm-up and embargo overhead large; 33% of
  raw rows are unused.
- **Clean simulation data.** Zero missing values is a property of SUMO, not
  evidence that the pipeline handles real detector loss; the nullable and
  sensor-missing paths still need an explicit stress test.
- **Pooled dynamics.** One model assumes the four approaches share dynamics.
  Direction indicators and F4 features mitigate this, but per-direction error
  must still be reported.
- **Aggregate measurements.** `vehicle_count` is an approach count, not a
  turning-movement count, so the model cannot resolve left/through/right demand.
- **No preprocessing fitted at build time.** Imputation and scaling are
  deliberately deferred so they can be fitted on training data only.

## 11. Where this feeds the report

| Report topic | Evidence in this document | Status |
|---|---|---|
| Dataset construction and quality | Sections 1, 5, 6 | Complete |
| Five-minute target definition | Section 2 | Complete |
| Feature engineering and F0-F4 rationale | Section 3 | Complete |
| Chronological split and leakage control | Section 4 | Complete |
| Reproducibility | Section 7 | Complete |
| Verification | Section 8 | Complete |
| Model comparison, tuning and ablation | Not in this document | Next milestone |
| Cross-distribution robustness | Not in this document | Next milestone |
| Bootstrap confidence intervals | Not in this document | Next milestone |

The required EDA is now complete: `src/analysis/explore_sl_dataset.py` produces
the tables and figures, `notebooks/01_eda.ipynb` is a thin optional wrapper, and
[`sl_eda.md`](sl_eda.md) records the findings. The next M3 step is
`src/supervised/`: train Ridge, Decision Tree, Random Forest and
HistGradientBoosting, tune the top two, evaluate the blend, run the F0-F4
ablation, and export the frozen model plus the `[north, south, east, west]`
inference wrapper.

For the planned parallel sprint, the remaining shared foundation item is the M4
environment, fixed-time controller and metric contract. Once that contract is
tested and tagged, the SL models and RL agents can be built in parallel without
each contributor inventing a different state, action or metric definition.
