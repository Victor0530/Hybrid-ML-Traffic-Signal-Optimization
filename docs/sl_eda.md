# Exploratory Data Analysis — Supervised-Learning Dataset

**Status:** complete for the M3 dataset. This report covers the
required EDA deliverable; model training and tuning are the next M3 step.

**Scope.** This is a descriptive analysis of the processed long-format dataset.
It checks the target, feature distributions, persistence, direction differences
and split shift before any model is fitted. It does not report model performance.

**Canonical dataset:** `data/processed/sl_dataset.csv` (38,400 rows, 20 runs).
**Reproduction:** `python -m src.analysis.explore_sl_dataset`
**Notebook wrapper:** `notebooks/01_eda.ipynb` (optional; calls the same script)
**Outputs:** `results/sl/eda/` (10 CSV/JSON tables and 8 PNG figures; gitignored).

The dataset design, target definition, feature sets, split boundaries, embargo
and audit are documented in
[`supervised_learning_dataset.md`](supervised_learning_dataset.md). This report
does not repeat that methodology.

---

## 1. What was analysed

The dataset has one row per `(run_id, simulation_time, direction)`. It covers
five scenarios (`low`, `medium`, `high`, `uneven`, `changing`) and four seeds
(42–45), with four approach directions (`north`, `south`, `east`, `west`).
Samples are every 5 s. The target is `vehicle_count` 300 s ahead
(`target_vehicle_count`).

The analysis uses the full F4 feature set (41 features). The five feature sets
are defined once in
[`supervised_learning_dataset.md`](supervised_learning_dataset.md#3-feature-sets);
they are not repeated here.

Checks performed:

- target distribution overall, by split, scenario, direction and phase;
- feature summary and feature–target Pearson correlations;
- within-run autocorrelation of `vehicle_count` at lags 1–60 samples
  (5–300 s);
- train-versus-validation and train-versus-test standardised mean differences
  (SMD) as descriptive effect sizes;
- demand time series for a representative run.

No rows were removed, imputed or resampled for this analysis. The dataset
already has zero missing cells and zero duplicate keys.

---

## 2. Dataset overview

| Fact | Value |
|---|---:|
| Modelling rows | 38,400 |
| Runs | 20 |
| Scenarios | 5 |
| Seeds | 4 |
| Directions per timestamp | 4 |
| Sampling interval | 5 s |
| Missing cells | 0 |
| Duplicate `(run_id, time, direction)` keys | 0 |
| Zero-vehicle rows | 0 |

The zero-vehicle result is expected: the simulation records every approach on
every sample, and the configured demand is high enough that an approach is not
empty for a whole 5 s sample in these runs. It does **not** mean the model will
never see a low count; the `low` scenario still has a target mean near 5.

---

## 3. Target distribution

### 3.1 By split

| Split | Rows | Mean | Std | Median | Max |
|---|---:|---:|---:|---:|---:|
| Train | 24,000 | 10.07 | 7.89 | 8 | 60 |
| Validation | 9,600 | 14.09 | 13.64 | 11 | 76 |
| Test | 4,800 | 11.14 | 13.45 | 8 | 76 |

Validation and test have much heavier right tails than train. This is not a
data-quality fault: it is caused by the scenario mix at each point in time,
especially the `changing` scenario described below. It means validation and test
MAE are not directly comparable to train MAE, and a single pooled target mean is
not representative of every split.

### 3.2 By scenario and split

| Scenario | Train mean | Validation mean | Test mean |
|---|---:|---:|---:|
| `low` | 4.74 | 4.84 | 4.68 |
| `medium` | 10.38 | 10.41 | 10.33 |
| `high` | 12.99 | 13.10 | 12.95 |
| `uneven` | 13.87 | 20.54 | 21.12 |
| `changing` | 8.37 | **21.56** | **6.62** |

The first three scenarios are stationary and their split means agree closely.
`uneven` and `changing` are not stationary in the same way:

- `uneven` keeps a fixed asymmetric demand split (north 800, south 250,
  east 250, west 600 vehicles/h), but some approaches are oversaturated. Queues
  and vehicle counts build over the run rather than settling immediately:
  north's target mean is 37.3 in train but 64.0 in validation and 66.7 in test,
  and the scenario's mean queue length rises from 5.0 to 9.0–10.6. The later
  splits are therefore genuinely harder on north, not merely a different random
  sample of the same stationary process.
- `changing` steps from 400 to 800 vehicles/h at 1800 s and back to 400 at
  2700 s. The validation block (2105–2700 s) sits entirely in the high-demand
  period, while the test block (3005–3300 s) sits in the low-demand period.
  This explains the validation mean of 21.56 versus the test mean of 6.62.

**Important interpretation.** The chronological split is deliberately testing
temporal generalisation, not a random sample of the same distribution. The
`changing` validation block is a high-demand stress test. Validation is still
the right split for model selection because it is earlier than test and never
seen during fitting, but validation and test scores must be reported separately
and should not be treated as interchangeable estimates of the same quantity.

### 3.3 By direction

| Direction | Rows | Mean | Std | Median | Max |
|---|---:|---:|---:|---:|---:|
| North | 9,600 | 17.58 | 17.33 | 11 | 76 |
| West | 9,600 | 9.86 | 5.09 | 9 | 36 |
| South | 9,600 | 8.86 | 5.73 | 7 | 36 |
| East | 9,600 | 8.54 | 5.43 | 7 | 35 |

North is the highest-volume and most variable approach, mainly because of the
`uneven` scenario's 800 vehicles/h north demand. A single pooled model is still
appropriate, but evaluation should report per-direction MAE/RMSE in addition to
the pooled score; otherwise a model that is good on north can hide poor
performance on east or south.

---

## 4. Feature–target relationships

Pearson correlation with the 300 s-ahead target, over all modelling rows:

| Feature | Set | Correlation |
|---|---|---:|
| `count_lag_12` (60 s ago) | F1 | **0.949** |
| `count_roll_mean_36` (3 min mean) | F1 | 0.922 |
| `count_roll_mean_60` (5 min mean) | F1 | 0.913 |
| `count_roll_mean_12` (1 min mean) | F1 | 0.902 |
| `vehicle_count` (current) | F0 | 0.863 |
| `occupancy` | F2 | 0.862 |
| `count_lag_6` (30 s ago) | F1 | 0.862 |
| `count_lag_36` (180 s ago) | F1 | 0.850 |
| `count_lag_1` (5 s ago) | F1 | 0.850 |
| `count_lag_3` (15 s ago) | F1 | 0.838 |
| `count_lag_60` (300 s ago) | F1 | 0.816 |
| `queue_roll_mean_12` | F2 | 0.799 |
| `waiting_roll_mean_12` | F2 | 0.717 |
| `queue_length` | F2 | 0.670 |
| `count_share` | F4 | 0.660 |

The dominant signal is count persistence: recent count history and short
rolling means are more informative than the instantaneous count alone. This is
why F1 is expected to produce the largest single jump over F0. Queue, waiting
time and occupancy add a second, correlated view of congestion; their
incremental value over counts is what the F2 ablation will test.

Correlations are descriptive and pooled across scenarios. They do not prove
that a feature causes the target, and they can change within a split or a
scenario. The full table, including per-split correlations, is in
`results/sl/eda/correlations.csv`.

---

## 5. Persistence and autocorrelation

Autocorrelation of `vehicle_count` within each run and direction (pairs never
cross a run boundary):

| Lag | All | North | South | East | West |
|---:|---:|---:|---:|---:|---:|
| 5 s | 0.989 | 0.996 | 0.969 | 0.967 | 0.957 |
| 30 s | 0.891 | 0.953 | 0.709 | 0.686 | 0.572 |
| 60 s | 0.887 | 0.950 | 0.697 | 0.678 | 0.571 |
| 180 s | 0.915 | 0.961 | 0.760 | 0.771 | 0.726 |
| 300 s | 0.787 | 0.903 | 0.442 | 0.437 | 0.290 |

Two points matter for modelling:

1. **Short-term persistence is very strong.** The 5 s autocorrelation is 0.989
   overall, so lagged counts are legitimately powerful predictors. The 300 s
   target is still correlated with current information (0.787 at lag 300), which
   is why even F0 has a non-trivial baseline.
2. **Dynamics differ by direction.** North remains highly predictable at
   300 s (0.903), while west is much less persistent (0.290). A pooled model must
   therefore include direction identity (the one-hot features already do) and
   should be evaluated per direction. It also suggests that a single global
   error target may be too coarse: errors on west/east may deserve separate
   discussion in the report.

The non-monotonic pattern around 60–180 s is consistent with signal-cycle and
demand effects; it is not a reason to add a hand-crafted periodic feature before
the F0–F4 ablation shows that such a feature is needed.

---

## 6. Signal phase

| Phase | Share of rows | Mean target | Std |
|---:|---:|---:|---:|
| 0 | 31.3% | 11.43 | 9.92 |
| 1 | 5.8% | 10.72 | 10.66 |
| 2 | 5.8% | 10.84 | 10.87 |
| 3 | 5.8% | 10.85 | 11.11 |
| 4 | 34.4% | 11.49 | 11.14 |
| 5 | 5.6% | 10.71 | 10.16 |
| 6 | 5.6% | 10.69 | 9.91 |
| 7 | 5.6% | 10.57 | 9.72 |

Phases 0 and 4 dominate because they are the long green phases in the fixed-time
cycle. Mean target is similar across phases, so phase alone is a weak
univariate predictor. This does **not** mean phase is useless: interactions
between phase, queue and direction may still help tree models. The F3 ablation
will answer that question; the EDA should not be used to pre-empt it.

---

## 7. Split shift

Standardised mean difference (SMD) versus train. Positive means the later split
has a larger mean; |SMD| above about 0.2 is a non-trivial descriptive shift.

| Column | Train mean | Validation mean | Test mean | SMD val | SMD test |
|---|---:|---:|---:|---:|---:|
| `count_total` | 37.71 | 57.62 | 45.02 | **0.871** | 0.320 |
| `queue_total` | 12.62 | 23.90 | 17.11 | **0.786** | 0.313 |
| `count_roll_std_60` | 2.50 | 2.79 | 2.96 | 0.293 | **0.473** |
| `speed_roll_mean_12` | 8.71 | 7.69 | 8.55 | **−0.459** | −0.073 |
| `count_roll_mean_12` | 9.40 | 14.28 | 11.26 | 0.435 | 0.166 |
| `queue_range` | 7.34 | 12.44 | 11.45 | 0.434 | 0.350 |
| `occupancy` | 0.048 | 0.074 | 0.058 | 0.434 | 0.158 |
| `vehicle_count` | 9.43 | 14.41 | 11.25 | 0.432 | 0.158 |
| `count_lag_1` | 9.42 | 14.37 | 11.25 | 0.430 | 0.159 |
| `count_lag_60` | 9.26 | 13.70 | 11.80 | 0.414 | 0.237 |

The largest shifts are exactly the variables that describe demand pressure:
total counts, total queue, occupancy and recent count means. Validation is the
most different from train, driven by the high-demand `changing` block noted in
Section 3.2. Test is closer to train on average but still shifted, and its
variability is higher.

Consequences for the modelling report:

- Do not describe the splits as IID samples. They are chronological blocks and
  the later blocks contain different demand regimes.
- Report validation and test metrics separately, and include per-scenario and
  per-direction breakdowns. A pooled test MAE alone would hide the shift.
- Keep the planned scenario-held-out / cross-distribution evaluation in the
  final comparison. It answers a different question from the chronological
  split: "does the model generalise to an unseen demand pattern?"
- When comparing models, use the same split and the same run-level bootstrap
  procedure. Do not select a model on test because its test mean happens to be
  closer to train.

---

## 8. What this means for the next M3 step

1. **Start with F0 and F1.** Count history is the strongest signal. If F1 does
   not beat F0 clearly, something is wrong with the lag/rolling construction.
2. **Use F2 to test congestion value.** Queue, waiting time and occupancy are
   correlated with the target but also with counts; only the ablation can show
   whether they add information.
3. **Treat F3 as an interaction test, not a guaranteed win.** Phase is weakly
   related to the target on its own.
4. **Keep direction one-hots and per-direction metrics.** North behaves very
   differently from west/east.
5. **Expect validation to be harder than test for `changing`.** That is the
   intended stress test, not a bug. Report it explicitly.
6. **Use run-level bootstrap confidence intervals.** With 20 runs, resampling
   rows would understate uncertainty and ignore within-run correlation.

---

## 9. Limitations

- This is a single four-way intersection with a fixed-time signal and synthetic
  SUMO demand; the conclusions are not automatically transferable to other
  junctions or real sensor data.
- Correlations are pooled across scenarios and time. They are useful for feature
  screening, not for causal claims.
- SMD is a descriptive effect size, not a formal drift test. No p-values are
  reported because the rows are autocorrelated and not independent.
- The EDA does not evaluate any model. Model accuracy, residuals and feature
  importance belong in the supervised-learning results.
- The `changing` scenario's high-demand validation block is a property of the
  chosen split. If the split changes, this analysis must be rerun.

---

## 10. Reproduction and outputs

```powershell
# Rebuild the dataset first if it is missing
python -m src.preprocessing.build_dataset

# Run the EDA
python -m src.analysis.explore_sl_dataset

# Optional: choose another run for the demand plot
python -m src.analysis.explore_sl_dataset --run-id changing_seed43
```

Tables:

```text
results/sl/eda/target_by_split.csv
results/sl/eda/target_by_scenario.csv
results/sl/eda/target_by_direction.csv
results/sl/eda/target_by_scenario_direction.csv
results/sl/eda/target_by_scenario_split.csv
results/sl/eda/feature_summary.csv
results/sl/eda/correlations.csv
results/sl/eda/autocorrelation.csv
results/sl/eda/split_shift.csv
results/sl/eda/target_by_phase.csv
results/sl/eda/eda_summary.json
```

Figures:

```text
results/sl/eda/target_distribution_by_split.png
results/sl/eda/target_by_scenario_split.png
results/sl/eda/target_by_direction.png
results/sl/eda/top_correlations.png
results/sl/eda/split_shift.png
results/sl/eda/demand_timeseries.png
results/sl/eda/count_lag_60_vs_target.png
results/sl/eda/target_by_phase.png
```

The EDA helpers are covered by `tests/test_sl_eda.py`. The script is
deterministic and does not modify the processed dataset.

---

## 11. Where to use this in the report

| Report section | Use |
|---|---|
| Data and methodology | Dataset shape, target, split table, EDA procedure. |
| Descriptive statistics | Target by split/scenario/direction, phase table. |
| Feature justification | Correlation and autocorrelation tables. |
| Threats to validity | Split shift, especially `changing` validation; single-intersection limitation. |
| Results setup | Per-scenario and per-direction reporting; run-level bootstrap. |
| Discussion | Why count history dominates; why direction differences matter. |
