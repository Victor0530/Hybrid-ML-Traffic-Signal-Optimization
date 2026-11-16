# Shared SL/RL contract v1

> **Status:** implemented and frozen for parallel development.
> Git tag: `contract-v1`.
>
> **Purpose:** give every supervised-learning (SL) and reinforcement-learning
> (RL) teammate the same data access, preprocessing, model interface, SUMO
> environment, action rules and evaluation metrics. Parallel work must build on
> these interfaces instead of creating competing versions.

This document is the API reference for the frozen contract. The dataset
methodology remains in
[`supervised_learning_dataset.md`](supervised_learning_dataset.md), the EDA
findings in [`sl_eda.md`](sl_eda.md), and the fixed-time baseline in
[`fixed_time_baseline.md`](fixed_time_baseline.md).

---

## 1. Frozen modules

| Module | Responsibility | Consumers |
|---|---|---|
| `src/supervised/data.py` | Load the frozen dataset and return the train/validation/test splits with a selected feature set. | SL lanes A and B |
| `src/supervised/pipeline.py` | Median imputation with missingness indicators, optional scaling, fitted on training data only. | SL lanes A and B |
| `src/supervised/registry.py` | Register and construct model candidates through one plugin interface. | SL lanes A and B |
| `src/supervised/baselines.py` | Persistence baseline (`future count = current count`). | All SL comparison tables |
| `src/supervised/predictor.py` | Fit, save, load and call a model through the `[N, S, E, W]` prediction API. | M5 hybrid controller |
| `src/evaluation/metrics.py` | One implementation of regression and traffic metrics. | SL, RL and final evaluation |
| `src/reinforcement/environment.py` | TraCI environment with `reset`, `step`, safe phase transitions and episode metrics. | RL lanes C and D |
| `src/reinforcement/controllers.py` | Fixed-time 90 s controller used as the non-RL baseline. | RL and final comparison |
| `configs/rl.yaml` | Environment, action, reward, training and evaluation settings. | RL lanes C and D |

Do **not** edit these files in a model lane. Add new model or agent modules and
register them through the existing interfaces. If a shared interface genuinely
needs to change, raise it for review and create a new contract version.

---

## 2. Supervised-learning contract

### 2.1 Loading the frozen data

```python
from src.supervised import load_splits

splits = load_splits(feature_set="F4")
train, validation, test = splits["train"], splits["validation"], splits["test"]

print(train.X.shape, train.y.shape)   # (24000, 41) (24000,)
print(validation.n_samples)            # 9600
print(test.n_samples)                  # 4800
```

`load_splits` reads `data/processed/sl_dataset.csv` and the feature map produced
by `src.preprocessing.build_dataset`. It validates that the requested split and
feature columns exist and rejects empty splits or undefined targets. Default
paths come from `configs/sl.yaml`; use `config=...` or `path=...` only for a
controlled experiment.

`train.X` is a `pandas.DataFrame` with stable feature names. Never reorder or
rename columns by hand; the fitted pipeline and the M5 predictor rely on this
order.

### 2.2 Preprocessing

```python
from src.supervised.pipeline import fit_preprocessor, build_model_pipeline

preprocessor = fit_preprocessor(train.X, scale=True)
pipeline = build_model_pipeline(estimator, scale=True)
```

The contract pipeline is:

1. `SimpleImputer(strategy="median", add_indicator=True)`
2. optional `StandardScaler`

Fit it on `train.X` only. Validation and test data are transformed with the
already-fitted object. The dataset builder deliberately performs **no**
imputation or scaling, so every model lane must use this train-only pipeline.

### 2.3 Model registry

```python
from src.supervised import build_model, available_models, register_model

print(available_models())
# ('ridge', 'decision_tree', 'random_forest', 'hist_gradient_boosting')

model = build_model("ridge")
```

Registered first-pass candidates:

| Name | Estimator | Default notes |
|---|---|---|
| `ridge` | `Ridge` | `alpha=1.0`, standardized features |
| `decision_tree` | `DecisionTreeRegressor` | `random_state=42` |
| `random_forest` | `RandomForestRegressor` | 100 trees, `random_state=42` |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor` | `random_state=42` |
| `persistence` | `PersistenceRegressor` | Baseline only; not an ML candidate |

Add a model without editing the registry module:

```python
from src.supervised import register_model

register_model(
    "my_model",
    factory=lambda params: MyRegressor(**params),
    default_params={"alpha": 1.0},
    scale=False,
)
```

### 2.4 Fitting and prediction

```python
from src.supervised import fit_named_predictor

predictor = fit_named_predictor(
    "ridge",
    train.X,
    train.y,
    metadata={"feature_set": "F4", "split": "train"},
)
predictions = predictor.predict_array(validation.X)
```

For the hybrid controller, pass one observation row per direction:

```python
future_counts = predictor.predict(window)  # {"north": ..., "south": ..., "east": ..., "west": ...}
```

`predict` requires exactly one row for each of `north`, `south`, `east` and
`west`. The returned values are the predicted `vehicle_count` 300 s ahead, in
vehicles per sampling interval. Save and load fitted objects with
`predictor.save(path)` and `Predictor.load(path)`; artifacts belong under
`models/sl/` and are gitignored.

### 2.5 SL evaluation rules

- Select and tune on **validation** only.
- Touch **test** once, after the final model is chosen.
- Report MAE, RMSE, R² and MAPE with `regression_metrics`.
- Report metrics overall, per direction, per scenario and per feature set.
- Use run-level bootstrap confidence intervals, not row-level resampling.
- Use `TimeSeriesSplit(gap=60)` for time-aware tuning; never shuffled K-fold.

---

## 3. RL contract

### 3.1 Environment usage

```python
from src.reinforcement import TrafficEnvironment

with TrafficEnvironment(scenario="medium", seed=42) as env:
    observation = env.reset()
    result = env.step(0)             # select a green phase
    print(result.reward, result.terminated, result.info)
    state = observation.as_vector("S0")
```

The constructor accepts `config`, `scenario`, `seed`, `gui`, `fixed_time`,
`duration` and `tls_id`. `duration` defaults to the simulation duration in
`configs/traffic_scenarios.yaml`. `reset` returns the first observation and
`step` returns a `StepResult`.

`TrafficEnvironment.run_episode(policy=None, max_steps=None)` is provided for
smoke tests and simple baselines. A policy implements `reset()` and
`act(observation) -> int | None`.

### 3.2 Actions and safety

The agent chooses only one of the four green phases:

| Action | Green phase | Meaning | Inserted yellow |
|---|---|---|---|
| `0` | 0 | North–south through | 1 |
| `1` | 2 | North–south left | 3 |
| `2` | 4 | East–west through | 5 |
| `3` | 6 | East–west left | 7 |

The environment inserts the matching yellow/clearance phase automatically. An
agent cannot select yellow. Additional rules:

- `None` keeps the current valid phase.
- A switch before `min_green` is ignored.
- `max_green` forces a switch to the next green phase.
- `fixed_time=True` runs the network's native static 90 s programme and ignores
  actions; this is the fixed-time baseline.
- Invalid action IDs raise `ValueError` rather than silently doing something
  unsafe.

The decision interval, minimum green, maximum green and yellow duration come
from `configs/rl.yaml` and are part of the contract.

### 3.3 Observation layout

Directions are always ordered **north, south, east, west**. Values are raw
units (vehicles, vehicles queued, m/s, seconds), not normalized.

| State | Size | Layout |
|---|---:|---|
| `S0` | 20 | indices 0–3 vehicle counts, 4–7 queue lengths, 8–11 mean speeds, 12–19 one-hot signal phase |
| `S1` | 21 | `S0` + index 20 `time_in_phase` |
| `S2` | 25 | `S1` + indices 21–24 four SL-predicted future counts in direction order |

`S2` requires the four predictions explicitly:

```python
state = observation.as_vector("S2", predictions=[north, south, east, west])
```

`Observation` also exposes `total_queue` and `total_waiting` for reward and
diagnostics. Waiting time is **not** in the state vector; it is part of the
reward.

### 3.4 Reward and episode information

```text
reward = -(alpha * total_queue + beta * total_waiting_time)
```

Defaults are `alpha=1.0` and `beta=0.1` in `configs/rl.yaml`. Reward-component
diagnostics must be reported so one term cannot dominate unnoticed.

`StepResult.info` includes the simulation time, phase, arrived count and
teleport information. Episode-level metrics are computed with the shared
function in `src/evaluation/metrics.py`, not with a lane-specific formula.

### 3.5 Fixed-time baseline

`src/reinforcement/controllers.py` defines the native 90 s plan:

```text
phase 0: 33 s   phase 1: 3 s   phase 2: 6 s   phase 3: 3 s
phase 4: 33 s   phase 5: 3 s   phase 6: 6 s   phase 7: 3 s
```

`FixedTimeController` exposes `reset()`, `act(observation)` and
`phase_for_time(seconds)`. It is the conventional baseline and must **not** be
counted as an RL candidate.

---

## 4. Shared metrics

### 4.1 Regression metrics

`regression_metrics(y_true, y_pred)` returns:

- `mae`
- `rmse`
- `r2` (`NaN` when the target is constant)
- `mape` (percentage, zero truths excluded)

### 4.2 Traffic metrics

`aggregate_traffic_metrics(step_records, duration_seconds, travel_times=None)`
returns:

- `mean_queue_total`, `max_queue_total`, `mean_queue_per_approach`
- `mean_waiting_total`, `max_waiting_total`, `mean_waiting_per_approach`
- `throughput_veh_per_h`
- `arrived_vehicles`, `completed_vehicles`
- `mean_travel_time_s`
- `episode_duration_s`

Queue and waiting are summed across the four approaches at each decision step,
then averaged over the episode. Throughput is `arrived / duration * 3600`.
Travel time is the mean of completed trips. These definitions are the same for
fixed-time, RL-only and hybrid comparisons.

---

## 5. Parallel work lanes

The contract is tagged only after the checks in §6 pass. From that point:

| Lane | Owner | Scope | Deliverables |
|---|---|---|---|
| A | SL teammate | Ridge and Decision Tree modules | model module, registry entry, config block, seeded reproduce command, validation metrics JSON, notes |
| B | SL teammate | Random Forest and HistGradientBoosting modules | same |
| C | RL teammate | Q-Learning and DQN agents | agent module, config block, seeded training/evaluation command, validation metrics JSON, notes |
| D | Project owner | Double DQN, fixed-time/actuated baseline evaluation, review and integration | baseline results, review notes, integration |

Rules for every lane:

- Do not edit shared contract files; add new modules through the registry or
  agent interface.
- Do not tune or run the final test evaluation in the parallel phase.
- Do not commit generated data, models or results.
- Provide one seeded command that reproduces the reported validation metrics.
- Run `pytest` before handing work back.
- Record assumptions and any deviation from this contract.

---

## 6. Verification and freeze

Run the full suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: **46 passed** (14 M1/M2, 9 dataset, 6 EDA, 17 contract tests).

Run the environment smoke tests from the repository root:

```powershell
.venv\Scripts\python.exe -m src.reinforcement.environment --scenario medium --seed 42 --duration 120 --policy fixed
.venv\Scripts\python.exe -m src.reinforcement.environment --scenario medium --seed 42 --duration 120 --policy alternate
```

The fixed-time smoke run should report approximately:

```text
mean_queue_total 7.875
mean_waiting_total 136.5
throughput_veh_per_h 570
mean_travel_time_s 81.0
```

The exact values can move with a SUMO version change; the run must complete and
produce a metrics JSON object.

Freeze checklist:

- [x] Dataset/split loader and feature ordering verified on the canonical data.
- [x] Train-only preprocessing pipeline tested.
- [x] Four SL candidates plus persistence baseline registered.
- [x] Prediction API returns `[N, S, E, W]` and saves/loads.
- [x] RL environment resets, steps and handles safe transitions.
- [x] Fixed-time controller matches the 90 s native plan.
- [x] Shared regression and traffic metrics tested.
- [x] `configs/rl.yaml` committed.
- [x] Full test suite passes.
- [x] Commit and tag `contract-v1`.
- [x] Push branch and tag to `origin`.

After the tag, work proceeds in parallel and this document is the reference for
all shared behavior.

## 7. Addendum: post-freeze additions

No frozen module (Section 1) or frozen config (`configs/rl.yaml`) was edited
after the `contract-v1` tag. The following files were added on top of the
frozen contract so the SL and RL lanes can train in parallel:

| Addition | Purpose |
| --- | --- |
| `src/utils/config.py`, `src/utils/reproducibility.py`, `src/utils/__init__.py` | Shared `load_yaml`/`deep_merge` config loading and `set_global_seed` reproducibility helpers. |
| `src/supervised/train.py`, `src/supervised/evaluate.py` | Shared SL harness: fit/tune registered candidates (`--tune` uses `TimeSeriesSplit` with a 300 s gap), write metric JSON/`summary.csv`, save/load predictors, evaluate a split. |
| `src/reinforcement/agents.py` | Agent registry (`register_agent`, `make_agent`, `available_agents`). Ships **empty**: each lane registers its own agent module. |
| `src/reinforcement/train.py`, `src/reinforcement/evaluate.py` | Shared RL harness: train a registered agent with episode logging + checkpoints, evaluate fixed-time or a learned checkpoint across scenarios/seeds. |
| `configs/rl_agents.yaml` | Per-agent hyperparameters; merged over `configs/rl.yaml` training defaults (rl.yaml stays frozen). Ships with only the shared `training` block. |
| `configs/sl.yaml` `models:` block | Per-candidate params/grids for the SL harness (sl.yaml was never frozen). |
| `tests/test_shared_harnesses.py` | Tests covering the helpers, both harnesses and the agent registry (suite now 56 tests). |

**Post-freeze fix to a frozen module:** `PersistenceRegressor` in
`src/supervised/baselines.py` did not inherit from `sklearn.base.BaseEstimator`,
so under scikit-learn 1.9 `Pipeline.predict(...)` raised
`AttributeError: ... has no attribute '__sklearn_tags__'` when the persistence
baseline was the final estimator. The fix (approved before any commit or
teammate branch existed) makes the estimator satisfy the scikit-learn protocol:

1. inherit `RegressorMixin, BaseEstimator`;
2. rename the fitted attribute `_column_index` → `column_index_`, because
   `check_is_fitted` only recognises trailing-underscore names;
3. add a regression test that predicts through a fitted pipeline.

The temporary `predict_pipeline` workaround was removed; harness code now calls
`pipeline.predict(X)` directly. Because the edit landed before any lane
branched from `contract-v1`, no re-tag or version bump was needed.
