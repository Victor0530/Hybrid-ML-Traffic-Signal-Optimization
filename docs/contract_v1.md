# Shared SL/RL contract v1

> **Status:** implemented and frozen for parallel development; amended by the
> contract-v2 review fixes listed in §9.
> Git tags: `contract-v1` (original freeze), `contract-v2` (review fixes).
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
| `src/supervised/data.py` | Load the frozen dataset and return the train/validation/test splits with a selected feature set. | SL lanes A, B and C |
| `src/supervised/pipeline.py` | Median imputation with missingness indicators, optional scaling, fitted on training data only. | SL lanes A, B and C |
| `src/supervised/registry.py` | Register and construct model candidates through one plugin interface. | SL lanes A, B and C |
| `src/supervised/baselines.py` | Persistence baseline (`future count = current count`). | All SL comparison tables |
| `src/supervised/predictor.py` | Fit, save, load and call a model through the `[N, S, E, W]` prediction API. | M5 hybrid controller |
| `src/evaluation/metrics.py` | One implementation of regression and traffic metrics. | SL, RL and final evaluation |
| `src/reinforcement/environment.py` | TraCI environment with `reset`, `step`, safe phase transitions and episode metrics. | RL lanes A and D |
| `src/reinforcement/controllers.py` | Fixed-time 90 s controller used as the non-RL baseline. | RL and final comparison |
| `configs/rl.yaml` | Environment, action, reward, training and evaluation settings. | RL lanes A and D |

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
# ('decision_tree', 'hist_gradient_boosting', 'random_forest', 'ridge', 'xgboost')

model = build_model("ridge")
```

Registered first-pass candidates:

| Name | Estimator | Default notes |
|---|---|---|
| `ridge` | `Ridge` | `alpha=1.0`, standardized features |
| `decision_tree` | `DecisionTreeRegressor` | `random_state=42` |
| `random_forest` | `RandomForestRegressor` | 100 trees, `random_state=42` |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor` | `random_state=42` |
| `xgboost` | `XGBRegressor` | `random_state=42`, `n_jobs=-1`; fifth candidate added post-freeze (§7) |
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

The action **is** the green phase ID. `env.step()` accepts only the four green
phases; `0`–`3` action indices are not used:

| Action | Meaning |
|---|---|
| `0` | North–south through |
| `2` | North–south left |
| `4` | East–west through |
| `6` | East–west left |

The environment never jumps straight from one green to another. It first runs
the **clearance path** for the requested switch (contract-v2 fix, §9), so
permissive left-turners are never trapped by cross traffic:

| from \ to | 0 | 2 | 4 | 6 |
|---|---|---|---|---|
| **0** | — | 1 | 1, 2, 3 | 1, 2, 3 |
| **2** | 3 | — | 3 | 3 |
| **4** | 5, 6, 7 | 5, 6, 7 | — | 5 |
| **6** | 7 | 7 | 7 | — |

Each number is a phase held in sequence before the target green starts:
yellow phases run for `yellow_seconds` (3 s), protected-left green phases run
for `left_clearance_seconds` (6 s). Example: leaving phase 0 for phase 4 walks
`1 → 2 → 3` (yellow, protected NS-left flush, left yellow) and only then
applies phase 4 — the same route the native SUMO programme uses. If a decision
deadline interrupts the walk, the pending target is remembered and the walk
resumes on the next step.

An agent cannot select yellow or any clearance phase directly. Additional
rules:

- `None` keeps the current valid phase (or resumes an interrupted walk).
- A switch before `min_green` is ignored while the current green is stable.
- `max_green` forces a switch to the next green phase.
- `fixed_time=True` runs the network's native static 90 s programme and ignores
  actions; this is the fixed-time baseline.
- Invalid action IDs (anything outside 0/2/4/6) raise `ValueError` rather than
  silently doing something unsafe.

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

The contract is tagged only after the checks in §6 pass. From that point,
lanes branch from `main` (exact commands in the "Your assigned task" section
of `docs/reproduce.md`):

| Lane | Owner | Scope | Deliverables |
|---|---|---|---|
| A | SL + RL teammate | SL candidate `ridge`; RL agent Q-Learning | seeded SL training command + validation metrics JSON + notes (config edits confined to `models.ridge`); Q-Learning module, registration line in `agents.py`, `q_learning:` block in `configs/rl_agents.yaml`, seeded training/evaluation command + metrics JSON |
| B | SL teammate | Candidates `decision_tree` and `random_forest` | seeded training command, validation metrics JSON, notes; config edits confined to your `models.*` blocks |
| C | SL teammate | Candidates `hist_gradient_boosting` and `xgboost` | same |
| D | Project owner | RL agents DQN and Double DQN; fixed-time/actuated baseline evaluation, review and integration | DQN/Double DQN modules + registration lines + `configs/rl_agents.yaml` blocks, baseline results, review notes, integration |

Rules for every lane:

- Do not edit shared contract files; add new model or agent modules through
  the registry or agent interface.
- The five SL candidates are already registered and wired into the shared
  train harness: you may edit only **your own** `models.<name>` blocks in
  `configs/sl.yaml` (never frozen). RL ships no agent — create your module
  and register it as `src/reinforcement/agents.py` documents.
- Param changes stay inside **your own** `models.<name>` blocks in
  `configs/sl.yaml`; do not run the final test evaluation in the parallel
  phase (validation only) and do not touch shared files.
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

Expected: **61 passed** (14 M1/M2, 9 dataset, 7 EDA, 19 contract, 12
shared-harness tests).

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
- [x] Four SL candidates plus persistence baseline registered (a fifth,
  `xgboost`, was added post-freeze — see §7).
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
| `tests/test_shared_harnesses.py` | Tests covering the helpers, both harnesses, the time-block folds and the agent registry (suite now 61 tests). |

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

**Post-freeze addition (owner-approved):** `xgboost` was registered as a fifth
SL candidate — an `XGBRegressor` factory in `src/supervised/registry.py`, a
`models.xgboost` block in `configs/sl.yaml` and `xgboost==3.2.0` in
`requirements.txt`. No existing model or harness behaviour changed;
`available_models()` simply returns one more name.

---

## 8. Model selection rationale

None of the models below are trained in this repository yet — each lane fits
its own candidates and reports validation metrics. This section records why
each one was chosen and what it was preferred over.

### 8.1 SL candidates (five, plus the persistence baseline)

| Model | Why it is in the lineup | Preferred over |
|---|---|---|
| `persistence` | Naive floor: predict the current count. Shows immediately whether ML beats "no change". | Not an ML candidate; it can only win by being unbeatable. |
| `ridge` | Interpretable linear baseline with standardised coefficients; the lag features are strongly collinear (see the EDA report), so L2 shrinkage stabilises plain least squares. | OLS (unstable coefficients on collinear lags), KNN/SVR (poor extrapolation, slow at predict time), ARIMA/SARIMA (wrong structure for cross-sectional features). |
| `decision_tree` | Nonlinear thresholds and a single-tree picture that is easy to explain in the report. | Alone it is high-variance — which is why it is compared against the ensembles rather than used as the final model. |
| `random_forest` | Strong out-of-the-box ensemble with almost no tuning; impurity and permutation importances for the analysis section. | A single tree (variance) and boosting as the only option (all eggs in one basket). |
| `hist_gradient_boosting` | Histogram-based gradient boosting native to scikit-learn: no extra dependency, native NaN handling, fast on this dataset. | XGBoost/LightGBM as libraries — same algorithm family, extra dependency. |
| `xgboost` | Fifth candidate: the well-known GBDT reference with more tuning knobs (depth, rounds, learning rate); a cross-library check against the sklearn-native booster. | Added deliberately as a rival to `hist_gradient_boosting`; if the two only tie, that is a reportable ablation result. |

Rejected for scope: KNN, SVR, ARIMA/SARIMA and deep sequence models — wrong
inductive bias or overkill for 38,400 rows × 41 features. PLAN §7.2 lists
XGBoost as optional; it is included here as the fifth candidate.

### 8.2 RL candidates (three)

| Model | Why it is in the lineup | Preferred over |
|---|---|---|
| Q-Learning | Simplest RL baseline with a discretised state and an interpretable Q-table; establishes the floor. | Nothing — it is the floor every other agent must beat. |
| DQN | Handles the full numeric state (S0/S1) with a nonlinear value function. | Linear function approximators (too weak), a Q-table on a continuous state (needs coarse hand-made bins). |
| Double DQN | Target decoupling — `a* = argmax_a Q_online(s', a)`, `y = r + γ·Q_target(s', a*)` — reduces max-operator overestimation; a one-line change over DQN with no new hyperparameters. | Not guaranteed to win; it is the algorithmic-improvement comparison the plan requires. |

Rejected: PPO (on-policy and sample-inefficient against expensive hour-long
SUMO episodes, high tuning risk — PLAN lists it optional only) and any fourth
algorithm (padding the table). Fixed-time and actuated controllers are
baselines, not RL candidates.

---

## 9. Contract v2 amendments (review fixes)

The first parallel-phase review reports ([Lane A deadlock report](Junction%20Deadlock%20Report.html),
Lane C [`issues_found.md`](issues_found.md)) produced this batch of fixes. Everything below is already
covered by tests; the suite grew from 56 to 61. Tags: `contract-v1` (original
freeze), `contract-v2` (this batch).

### 9.1 RL environment

| Fix | Issue | What changed |
|---|---|---|
| Clearance routing | A-1, RL-1 | `_transition_to` now walks the native clearance path (§3.2 matrix) instead of jumping greens, so permissive left-turners are flushed before cross traffic goes green. New `left_clearance_seconds: 6` in `configs/rl.yaml`; helper `clearance_path(current, target, through_greens)` is unit-tested for all 16 green→green pairs. **All previously reported RL numbers are invalid — re-run evaluations against v2.** |
| Pending target | RL-4 | An interrupted clearance walk stores `_pending_target`; `action=None` resumes it instead of snapping back to the old green. |
| Unique route files | A-2 | `write_scenario_routes(..., suffix=...)`; the environment passes a per-process/per-instance suffix, so parallel same-scenario runs no longer overwrite each other's generated routes. Data collection is unchanged (default suffix). |
| Greedy evaluation | RL-2 | `evaluate.py` forces `agent.explore = False` **and** `agent.epsilon = 0.0` (when present); both flags are documented in the `src/reinforcement/agents.py` interface. |
| Training seed offset | RL-3 | Training episodes use `seed + 1000 + episode`, so episodes never land on the evaluation grid (seeds 42–45). Evaluation seeds unchanged. |
| Sample std | RL-5 | `evaluate.py` reports sample standard deviation (`ddof=1`); `0.0` for a single run. |
| Parallel-safe routes | A-2 | Verified with two simultaneous environment processes writing distinct `*_pid_id.rou.xml` files. |

### 9.2 Supervised learning

| Fix | Issue | What changed |
|---|---|---|
| HGB early stopping | SL-1 | `models.hist_gradient_boosting.params` now sets `early_stopping: false` — sklearn's default `"auto"` carved a random 10% holdout out of the ordered training rows. **Lane C should re-run `hist_gradient_boosting`.** |
| Seeded models | SL-2 | `ModelSpec.seeded` (persistence: `False`); the harness passes `random_state=<--seed>` into every seeded model, so `--seed 43` no longer reproduces `--seed 42` results. |
| Time-block CV | SL-3 | `--tune` uses `time_block_folds()` — expanding time-block folds with a 300 s embargo, all runs in every fold — instead of `TimeSeriesSplit`, which split by row order and made folds into scenario blocks. |
| Prediction input | SL-4 | `Predictor.predict_array` wraps NumPy input in a DataFrame with the stored feature names (no sklearn feature-name warnings). |

### 9.3 Supporting code, tests and docs

| Fix | Issue | What changed |
|---|---|---|
| Agent-registry test | A-4 | `test_registry_rejects_unknown_names` no longer asserts the registry is empty — lanes register into the shared module by design. |
| EDA autocorrelation | EDA-1 | Pairs are matched by exact timestamp (`simulation_time + lag*5` merge), so embargo gaps no longer create fake 5-second pairs. |
| EDA plot labels | EDA-2 | Axis labels say "vehicles on approach" instead of "vehicles / 5 s". |
| Baseline seeds | REP-1 | `characterise_baseline --seeds ...` accepts an explicit seed list; `experiments/reproduce.py` passes its `--seeds` through. |
| `PYTHONHASHSEED` | UTIL-1 | No longer assigned at runtime (it only affects child processes anyway); the comment explains it must be set before the interpreter starts. |
| Demand comment | SIM-1 | The misleading "decorrelates repeated demand draws" comment in `train.py` now states the truth: `vehsPerHour` arrivals are deterministic; seeds vary driver behaviour only. **Known limitation to note in the final report — recollecting with exponential headways would invalidate the frozen dataset, so it is documented instead.** |
| Lane table / tuning rule | DOC-1, DOC-2 | §1 consumer columns and §3.2 action table corrected; §5 and `reproduce.md` now allow param edits confined to a lane's own `models.*` blocks. |

**Deferred (not part of v2):** HYB-1 (online F4 feature buffer for S2) is
built with M5; DOC-3 (`lane_c_notes.md` §7 wording) is Lane C's to fix on their
next tag.

Verification for this batch: `61 passed`; both environment smoke runs unchanged
for the fixed-time path (mean queue 7.875 / throughput 570 / travel 81.0), and
a clearance walk `0 → 4` reaches phase 4 with **0 teleports**.
