# Code Review Report — Hybrid ML Traffic Signal Optimisation

**Date:** 2026-11-20
**Reviewed:** `main` at `93f9c58`, plus the Lane C hand-back tag `lane-c-v1`
(`a6c45bb`)
**Reviewer:** Terrence (Lane C)

## How the review was done

- **Code and configuration:** all tracked Python modules, configs, the SUMO
  network files, the notebook and the tests were read. The repository contains
  no SQL.
- **Checks that were run:**
  - `pytest`: 56 passed.
  - Saved Lane C models and metrics were inspected.
  - SL training was rerun with `--seed 43`.
  - The `--tune` CV folds were inspected.
  - The signal phase strings in `intersection.net.xml` were checked.
- **Limits:**
  - No RL agent exists yet, so RL issues come from reading the code and the
    network, not from training runs.
  - RL issues are listed for completeness. The RL lanes are still in progress.

## Summary

| ID | Severity | Area | Issue |
|---|---|---|---|
| RL-1 | **High** | RL environment | Left turns get no yellow when the agent jumps between non-adjacent phases |
| RL-2 | Medium | RL evaluation | Greedy evaluation relies on an `explore` attribute that the agent interface never defines |
| RL-3 | Medium | RL training | Training seeds overlap the evaluation seeds |
| RL-4 | Low | RL environment | `action=None` after an interrupted yellow returns to the old green |
| RL-5 | Low | RL evaluation | Standard deviation uses `ddof=0` with only 4 seeds |
| SIM-1 | Medium | Simulation | Demand is deterministic; seeds only change driver behaviour |
| SL-1 | Medium | SL (Lane C) | HGB silently trains on 90% of the training rows |
| SL-2 | Medium | SL harness | `--seed` does not reach the models |
| SL-3 | Medium | SL harness | `--tune` CV folds are split by scenario, not by time |
| SL-4 | Low | SL predictor | NumPy input to `Predictor.predict_array` triggers a feature-name warning |
| HYB-1 | Medium | Integration | No online feature builder for the S2 (SL-assisted) RL state |
| EDA-1 | Low | EDA | Autocorrelation pairs cross the embargo gaps |
| EDA-2 | Low | EDA | Plot labels describe `vehicle_count` as "per 5 s" |
| REP-1 | Low | Reproduction | Baseline analysis hard-codes seeds 42–45 |
| UTIL-1 | Low | Utilities | Setting `PYTHONHASHSEED` at runtime has no effect |
| DOC-1 | Low | Docs | Contract §1 still shows the old lane assignment |
| DOC-2 | Low | Docs | Contract §5 and `reproduce.md` still say "no tuning" |
| DOC-3 | Low | Docs | Lane C notes wrongly say HGB has no randomness |

---

## RL issues

### RL-1 (High) — Left turns get no yellow on non-adjacent phase jumps

**Where:** `src/reinforcement/environment.py` lines 463–471 (`_transition_to`);
`sumo/network/intersection.net.xml` lines 114–121.

**What:** The environment leaves a green phase through its paired yellow
(`0→1`, `2→3`, `4→5`, `6→7`). Phases 1 and 5 are not full clearance phases:

```
phase 0: GGgrrrGGgrrr    phase 1: yygrrryygrrr   <- N/S left turns still 'g'
phase 4: rrrGGgrrrGGg    phase 5: rrryygrrryyg   <- E/W left turns still 'g'
```

In the native cycle (`0→1→2`) the left turns continue into their protected
phase 2, so this is correct. However, the agent may jump straight from `0` to
`4` or `6` (or from `4` to `0` or `2`). The left-turn movements then go from
green straight to red with no yellow.

**Impact:**

- **`0 → 6`:** the north/south left turns lose their green while the east/west
  left turns gain theirs, with no clearance time in between.
- **Possible effects:** conflicts, emergency braking or teleports inside the
  junction.
- **Results affected:** the RL training and evaluation numbers.

**Suggested fix:** add full-yellow transition states for jumps that skip the
left-turn phase, for example `yyyrrryyyrrr` when leaving phase 0 for anything
other than phase 2. Alternatively, route such jumps through `2→3` or `6→7`.

### RL-2 (Medium) — Greedy evaluation depends on an undefined attribute

**Where:** `src/reinforcement/evaluate.py` line 116;
`src/reinforcement/agents.py` (interface docstring).

**What:** The evaluator sets `agent.explore = False` to switch off exploration.
The documented agent interface only lists `epsilon`, not `explore`. If an agent
follows the documented interface, setting `explore` has no effect.

**Impact:** "Greedy" evaluation may still take random actions, so results are
not deterministic and not comparable between agents.

**Suggested fix:** add `explore` (or an `act(observation, greedy=True)`
argument) to the documented interface. Alternatively, set `agent.epsilon = 0.0`
in the evaluator.

### RL-3 (Medium) — Training seeds overlap evaluation seeds

**Where:** `src/reinforcement/train.py` line 140; `configs/rl.yaml` line 46.

**What:**

- Training uses `seed + episode`, so seeds 42, 43, … 241.
- Evaluation uses seeds 42–45, which are the first four training episodes.

**Impact:** the agent is evaluated on runs it trained on, which can make the
results look better than they are.

**Suggested fix:** use disjoint seeds, for example training from seed 1000
upward and keeping 42–45 for evaluation only.

### RL-4 (Low) — `action=None` after an interrupted yellow returns to the old green

**Where:** `src/reinforcement/environment.py` lines 414–420.

**What:**

- If the decision deadline arrives during a yellow, the next `step(None)`
  resolves the target as `yellow_to_green[current]`.
- That is the green **before** the yellow, so the signal goes green A →
  yellow A → green A, and the switch the agent asked for is dropped.

**Suggested fix:** store the pending target phase and complete it when the
next action is `None`.

### RL-5 (Low) — Population standard deviation with 4 seeds

**Where:** `src/reinforcement/evaluate.py` line 84.

**What:** `np.std` uses `ddof=0`. With n = 4 this understates the spread.

**Suggested fix:** use `ddof=1` (sample standard deviation), or report it
explicitly as the population value.

---

## Simulation issue

### SIM-1 (Medium) — Demand is deterministic

**Where:** `src/simulation/scenario_generator.py` line 83; comment in
`src/reinforcement/train.py` line 139.

**What:** Flows use `vehsPerHour`, which SUMO inserts at **equal intervals**.
The SUMO `--seed` therefore only changes driver behaviour (`sigma=0.5`), not
arrivals. The comment "a fresh seed per episode decorrelates repeated demand
draws" is not accurate.

**Impact:**

- The four seeds per scenario are close to duplicates.
- Seed-to-seed variance and run-level bootstrap confidence intervals will be
  narrower than for real traffic.
- RL agents see the same arrival pattern in every episode.

**Suggested fix:** use random arrivals, for example `period="exp(<rate>)"` or
`probability=` on each flow. This would require recollecting the data, so
alternatively document the limitation in the report.

---

## SL issues

### SL-1 (Medium) — HGB trains on only 90% of the training rows

**Where:** `src/supervised/registry.py` lines 108–112
(`HistGradientBoostingRegressor` with the default `early_stopping="auto"`).

**What:**

- With more than 10,000 rows, `"auto"` turns early stopping on. Verified on the
  saved F4 model: `do_early_stopping_ = True`, `validation_fraction = 0.1`.
- 10% of the training rows (2,400) are held out **at random**, ignoring time
  order. HGB therefore trains on 21,600 rows, while XGBoost and the other
  models use all 24,000.
- On this data the model never stops early (`n_iter_ = 300` of 300), so the
  held-out rows are simply lost.

**Impact:** HGB is compared with the other candidates on unequal training data.

**Suggested fix:** set `early_stopping: false` in
`models.hist_gradient_boosting.params`, or decide explicitly to keep it. This
is in Lane C's block, so the owner decides; HGB would then need to be re-tuned.

### SL-2 (Medium) — `--seed` does not reach the models

**Where:** `src/supervised/registry.py` lines 101–116;
`src/supervised/train.py` `_fit_one`.

**What:**

- The registry fixes `random_state=42` for every tree model.
- The CLI `--seed` only seeds NumPy and is recorded in the JSON.
- Verified: `--seed 43` gives identical HGB and XGBoost results to `--seed 42`.

**Impact:** multi-seed runs and seed-variance reporting are not possible through
the harness. The `seed` field in each JSON suggests a seed effect that does not
exist.

**Suggested fix:** pass `random_state=seed` into `build_model` whenever the
estimator accepts it.

### SL-3 (Medium) — `--tune` CV folds are split by scenario, not by time

**Where:** `src/supervised/train.py` line 104.

**What:** Training rows are ordered by `run_id`, so `TimeSeriesSplit(n_splits=3,
gap=240)` cuts the data into scenario blocks:

| Fold | Training scenarios | Validation rows |
|---|---|---|
| 1 | changing, high | high 3,600, low 2,400 |
| 2 | changing, high, low | medium 3,600, low 2,400 |
| 3 | changing, high, low, medium | **uneven 4,800** (never seen in training), medium 1,200 |

There is no leakage, but this is not the "time-aware CV" the contract
describes. The 240-row gap does not mean 300 s here.

**Impact:**

- **Stage 4 (tune with time-aware CV):** it would select parameters for
  generalising to an unseen scenario, not to a later time period.

**Suggested fix:** build folds by time inside each run, for example with a
custom splitter on `simulation_time` that keeps all runs in every fold and
leaves a 300 s gap. Alternatively, use grouped CV by run and document it.

### SL-4 (Low) — NumPy input triggers a feature-name warning

**Where:** `src/supervised/predictor.py` lines 47–54.

**What:** The pipeline is fitted on a DataFrame. Passing a NumPy array to
`predict_array` makes scikit-learn warn "X does not have valid feature names".
The results are still correct.

**Suggested fix:** wrap the array in a DataFrame with `self.feature_names`
before calling `predict`.

---

## Integration issue

### HYB-1 (Medium) — No online feature builder for the S2 state

**Where:** `src/reinforcement/environment.py` (`Observation.as_vector`, state
S2); `src/supervised/predictor.py`.

**What:**

- State S2 needs four SL predictions.
- The SL models need the full F4 feature vector: lags and rolling statistics
  over 5 s samples, up to 300 s of history.
- The environment only stores the current snapshot and decides every 15 s.
- No code keeps the 5 s history or builds F4 features during an episode.

**Impact:** the hybrid (SL + RL) stage cannot run until this exists.

**Suggested fix:** add an online feature buffer that the environment updates
every 5 s and that produces F4 rows for `Predictor.predict`. Reuse
`build_features` so that the features match training exactly.

---

## EDA, reproduction and utility issues

### EDA-1 (Low) — Autocorrelation pairs cross the embargo gaps

**Where:** `src/analysis/explore_sl_dataset.py` lines 185–187.

**What:** `shift(lag)` moves by rows. The processed dataset has no rows in the
embargo gaps (1,805–2,100 s and 2,705–3,000 s), so a few pairs labelled "lag 1"
(5 s) are actually more than 300 s apart.

**Impact:** small bias in `autocorrelation.csv`.

**Suggested fix:** shift on the raw continuous series, or drop pairs whose time
difference is not `lag × 5 s`.

### EDA-2 (Low) — Wrong unit in plot labels

**Where:** `src/analysis/explore_sl_dataset.py` lines 318 and 402.

**What:** The labels say "vehicles / 5 s". `vehicle_count` is the number of
vehicles on the approach at that moment, not a count per 5 s.

**Suggested fix:** change the label to "vehicles on approach".

### REP-1 (Low) — Baseline analysis hard-codes seeds 42–45

**Where:** `src/analysis/characterise_baseline.py` line 37.

**What:** `experiments/reproduce.py --seeds` accepts any seeds, but the baseline
stage requires exactly 42–45. For example, `--seeds 42` on a clean checkout
fails at the baseline step.

**Suggested fix:** pass the chosen seeds through to the baseline analysis, or
document that only 42–45 are supported.

### UTIL-1 (Low) — `PYTHONHASHSEED` set too late

**Where:** `src/utils/reproducibility.py` line 27.

**What:** Python reads this variable only at interpreter start-up. Setting it
inside a running process has no effect on that process; it only affects child
processes.

**Suggested fix:** remove the line, or document that it only affects
subprocesses.

---

## Documentation issues

### DOC-1 (Low) — Contract §1 shows the old lane assignment

**Where:** `docs/contract_v1.md` lines 23–31.

**What:** The table still says "SL lanes A and B" and "RL lanes C and D". Since
commit `93f9c58` the SL lanes are A, B and C, and the RL lanes are A and D.

### DOC-2 (Low) — "No tuning" rule is out of date

**Where:** `docs/contract_v1.md` line 302; `docs/reproduce.md` line 95.

**What:** Both still say "do not tune in the parallel phase", but the owner
approved parameter changes for all SL candidates on 2026-11-17.

### DOC-3 (Low) — Lane C notes: incorrect statement about HGB

**Where:** `docs/lane_c_notes.md` §7 (in tag `lane-c-v1`).

**What:** The notes say "HGB has no randomness in this setup". This is wrong:
HGB's early-stopping holdout is a random 10% split (`random_state=42`); see
SL-1.

**Fix:** Lane C will correct this in the next tag.

---

## Verified as correct

- **Splits:** train / validation / test are chronological, with a 300 s embargo
  between them. No target is shared across splits.
- **Features:**
  - Lag and rolling features only use past data.
  - The F4 cross-direction features only use the same timestamp.
  - The warm-up removes rows without full history.
- **Preprocessing:** imputation and scaling are fitted on the training split
  only, inside the pipeline.
- **Metrics:** the MAE, RMSE, R² and MAPE formulas are correct. MAPE skips no
  rows on these splits, because no target is 0.
- **Persistence baseline:** it reads `vehicle_count` correctly inside the
  pipeline.
- **Data collection:** config, route and network hashes, schema checks and
  atomic writes work as intended.
- **Lane C:** the notes match the regenerated metrics JSON for F0–F4.
- **Tests:** all 56 pass.
