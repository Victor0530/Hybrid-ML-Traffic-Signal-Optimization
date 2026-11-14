# Project Files and Folders

Living reference for the **Hybrid SL + RL Adaptive Traffic Signal Optimisation** project.

This document is ordered by milestone so it can grow with the project. Use it with
[`RUNBOOK.md`](RUNBOOK.md): this file explains **what each file is for**, while the
runbook explains **how to run and reproduce it**.

**Status legend**

| Badge | Meaning |
|---|---|
| ✅ | Implemented and verified |
| 🚧 | Currently being built |
| 📋 | Planned — not in the repository yet |

**Last updated:** 2026-11-13, after completing the Milestone 2 baseline characterisation.

---

## Milestone status

| Milestone | Goal | Status |
|---|---|---|
| **M1** | Environment and SUMO simulation work | ✅ Complete |
| **M2** | Reproducible traffic dataset and fixed-time baseline | ✅ Complete |
| **M3** | Supervised-learning pipeline | 📋 Next |
| **M4** | RL-only signal controller | 📋 Planned |
| **M5** | Hybrid SL + RL controller | 📋 Planned |
| **M6** | Final comparison and report | 📋 Planned |

---

## Repository tree

```text
Adaptive Traffic Signal Optimisation System/
├── .venv/                          # ✅ local Python 3.11 environment (gitignored)
├── .gitignore                      # ✅ Git exclusions
├── requirements.txt                # ✅ pinned Python dependencies
├── README.md                       # ✅ project overview
├── PLAN.md                         # ✅ detailed technical specification
├── FILES.md                        # ✅ this file
├── RUNBOOK.md                      # ✅ run/reproduce guide
├── AssignmentSpec.md               # ✅ local reference only (gitignored)
├── docs/
│   └── fixed_time_baseline.md      # ✅ fixed-time baseline report
│
├── configs/
│   ├── traffic_scenarios.yaml      # ✅ demand + simulation settings
│   ├── sl.yaml                     # 📋 planned SL configuration
│   ├── rl.yaml                     # 📋 planned RL configuration
│   ├── hybrid.yaml                 # 📋 planned hybrid configuration
│   └── final.yaml                  # 📋 planned final-evaluation configuration
│
├── data/
│   ├── raw/                        # ✅ generated observation CSVs (gitignored)
│   └── processed/                  # 📋 cleaned datasets and splits (gitignored)
│
├── sumo/
│   ├── network/
│   │   ├── intersection.nod.xml    # ✅ junction and edge-end nodes
│   │   ├── intersection.edg.xml    # ✅ road geometry and lanes
│   │   ├── intersection.con.xml    # ✅ lane-to-lane connections
│   │   ├── build_network.py        # ✅ netconvert wrapper
│   │   ├── intersection.net.xml    # ✅ generated SUMO network
│   │   └── intersection.sumocfg    # ✅ demo simulation config
│   └── routes/
│       ├── medium.rou.xml          # ✅ static demo traffic
│       └── generated/              # ✅ per-scenario routes (gitignored)
│
├── src/
│   ├── __init__.py                 # ✅ package marker
│   ├── simulation/                 # ✅ SUMO setup and data collection
│   │   ├── __init__.py
│   │   ├── sumo_binary.py
│   │   ├── check_connection.py
│   │   ├── scenario_generator.py
│   │   └── data_collector.py
│   ├── analysis/                   # ✅ fixed-time baseline analysis
│   │   ├── __init__.py
│   │   └── characterise_baseline.py
│   ├── preprocessing/              # 📋 SL dataset and features
│   ├── supervised/                 # 📋 SL training/evaluation
│   ├── reinforcement/              # 📋 RL environment and agents
│   ├── hybrid/                     # 📋 hybrid controller
│   ├── evaluation/                 # 📋 final comparison and statistics
│   └── utils/                      # 📋 shared helpers
│
├── models/                         # 📋 trained models (gitignored)
├── results/                        # ✅/📋 generated outputs (gitignored)
├── experiments/                    # 📋 end-to-end experiment runner
├── notebooks/                      # 📋 EDA and analysis notebooks
└── tests/                          # ✅ automated tests
```

> Folders marked 📋 do **not** exist yet. They are listed now so each milestone has
> a planned location and the structure stays consistent.

---

## M1 — Environment and simulation

**Goal:** build a valid four-way SUMO intersection and prove Python can control it
through TraCI.

### Files

| Path | Status | What it is | Output / notes |
|---|---|---|---|
| `requirements.txt` | ✅ | Pinned Python dependencies for the whole project: `traci`, `sumolib`, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, PyYAML, pytest. Optional extras are commented out. | Installed into `.venv` with `pip install -r requirements.txt`. |
| `.venv/` | ✅ | Local Python 3.11 environment. | Gitignored. Recreate with `python -m venv .venv`. |
| `configs/traffic_scenarios.yaml` | ✅ | Single source of truth for simulation duration, sampling interval, prediction horizon, turning split, and all five demand scenarios. | Read by `scenario_generator.py` and `data_collector.py`. |
| `sumo/network/intersection.nod.xml` | ✅ | Five nodes: central junction `C` (type `traffic_light`) and `N`, `S`, `E`, `W` edge ends at ±500 m. | Input to `netconvert`. |
| `sumo/network/intersection.edg.xml` | ✅ | Eight edges (`north_in/out`, `south_in/out`, `east_in/out`, `west_in/out`), two lanes per direction, **500 m node-to-node spacing** (489.60 m drivable lane length), 13.89 m/s speed limit. | Input to `netconvert`. |
| `sumo/network/intersection.con.xml` | ✅ | Twelve lane-to-lane movements: lane 0 handles straight/right, lane 1 handles the protected left. | Input to `netconvert`. |
| `sumo/network/build_network.py` | ✅ | Locates `netconvert` on PATH with a `%SUMO_HOME%\bin` fallback, then compiles the three plain-XML files into `intersection.net.xml` using relative paths and `cwd=sumo/network`. | Prints the command and `Built .../intersection.net.xml`; keeps the net XML portable. |
| `sumo/network/intersection.net.xml` | ✅ generated | Compiled network containing the junction, lane connections, and an 8-phase, 90 s fixed-time traffic-light program (`C`). | Used by `intersection.sumocfg` and every TraCI run. |
| `sumo/network/intersection.sumocfg` | ✅ | Demo SUMO config: 1-hour run, 1 s steps, 300 s teleport timeout, statistics enabled. | Uses `sumo/routes/medium.rou.xml`. |
| `sumo/routes/medium.rou.xml` | ✅ | Static medium-demand demo traffic: 12 routes and 12 flows, 600 veh/h per approach split 70 % straight / 15 % left / 15 % right. | Used by the demo config and `check_connection.py`. |
| `src/simulation/sumo_binary.py` | ✅ | `resolve_sumo_binary(name)` locates `sumo.exe`/`sumo-gui.exe`/`netconvert.exe` on PATH, falls back to `%SUMO_HOME%\bin`, appends `.exe` only on Windows, and raises a clear error if missing. `get_sumo_version(binary)` reads the actual binary version. | Prevents stale-PATH failures and records the correct SUMO version. |
| `src/simulation/check_connection.py` | ✅ | Minimal TraCI smoke test. Runs the medium scenario, passes `--end` to SUMO, and prints the signal phase, red/yellow/green state, and approach counts once per simulated minute. | Console table; no data files. |
| `src/__init__.py`, `src/simulation/__init__.py` | ✅ | Package markers so `python -m src.simulation...` works from the project root. | — |

### M1 outputs

- A valid network with **4 approaches × 2 lanes**, **12 turning movements**, and a
  fixed-time signal program.
- A repeatable TraCI connection test that prints, for example:

```text
  time  phase         state    north_in    south_in     east_in     west_in
    60      4  rrrGGgrrrGGg          10          10           6           5
```

- The generated `intersection.net.xml` is **not** gitignored, so it can be committed
  for convenience. It can always be rebuilt with `build_network.py`.

---

## M2 — Data collection and fixed-time baseline

**Goal:** generate a reproducible dataset that the supervised-learning and RL
phases can consume.

### Files

| Path | Status | What it is | Output / notes |
|---|---|---|---|
| `src/simulation/scenario_generator.py` | ✅ | Converts the YAML scenario definitions into deterministic SUMO route files. Builds 12 routes and one flow per direction/movement/demand phase (3 phases for `changing`). | Writes `sumo/routes/generated/{scenario}.rou.xml`. |
| `sumo/routes/generated/` | ✅ generated | Five route files: `low`, `medium`, `high`, `uneven`, `changing`. | Gitignored; regenerate with `--all`. |
| `src/simulation/data_collector.py` | ✅ | Launches SUMO through TraCI, steps the simulation, samples every 5 s, tracks per-approach departures/arrivals, writes one CSV per run, and updates the manifest atomically. Validates existing CSVs and records config/route/network hashes. Non-default durations get a `_d{duration}` suffix. | Writes `data/raw/{run_id}.csv` and `run_manifest.csv`. |
| `data/raw/` | ✅ generated | Observation data for **5 scenarios × 4 seeds = 20 runs**, 17 columns per row. | Gitignored; one CSV per run. |
| `data/raw/run_manifest.csv` | ✅ generated | One row per run: status, timestamps, duration, hashes, SUMO version, loaded/departed/arrived/running totals, teleports, unmapped arrivals, error. | 20 `ok` rows in the current dataset. |
| `src/analysis/characterise_baseline.py` | ✅ | Fixed-time baseline analysis. Bins observations into 5-minute windows, computes queue/wait/speed/throughput per approach, flags saturation/recovery, and writes tables and plots. | `results/fixed_time_baseline/` plus console summary. |
| `results/fixed_time_baseline/` | ✅ generated | `baseline_summary.csv`, `baseline_timeseries.csv`, `data_quality_summary.csv`, and five PNG plots. | Gitignored. |
| `docs/fixed_time_baseline.md` | ✅ | Written M2 report: TLS phases, data quality, scenario results, build-up/recovery/saturation/starvation, and RL design implications. | Tracked by Git. |
| `results/smoke_tripinfo.xml` | ✅ optional | Output from an earlier headless SUMO demo. | `results/` is gitignored; safe to delete. |

### Current dataset at a glance

| Scenario | Demand (veh/h per approach) | Runs | Observation rows |
|---|---|---:|---:|
| `low` | 300 on all approaches | 4 | 11,520 |
| `medium` | 600 on all approaches | 4 | 11,520 |
| `high` | 700 on all approaches | 4 | 11,520 |
| `uneven` | N 800, S 250, E 250, W 600 | 4 | 11,520 |
| `changing` | 400 → 800 at 1,800 s → 400 at 2,700 s | 4 | 11,520 |
| **Total** | — | **20** | **57,600** |

Each run contains **720 timestamps** (5 s to 3600 s) × **4 approaches** =
**2,880 rows**. Seeding: 42, 43, 44, 45.

The `high` scenario is near capacity but stable (last-10-minute queue 4.1–4.6
vehicles/approach). The `uneven` scenario deliberately keeps north at 800
veh/h, so north is the asymmetric stress case (last-10-minute queue 37.7
vehicles/approach). The `changing` scenario peaks at 800 veh/h and then
recovers when demand drops back to 400.

### M2 outputs

- 20 observation CSVs with 17 columns:

```text
run_id, scenario, seed, simulation_time, direction, vehicle_count,
mean_speed, queue_length, waiting_time, occupancy,
traffic_light_phase, phase_duration, time_in_phase,
no_vehicles, sensor_missing, departed_count, arrived_count
```

`departed_count` and `arrived_count` are per approach over the preceding 5 s
sampling interval. `mean_speed` is blank when `vehicle_count == 0` or the speed
is invalid; `occupancy` is a fraction from 0 to 1; `waiting_time` is the sum of
current vehicle waiting times on the approach.

- `data/raw/run_manifest.csv` with columns:

```text
run_id, scenario, seed, status, started_at, finished_at,
simulation_end, duration, sampling_interval, prediction_horizon,
route_file, config_hash, route_hash, network_hash, sumo_version,
loaded_total, departed_total, arrived_total, running_total,
teleports, arrived_unmapped, error
```

- Every run currently has `status=ok`, **0 teleports**, and no missing speed values.
- Data-quality checks: 0 missing values, 0 zero-vehicle rows, 0 duplicate keys,
  0 unmapped arrivals. `mean_speed` ranges 0.028–16.106 m/s (values above
  13.89 m/s come from SUMO `speedFactor`), and `occupancy` ranges 0.005–0.403.
- The empty-approach and sensor-missing branches are implemented but not
  exercised by the canonical dataset; a controlled sensor-loss test belongs in M3.

### M2 baseline summary

Fixed-time results, averaged over four seeds:

| Scenario | Queue mean | Queue last 10 min | Waiting mean (s) | Speed mean (m/s) | Throughput (veh/h/approach) |
|---|---:|---:|---:|---:|---:|
| low | 1.35–1.40 | 1.35–1.49 | 25.0–25.8 | 9.65–9.79 | 291.5–293.0 |
| medium | 3.03–3.22 | 2.98–3.28 | 54.4–58.9 | 8.52–8.76 | 580.5–585.3 |
| high | 4.10–4.25 | 4.14–4.58 | 75.9–80.0 | 7.99–8.08 | 676.0–683.3 |
| uneven (north) | 22.71 | 37.70 | 441.6 | 3.06 | 705.0 |
| changing | 4.21–4.77 | 1.84–2.14 | 87.0–100.8 | 7.78–8.21 | 488.5–492.0 |

Only uneven-north is flagged as ending saturated. High is stable, and changing
recovers after the demand drops. Full tables, plots, and the RL design
implications are in [`docs/fixed_time_baseline.md`](docs/fixed_time_baseline.md).

---

## M3 — Supervised learning (planned)

**Goal:** predict traffic volume five minutes ahead from current and past
observations.

These paths are reserved for the next phase and do not exist yet.

| Path | Status | Planned purpose | Expected output |
|---|---|---|---|
| `src/preprocessing/` | 📋 | Load and audit the raw CSVs, handle missing/empty observations, build lag and rolling features, construct the 5-minute-ahead target, and create chronological train/validation/test splits. | Clean dataset and split files. |
| `data/processed/` | 📋 | Processed SL dataset and split files. | `data/processed/` contents (gitignored). |
| `src/supervised/` | 📋 | Baseline and candidate models: Linear Regression, Decision Tree, Random Forest, optional XGBoost, tuning, evaluation, and blending. | Metrics, predictions, feature importances. |
| `configs/sl.yaml` | 📋 | Feature-set choice, model hyperparameters, split definition, and random seeds. | Read by SL training scripts. |
| `models/sl/` | 📋 | Trained model artifacts and any fitted preprocessors. | `.joblib`/model files (gitignored). |
| `results/sl/` | 📋 | MAE/RMSE/R² tables, residual plots, forecast plots, and ablation results. | Figures and CSV/JSON metrics (gitignored). |
| `notebooks/` | 📋 | Exploratory data analysis and result interpretation. | Notebooks, optional. |
| `tests/` | ✅/📋 | M2 tests cover deterministic route generation, collector CSV validation, config hashing, and SUMO binary resolution. Feature-construction, leakage, and metric tests are added in M3. | `pytest` results. |

### M3 design rules to preserve

- Target: `vehicle_count` **300 s (60 samples)** ahead, per `run_id` and `direction`.
- Use `groupby(["run_id", "direction"])["vehicle_count"].shift(-60)`; never impute
  the target.
- Keep `mean_speed` empty when `vehicle_count == 0` and expose the
  `no_vehicles`/`sensor_missing` flags.
- Use `departed_count`/`arrived_count` for throughput diagnostics. If they are
  used as features, lag or shift them; never leak the current interval's future
  counts into the target.
- Split chronologically, never shuffle time-ordered rows.
- Fit preprocessing on the training split only.
- Keep the best SL model frozen for the hybrid phase.

---

## M4 — Reinforcement learning (planned)

**Goal:** train an RL signal controller in SUMO and compare it with fixed-time
control.

| Path | Status | Planned purpose | Expected output |
|---|---|---|---|
| `src/reinforcement/` | 📋 | SUMO environment wrapper, state/action/reward logic, Q-Learning and DQN agents, training and evaluation loops. | Training scripts and evaluation code. |
| `configs/rl.yaml` | 📋 | Episode length, learning rate, discount factor, exploration schedule, network architecture, seeds, and reward weights. | Read by RL training/evaluation. |
| `models/rl/` | 📋 | Checkpoints and final RL models. | Model files (gitignored). |
| `results/rl/` | 📋 | Learning curves, per-episode rewards, waiting/queue metrics, and ablation results. | Figures and metric tables (gitignored). |

### M4 design rules to preserve

- State `S0`/`S1`: current counts, queues, speeds, signal phase, and elapsed phase time.
- Action: choose a valid non-conflicting TLS phase; optional green duration.
- Reward: `-(alpha × total_queue + beta × total_waiting_time)`.
- Train **without** SL predictions first; this is the RL-only baseline.
- Use multiple seeds and report mean ± spread.

---

## M5 — Hybrid SL + RL (planned)

**Goal:** add frozen SL predictions to the RL state and retrain the agent.

| Path | Status | Planned purpose | Expected output |
|---|---|---|---|
| `src/hybrid/` | 📋 | Build the expanded state (current observations + SL predictions), integrate the frozen SL model, and train/evaluate the hybrid agent. | Hybrid training/evaluation code. |
| `configs/hybrid.yaml` | 📋 | Which SL model is frozen, prediction horizon, state composition, and RL hyperparameters. | Read by hybrid scripts. |
| `models/hybrid/` | 📋 | Hybrid agent checkpoints and final model. | Model files (gitignored). |
| `results/hybrid/` | 📋 | Hybrid learning curves and evaluation metrics. | Figures and tables (gitignored). |

### M5 design rules to preserve

- Use only the **best tuned SL model** and the **best tuned RL agent**.
- Retrain or fine-tune the hybrid agent; do not simply append predictions to an
  already-trained RL policy without retraining.
- Keep the SL model frozen during hybrid training.
- Never feed future ground-truth counts into the state.

---

## M6 — Final evaluation and reporting (planned)

**Goal:** compare fixed-time, RL-only, and hybrid controllers under identical
conditions and produce report-ready evidence.

| Path | Status | Planned purpose | Expected output |
|---|---|---|---|
| `src/evaluation/` | 📋 | Compute waiting time, queue length, travel time, throughput, and cumulative reward; bootstrap confidence intervals; paired statistical tests; recovery-time analysis. | Comparison tables and figures. |
| `experiments/run_all.py` | 📋 | One-command runner for all systems, scenarios, and seeds. | Complete experiment outputs. |
| `configs/final.yaml` | 📋 | Held-out scenarios, seeds, episode counts, and metric settings. | Read by the final runner. |
| `results/final/` | 📋 | Final comparison tables, confidence intervals, statistical tests, and figures. | Report-ready artifacts (gitignored). |

### M6 comparison matrix to preserve

| System | Description |
|---|---|
| Fixed-time | The 90 s TLS program currently generated by `netconvert`. |
| RL-only | Best tuned RL agent using current observations only. |
| Hybrid | Best tuned RL agent using observations + frozen SL predictions. |

Evaluate on the five scenarios with matched seeds and durations. Report mean and
spread, not just a single run.

---

## Shared and supporting files

| Path | Status | Purpose |
|---|---|---|
| `.gitignore` | ✅ | Keeps the venv, caches, generated data, models, results, and generated routes out of Git. Ignores Markdown except `PLAN.md`, `README.md`, `FILES.md`, `RUNBOOK.md`, and `docs/*.md`. |
| `README.md` | ✅ | High-level project overview, workflow, scope, and setup summary. Cross-links this file and `RUNBOOK.md`. |
| `PLAN.md` | ✅ | Detailed assignment specification: phases, data rules, SL/RL/hybrid design, evaluation, and milestones. |
| `AssignmentSpec.md` | ✅ local only | Original assignment brief, kept as a local reference. It is intentionally ignored by `.gitignore` and is **not** part of the tracked submission. Stored as UTF-16, so some tools may display it as binary. |
| `FILES.md` | ✅ | This file. |
| `RUNBOOK.md` | ✅ | Command-by-command run and reproduction guide. |
| `docs/fixed_time_baseline.md` | ✅ | Fixed-time M2 baseline report: TLS phases, data quality, scenario results, saturation/recovery, and RL implications. Tracked by Git. |
| `src/analysis/` | ✅ | M2 analysis code. `characterise_baseline.py` reads the raw dataset and writes `results/fixed_time_baseline/` plus the report evidence. |
| `tests/` | ✅/📋 | M2 automated tests for route generation, dataset schema validation, config hashing, and binary resolution; M3–M6 tests are added with their phases. |
| `src/utils/` | 📋 | Shared helpers such as seeding, logging, and config loading. |

---

## Generated artifacts and Git status

| Artifact | Created by | Git status | Expected quantity |
|---|---|---|---|
| `sumo/network/intersection.net.xml` | `build_network.py` | Not ignored; may be committed | 1 file |
| `sumo/routes/generated/*.rou.xml` | `scenario_generator.py` | Ignored | 5 files |
| `data/raw/*_seed*.csv` | `data_collector.py` | Ignored | 20 files, 2,880 rows each |
| `data/raw/run_manifest.csv` | `data_collector.py` | Ignored | 20 data rows |
| `results/fixed_time_baseline/baseline_summary.csv` | `characterise_baseline.py` | Ignored | 20 scenario/direction rows |
| `results/fixed_time_baseline/baseline_timeseries.csv` | `characterise_baseline.py` | Ignored | 5-min bins |
| `results/fixed_time_baseline/data_quality_summary.csv` | `characterise_baseline.py` | Ignored | 1 table |
| `results/fixed_time_baseline/*.png` | `characterise_baseline.py` | Ignored | 5 plots |
| `docs/fixed_time_baseline.md` | Written report | Tracked | 1 file |
| `results/*` (other) | demo/evaluation scripts | Ignored | varies |
| `models/*` | training scripts | Ignored | varies |
| `data/processed/*` | preprocessing scripts | Ignored | varies |

All ignored artifacts can be recreated using the commands in
[`RUNBOOK.md`](RUNBOOK.md). The raw dataset is **not** committed, so a collaborator
must either run the collector or receive a copy of `data/raw/`.

---

## Where to add new files

| If you are adding... | Put it in... |
|---|---|
| A new demand scenario | `configs/traffic_scenarios.yaml` |
| A new SUMO network or TLS program | `sumo/network/` |
| A route-generation or collection script | `src/simulation/` |
| A baseline or dataset analysis script | `src/analysis/` |
| A written phase report | `docs/` |
| SL cleaning/feature code | `src/preprocessing/` |
| SL model code | `src/supervised/` |
| RL environment/agent code | `src/reinforcement/` |
| Hybrid integration code | `src/hybrid/` |
| Final metrics/statistical tests | `src/evaluation/` |
| A full experiment entry point | `experiments/` |
| Tests | `tests/` |
| Generated outputs | `data/processed/`, `models/`, or `results/` |

When a new milestone is completed, update its **status badge** and file table here,
then add the exact commands and expected output to `RUNBOOK.md`.
