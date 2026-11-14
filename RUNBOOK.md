# Run and Reproduce Guide

Step-by-step commands for the **Hybrid SL + RL Adaptive Traffic Signal Optimisation**
project, ordered by milestone.

Use this guide with [`FILES.md`](FILES.md), which explains what each file does.

> **Status legend**
> ✅ implemented and runnable now · 📋 planned, command not available yet
>
> **Current state:** M1 and M2 (dataset + fixed-time baseline) are complete.
> M3–M6 sections show the intended workflow and will be filled in as each
> phase is implemented.

---

## 0. Prerequisites and one-time setup

### Required software

| Requirement | Version used here |
|---|---|
| Python | 3.11.9 |
| Eclipse SUMO | 1.28.0 |
| Operating system | Windows 10/11 (PowerShell commands below) |

SUMO is **not** installed through `pip`. Install Eclipse SUMO separately, then make
sure `SUMO_HOME` and the SUMO `bin`/`tools` folders are on the system PATH.

### Create and activate the virtual environment

Run from the project root:

```powershell
# Only needed the first time
py -3.11 -m venv .venv

# Allow the local activation script in this PowerShell session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, `python` points to `.venv\Scripts\python.exe`.

> ⚠️ Use `python`, **not** `py`, when running project scripts. `py` uses the global
> Python launcher and can bypass the virtual environment.

### SUMO environment variables

A newly opened terminal normally inherits these automatically. If `netconvert` or
`sumo` is not found, set them for the current PowerShell session:

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\SUMO'
$env:Path = "$env:SUMO_HOME\bin;$env:SUMO_HOME\tools;$env:Path"
```

To fix this permanently, close **all** VS Code/terminal windows after installing
SUMO, then reopen them. Existing processes keep the old environment.

### Verify the setup

```powershell
python --version
python -c "import traci; print('traci', traci.__version__)"
netconvert --version
sumo --version
```

Expected:

```text
Python 3.11.9
traci 1.28.0
Eclipse SUMO netconvert Version 1.28.0
Eclipse SUMO sumo Version 1.28.0
```

If the SUMO version differs from `traci`, install the matching `traci`/`sumolib`
release or use the copy bundled with SUMO.

---

## M1 — Build and verify the simulation ✅

### 1.1 Build the SUMO network

```powershell
python sumo\network\build_network.py
```

What it does:

1. Finds `netconvert`.
2. Compiles `intersection.nod.xml`, `intersection.edg.xml`, and
   `intersection.con.xml`.
3. Writes `sumo\network\intersection.net.xml`.

Expected output (file arguments are relative so the net XML stays portable):

```text
Running: C:\Program Files (x86)\SUMO\bin\netconvert.EXE --node-files intersection.nod.xml --edge-files intersection.edg.xml --connection-files intersection.con.xml --output-file intersection.net.xml --no-turnarounds true --offset.disable-normalization true (cwd=C:\...\sumo\network)
Built C:\...\sumo\network\intersection.net.xml
```

The script now finds `netconvert` through PATH **or** `%SUMO_HOME%\bin`, so it
still works from a terminal opened before SUMO was installed.

Check the result:

```powershell
Test-Path sumo\network\intersection.net.xml
```

Expected: `True`.

### 1.2 Run the TraCI smoke test

```powershell
python src\simulation\check_connection.py --end 300
```

This runs the medium scenario for 300 simulated seconds and prints one row per
minute. Expected output (seed 42):

```text
Starting C:\Program Files (x86)\SUMO\bin\sumo.EXE with intersection.sumocfg, seed=42
  time  phase         state    north_in    south_in     east_in     west_in
    60      4  rrrGGgrrrGGg          10          10           6           5
   120      0  GGgrrrGGgrrr           5           4          11          11
   180      7  rrrrryrrrrry          15          14           8           8
   240      4  rrrGGgrrrGGg           8           9           9           9
   300      0  GGgrrrGGgrrr           6           6          12          12
Finished at t=300s: 1 vehicles arrived in the final step, 66 still running.
```

Notes:

- SUMO startup lines (server port, loading messages, statistics) may appear before
  or after the table because of stdout buffering.
- The exact counts depend on the network, route file, SUMO version, and seed. With
  the pinned version and seed 42, the table above is reproducible.

### 1.3 Watch the simulation in SUMO GUI

```powershell
python src\simulation\check_connection.py --gui --end 300
```

A `sumo-gui` window opens and shows vehicles moving through the intersection.
This does not write a dataset.

You can also run the static demo config directly:

```powershell
sumo-gui -c sumo\network\intersection.sumocfg
```

Expected: a four-way intersection with two lanes per approach, signal phases
cycling, and vehicles following the rules of the road.

### M1 checklist

- [x] `intersection.net.xml` exists.
- [x] `check_connection.py` prints a table without a TraCI error.
- [x] SUMO GUI shows vehicles and signal changes.
- [x] `python --version` reports 3.11.9.
- [x] `build_network.py` works without SUMO on the session PATH.

---

## M2 — Generate the dataset and fixed-time baseline ✅

### 2.1 Generate route files for all scenarios

```powershell
python -m src.simulation.scenario_generator --all
```

Expected output:

```text
Wrote C:\...\sumo\routes\generated\low.rou.xml
Wrote C:\...\sumo\routes\generated\medium.rou.xml
Wrote C:\...\sumo\routes\generated\high.rou.xml
Wrote C:\...\sumo\routes\generated\uneven.rou.xml
Wrote C:\...\sumo\routes\generated\changing.rou.xml
```

Generated files:

```text
sumo\routes\generated\low.rou.xml
sumo\routes\generated\medium.rou.xml
sumo\routes\generated\high.rou.xml
sumo\routes\generated\uneven.rou.xml
sumo\routes\generated\changing.rou.xml
```

Generate only one scenario:

```powershell
python -m src.simulation.scenario_generator --scenario medium
```

### 2.2 Collect one run

```powershell
python -m src.simulation.data_collector --scenario medium --seed 42
```

Expected output on a fresh run:

```text
[run]  medium_seed42: 3600s, sampling every 5s
[done] medium_seed42: C:\...\data\raw\medium_seed42.csv

1/1 runs completed successfully
```

Outputs:

```text
data\raw\medium_seed42.csv       # 2,880 data rows
data\raw\run_manifest.csv        # one row for this run
```

> If the CSV already exists, the collector skips it:
>
> ```text
> [skip] medium_seed42: ... already exists (use --overwrite)
>
> 1/1 runs completed successfully
> ```
>
> Add `--overwrite` to regenerate it.

### 2.3 Collect the full dataset

The canonical dataset is **5 scenarios × 4 seeds = 20 runs**.

```powershell
python -m src.simulation.data_collector --seed 42 --seed 43 --seed 44 --seed 45
```

On a fresh checkout, expect 20 pairs of `[run]`/`[done]` lines and:

```text
20/20 runs completed successfully
```

Outputs:

```text
data\raw\low_seed42.csv        data\raw\low_seed43.csv
data\raw\low_seed44.csv        data\raw\low_seed45.csv
data\raw\medium_seed42.csv     data\raw\medium_seed43.csv
data\raw\medium_seed44.csv     data\raw\medium_seed45.csv
data\raw\high_seed42.csv       data\raw\high_seed43.csv
data\raw\high_seed44.csv       data\raw\high_seed45.csv
data\raw\uneven_seed42.csv     data\raw\uneven_seed43.csv
data\raw\uneven_seed44.csv     data\raw\uneven_seed45.csv
data\raw\changing_seed42.csv   data\raw\changing_seed43.csv
data\raw\changing_seed44.csv   data\raw\changing_seed45.csv
data\raw\run_manifest.csv
```

Each observation CSV has **2,880 data rows**; the full dataset has **57,600 rows**.

Runtime: about 12–17 seconds per 3,600 s simulation on this machine, so the full
20-run dataset takes roughly 4–6 minutes.

### 2.4 Force regeneration

If the files already exist and you want to rerun them:

```powershell
python -m src.simulation.data_collector --seed 42 --seed 43 --seed 44 --seed 45 --overwrite
```

Without `--overwrite`, an existing run is skipped **only if it is valid**. The
collector checks the 17-column schema, row count, timestamp coverage, direction
set, manifest status, and configuration hash before printing:

```text
[skip] medium_seed42: ... already exists (use --overwrite)
```

It refuses to skip and asks for `--overwrite` when the file is incomplete, uses an
old schema, has a missing/failed manifest row, or was generated with a different
configuration, route file, or network:

```text
[invalid] medium_seed42: column schema changed; rerun with --overwrite
[invalid] medium_seed42: configuration changed; rerun with --overwrite
[invalid] medium_seed42: route changed; rerun with --overwrite
[invalid] medium_seed42: network changed; rerun with --overwrite
[invalid] medium_seed42: manifest network hash is missing; rerun with --overwrite
```

The manifest stores `config_hash`, `route_hash`, and `network_hash`, so a changed
config, generated route, or network cannot silently be mixed with old observations.
The network hash uses a canonical XML form that ignores netconvert's timestamp
comment, so rebuilding an unchanged network does not invalidate the dataset.

If a run fails partway through `--overwrite`, the previous CSV is intentionally
left in place while the manifest row is marked `failed`. This preserves the last
known-good file for inspection. Dataset loaders must therefore filter on
`status == "ok"` rather than treating every CSV in `data/raw/` as valid.

### 2.5 Run all scenarios for one seed

```powershell
python -m src.simulation.data_collector --seed 42
```

This runs the five scenarios with seed 42 and produces 5 CSVs (14,400 rows).

### 2.6 Watch one run in SUMO GUI

```powershell
python -m src.simulation.data_collector --scenario medium --seed 42 --gui --overwrite
```

Expected: `sumo-gui` opens, the run is collected, and the CSV is replaced when
finished. The GUI is slower than headless mode.

### 2.7 Override the duration for a quick test

```powershell
python -m src.simulation.data_collector --scenario low --seed 99 --duration 300
```

Expected: `[run] low_seed99_d300: 300s, sampling every 5s` and 60 timestamps × 4
approaches = **240 rows**. The `_d300` suffix keeps the test run separate from the
canonical `low_seed99` file, so it cannot overwrite a full-length run. The manifest
gets a matching `low_seed99_d300` row; delete that CSV and manifest row if you do
not want the test run in the dataset.

### 2.8 Validate the dataset

Count observation files:

```powershell
(Get-ChildItem data\raw\*_seed*.csv).Count
```

Expected: `20`.

Count manifest statuses:

```powershell
Import-Csv data\raw\run_manifest.csv | Group-Object status | Select-Object Name, Count
```

Expected:

```text
Name Count
---- -----
ok      20
```

Count rows in one run:

```powershell
(Import-Csv data\raw\medium_seed42.csv | Measure-Object).Count
```

Expected: `2880`.

Count total rows across all observation CSVs:

```powershell
(Get-ChildItem data\raw\*_seed*.csv |
    ForEach-Object { (Get-Content $_ | Measure-Object -Line).Lines - 1 } |
    Measure-Object -Sum).Sum
```

Expected: `57600`.

Inspect one run:

```powershell
Import-Csv data\raw\medium_seed42.csv | Select-Object -First 8 | Format-Table -AutoSize
```

Expected columns (17):

```text
run_id, scenario, seed, simulation_time, direction, vehicle_count,
mean_speed, queue_length, waiting_time, occupancy,
traffic_light_phase, phase_duration, time_in_phase,
no_vehicles, sensor_missing, departed_count, arrived_count
```

Check that the manifest has the reproducibility fields:

```powershell
Import-Csv data\raw\run_manifest.csv |
    Select-Object run_id, status, config_hash, route_hash, network_hash,
                  loaded_total, departed_total, arrived_total, running_total,
                  arrived_unmapped |
    Format-Table -AutoSize
```

Expected: 20 rows, all `status=ok`, and no blanks in the hash or total columns.
`loaded_total >= departed_total >= arrived_total`; `running_total` is the number
still in the network at the final step.

Check that per-approach throughput sums match the manifest:

```powershell
python -c "import pandas as pd, glob; d=pd.concat(pd.read_csv(f) for f in glob.glob('data/raw/*_seed*.csv')); print(d.groupby('run_id')[['departed_count','arrived_count']].sum().to_string())"
```

Each run's `arrived_count` total should equal that run's `arrived_total` in the
manifest.

### 2.9 Characterise the fixed-time baseline

This is the §6.6 baseline analysis for M2. It reads the 20 CSVs and manifest, bins
observations into 5-minute windows, computes per-approach throughput from
`arrived_count`, writes summary/timeseries CSVs and plots, and prints a compact
summary.

```powershell
python -m src.analysis.characterise_baseline
```

Expected output (abridged; the actual table has all 20 scenario/direction rows):

```text
Loaded 20 runs and 57,600 observation rows.

Data quality summary:
              metric     value
                runs    20.000
    observation_rows 57600.000
           scenarios     5.000
               seeds     4.000
missing_values_total     0.000
  mean_speed_missing     0.000
   zero_vehicle_rows     0.000
    no_vehicles_rows     0.000
 sensor_missing_rows     0.000
      duplicate_keys     0.000
    arrived_unmapped     0.000
      mean_speed_min     0.028
      mean_speed_max    16.106
       occupancy_min     0.005
       occupancy_max     0.403

Baseline summary:
scenario direction  queue_mean  queue_last10_mean  waiting_time_mean  mean_speed_mean  throughput_veh_per_h  saturated_end
changing      east       4.214              1.838             86.982            8.213                488.50          False
    high     north       4.104              4.340             75.914            8.067                683.25          False
  medium      west       3.054              2.981             55.423            8.706                580.75          False
  uneven     north      22.709             37.700            441.579            3.060                705.00           True
     low      east       1.350              1.488             25.047            9.792                292.50          False
...

Wrote C:\...\results\fixed_time_baseline\baseline_summary.csv
Wrote C:\...\results\fixed_time_baseline\baseline_timeseries.csv
Wrote C:\...\results\fixed_time_baseline\data_quality_summary.csv
Wrote plots to C:\...\results\fixed_time_baseline
```

Outputs:

```text
results\fixed_time_baseline\baseline_summary.csv        # per scenario/direction metrics
results\fixed_time_baseline\baseline_timeseries.csv     # 5-min bins per scenario/direction
results\fixed_time_baseline\data_quality_summary.csv    # missing/duplicate/flag counts
results\fixed_time_baseline\*.png                       # baseline plots
docs\fixed_time_baseline.md                    # written interpretation
```

Use `baseline_summary.csv` for the report tables and `docs/fixed_time_baseline.md` for the
interpretation. The current dataset gives: low/medium stable; high 700 stable;
uneven north intentionally saturated; changing recovers after the 800 veh/h peak.

### 2.10 Run the automated tests

```powershell
python -m pytest -q
```

Expected:

```text
............                                                             [100%]
14 passed
```

The M2 tests cover byte-identical route generation, calibrated scenario values,
collector CSV-schema validation, configuration hashing, stable network hashing
that ignores netconvert's timestamp, SUMO binary resolution error handling, and
baseline provenance validation against the exact canonical 20-run set.

---

## Data dictionary

### Observation CSV columns

| Column | Meaning |
|---|---|
| `run_id` | `{scenario}_seed{seed}`, plus `_d{duration}` when duration differs from 3600 s (e.g. `low_seed99_d300`) |
| `scenario` | `low`, `medium`, `high`, `uneven`, or `changing` |
| `seed` | SUMO random seed for the run |
| `simulation_time` | Simulated seconds; multiples of 5 |
| `direction` | Incoming approach: `north`, `south`, `east`, `west` |
| `vehicle_count` | Vehicles on the incoming edge at this sample |
| `mean_speed` | Mean speed in m/s, 3 decimals; blank when no vehicles or sensor missing. Can exceed 13.89 m/s because SUMO vehicles have a speed factor above 1.0. |
| `queue_length` | Halting vehicles on the edge |
| `waiting_time` | **Sum** of the current waiting times of all vehicles on the edge, in seconds. Not a per-vehicle average. |
| `occupancy` | Last-step occupancy as a **fraction 0–1** (not percent) |
| `traffic_light_phase` | SUMO TLS phase index (`0`–`7` for the current program) |
| `phase_duration` | Programmed duration of the current phase, in seconds |
| `time_in_phase` | Elapsed time in the current phase, in seconds |
| `no_vehicles` | `1` when `vehicle_count == 0`, otherwise `0` |
| `sensor_missing` | `1` when a vehicle count exists but mean speed is negative/NaN, otherwise `0` |
| `departed_count` | Vehicles that departed onto this approach during the preceding 5 s sampling interval |
| `arrived_count` | Vehicles from this approach that completed their trip during the preceding 5 s sampling interval |

Important:

- `vehicle_count == 0` is a **valid** observation, not missing data.
- `mean_speed` is intentionally empty when there are no vehicles or when the
  sensor value is invalid; pandas will read it as `NaN`.
- In the current 20-run dataset there are **no** empty-speed rows, no zero-vehicle
  rows, and both flags are always zero. The empty/sensor-missing branches exist
  but are not exercised by the canonical data; test them separately rather than
  fabricating dirty training rows.
- Never impute the prediction target.
- Keep the `no_vehicles` and `sensor_missing` flags as features.
- `departed_count` and `arrived_count` support per-approach throughput. Summing
  `arrived_count` over a run equals the manifest `arrived_total`.

### Run manifest columns

| Column | Meaning |
|---|---|
| `run_id` | `{scenario}_seed{seed}`, or `{scenario}_seed{seed}_d{duration}` for non-default durations |
| `status` | `ok` or `failed` |
| `started_at`, `finished_at` | UTC ISO timestamps; these change on every run |
| `simulation_end` | Last simulated time reached |
| `duration` | Requested duration in seconds |
| `sampling_interval` | Seconds between observations (5) |
| `prediction_horizon` | SL horizon in seconds (300) |
| `route_file` | Route file used by the run, as a project-relative POSIX path |
| `config_hash` | SHA-256 of the effective scenario configuration |
| `route_hash` | SHA-256 of the generated route file |
| `network_hash` | SHA-256 of the canonical XML of `intersection.net.xml`; netconvert's timestamp comment is excluded, so an unchanged network keeps the same hash |
| `sumo_version` | Version reported by the **SUMO binary** (e.g. `1.28.0`) |
| `loaded_total` | Vehicles loaded into the simulation |
| `departed_total` | Vehicles that departed (entered the network) |
| `arrived_total` | Vehicles that completed trips during the run |
| `running_total` | Vehicles still in the network at the final step |
| `teleports` | Total starting teleports during the run |
| `arrived_unmapped` | Arrivals whose approach could not be mapped to a departure; should be `0` |
| `error` | Empty on success, otherwise the exception text |

---

## Reproducibility rules

### What is deterministic

With the same inputs, the pipeline reproduces the same observation CSVs:

- Same SUMO version (**1.28.0**).
- Same `configs/traffic_scenarios.yaml`.
- Same network XML files.
- Same route files.
- Same seeds (`42`, `43`, `44`, `45`).
- Same sampling interval and duration.
- Same operating system and Python environment.

Network topology and route generation are deterministic. `netconvert` writes a
generation timestamp comment into `intersection.net.xml`, so the raw file bytes
change on every rebuild even when the topology is identical. The collector
therefore hashes a canonical XML form that excludes comments rather than the raw
file. The simulation itself is seed-controlled: SUMO's random number generator
drives departures, lane changes, and other stochastic behaviour.

The manifest records `config_hash`, `route_hash`, and `network_hash` for every run.
If you edit `configs/traffic_scenarios.yaml`, a generated route file, or the
network, the next collector run will refuse to skip old files and will ask for
`--overwrite`. This prevents mixing observations from different demand
definitions, routes, or signal/network layouts.

### What changes on every run

- `started_at` and `finished_at` in `run_manifest.csv`.
- TraCI server port and startup log lines.
- Wall-clock runtime.
- GUI rendering.

These do not affect the observation values.

### Hash check

Save hashes before and after a regeneration:

```powershell
New-Item -ItemType Directory -Force results | Out-Null

Get-FileHash data\raw\*_seed*.csv |
    Sort-Object Path |
    Export-Csv results\hashes_before.csv -NoTypeInformation

python -m src.simulation.data_collector --seed 42 --seed 43 --seed 44 --seed 45 --overwrite

Get-FileHash data\raw\*_seed*.csv |
    Sort-Object Path |
    Export-Csv results\hashes_after.csv -NoTypeInformation

Compare-Object `
    (Import-Csv results\hashes_before.csv) `
    (Import-Csv results\hashes_after.csv) `
    -Property Hash, Path
```

No output from `Compare-Object` means the observation files are identical. The
manifest will still differ because of its timestamps.

### Why the dataset is not committed

`data/raw/`, `data/processed/`, `models/`, `results/`, and
`sumo/routes/generated/` are listed in `.gitignore`. They are generated artifacts
and can be recreated. Share the CSVs separately if a teammate cannot run SUMO.

---

## M3 — Supervised-learning pipeline 📋

> **Status: planned — not implemented yet.**
> The commands below describe the intended interface and may change when the
> phase is built. Do not expect them to work on the current repository.

### Planned commands

```powershell
# Build the modelling dataset, features, target, and chronological splits
python -m src.preprocessing.build_dataset

# Train and tune the supervised models
python -m src.supervised.train --config configs\sl.yaml

# Evaluate the selected model on the held-out test split
python -m src.supervised.evaluate --config configs\sl.yaml --split test
```

### Planned outputs

```text
data\processed\sl_dataset.csv          # one row per run/direction/timestamp
data\processed\splits\train.csv
data\processed\splits\validation.csv
data\processed\splits\test.csv
models\sl\*.joblib                     # fitted models and preprocessors
results\sl\metrics.csv                 # MAE, RMSE, R²
results\sl\predictions.csv
results\sl\feature_importance.png
results\sl\residuals.png
```

### Expected dataset shape

- Raw samples: 20 runs × 720 timestamps × 4 approaches = **57,600 rows**.
- With a 300 s target and 5 s sampling, the last **60 samples per run** have no
  future target and must be dropped.
- Upper bound after target construction:
  20 × 660 × 4 = **52,800 rows**, before any additional missing-value handling.

### Design requirements

- Target: `vehicle_count` 300 s ahead, built with
  `groupby(["run_id", "direction"])["vehicle_count"].shift(-60)`.
- Chronological splits only; never shuffle time-ordered observations.
- Fit scalers/encoders on the training split only.
- Compare Linear Regression, Decision Tree, and Random Forest; optional XGBoost.
- Report MAE, RMSE, and R²; include feature-set ablations and residual analysis.
- Use `departed_count`/`arrived_count` for throughput diagnostics; do not use
  future interval counts as features unless they are shifted like any other lag.
- Freeze the best tuned SL model for M5.

---

## M4 — RL-only controller 📋

> **Status: planned — not implemented yet.**
> Intended interface; command names and flags will be finalized with the code.

### Planned commands

```powershell
# Train the RL agent against SUMO
python -m src.reinforcement.train --config configs\rl.yaml

# Evaluate a trained agent on held-out episodes
python -m src.reinforcement.evaluate --config configs\rl.yaml --episodes 10
```

### Planned outputs

```text
models\rl\*.pt                        # checkpoints and final agent
results\rl\training_curve.png
results\rl\eval_metrics.csv
results\rl\per_episode_rewards.csv
```

### What will be compared

- State `S0`/`S1`: current counts, queues, speeds, phase, elapsed phase time.
- Action: valid non-conflicting TLS phase (and optional green duration).
- Reward: `-(alpha × total_queue + beta × total_waiting_time)`.
- RL-only training **without** SL predictions, so it is a fair baseline.
- Multiple seeds; report mean and spread.

---

## M5 — Hybrid SL + RL controller 📋

> **Status: planned — not implemented yet.**
> Intended interface; command names and flags will be finalized with the code.

### Planned commands

```powershell
python -m src.hybrid.train --config configs\hybrid.yaml
python -m src.hybrid.evaluate --config configs\hybrid.yaml
```

### Planned outputs

```text
models\hybrid\*.pt
results\hybrid\training_curve.png
results\hybrid\eval_metrics.csv
```

### Design requirements

- Use the **best tuned SL model** and the **best tuned RL agent** only.
- Append frozen SL predictions to the RL state.
- Retrain or fine-tune the hybrid agent; do not reuse an RL policy unchanged.
- Keep the SL model frozen during hybrid training.
- Never use future ground-truth counts as state features.

---

## M6 — Final comparison and reporting 📋

> **Status: planned — not implemented yet.**
> Intended interface; command names and flags will be finalized with the code.

### Planned command

```powershell
python experiments\run_all.py --config configs\final.yaml
```

### Planned outputs

```text
results\final\comparison.csv
results\final\confidence_intervals.csv
results\final\statistical_tests.csv
results\final\waiting_time.png
results\final\queue_length.png
results\final\throughput.png
```

### Final comparison

| System | Description |
|---|---|
| Fixed-time | The 90 s TLS program generated by `netconvert` |
| RL-only | Best tuned RL agent with current observations |
| Hybrid | Best tuned RL agent with observations + SL predictions |

Use matched seeds and durations across all systems. Report waiting time, queue
length, travel time, throughput, and cumulative reward with bootstrap confidence
intervals and paired statistical tests.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: Could not find SUMO binary 'netconvert'` | Terminal process started before SUMO's PATH was updated, and `SUMO_HOME` is unset | Set `$env:SUMO_HOME` and prepend `$env:Path`, or fully restart VS Code/terminal. |
| `SUMO_HOME` prints nothing | Environment was not inherited | Use the session `$env:` commands above; reopen the terminal after installing SUMO. |
| `ModuleNotFoundError: No module named 'traci'` | Venv not activated, or `py` used instead of `python` | Activate `.venv` and run `python ...`; check `python --version`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `[skip] ... already exists` | Existing CSV is valid and matches the current config | Add `--overwrite` if you actually want to regenerate it. |
| `[invalid] ... column schema changed` | CSV was generated before the 17-column update | Add `--overwrite` to regenerate with `departed_count`/`arrived_count`. |
| `[invalid] ... configuration changed` | `config_hash` differs from the manifest | This is intentional after editing the scenario YAML. Add `--overwrite` to rebuild the affected runs. |
| `FileNotFoundError: Could not find SUMO binary 'sumo'` | SUMO binary not found | Check `Test-Path "$env:SUMO_HOME\bin\sumo.exe"`; fix PATH/`SUMO_HOME`. |
| TraCI connection error or port already in use | A previous SUMO process is still running | Close leftover `sumo.exe`/`sumo-gui.exe` processes, then rerun. |
| `high` scenario queues keep growing | Demand is above capacity | Current calibrated high (700/approach) is stable. If you raise it again, expect a backlog and document it as a stress scenario. |
| `uneven` north queue grows | Intentional asymmetric stress | North demand is 800 while south/east are 250; this is the designed saturation case, not a bug. |
| `changing` queues do not fall at the end | Recovery phase missing or insufficient | Current config includes 400 veh/h from 2700–3600 s; check the last10 queue in `results/fixed_time_baseline/baseline_summary.csv`. |
| `mean_speed` is blank in some rows | `vehicle_count == 0` or invalid speed | Expected. Use `no_vehicles`/`sensor_missing`; do not impute the target. Current canonical data has none. |
| GUI is much slower than headless | `sumo-gui` renders every step | Use headless mode for dataset generation; reserve `--gui` for demos. |
| Full dataset takes longer than expected | SUMO speed depends on CPU and scenario | A 3,600 s run typically takes 12–17 s here; 20 runs take 4–6 minutes. |

---

## Command cheat sheet

| Milestone | Command | Output |
|---|---|---|
| Setup | `python -m pip install -r requirements.txt` | Dependencies installed |
| M1 | `python sumo\network\build_network.py` | `intersection.net.xml` |
| M1 | `python src\simulation\check_connection.py --end 300` | Console table |
| M1 | `python src\simulation\check_connection.py --gui --end 300` | SUMO GUI |
| M2 | `python -m src.simulation.scenario_generator --all` | 5 route files |
| M2 | `python -m src.simulation.data_collector --scenario medium --seed 42` | 1 CSV + manifest |
| M2 | `python -m src.simulation.data_collector --seed 42 --seed 43 --seed 44 --seed 45` | 20 CSVs + manifest |
| M2 | Add `--overwrite` | Regenerate existing runs |
| M2 | `python -m src.analysis.characterise_baseline` | Baseline CSVs + plots + report |
| M2 | `python -m pytest -q` | 14 passed |
| M3 | `python -m src.preprocessing.build_dataset` | Processed dataset 📋 |
| M3 | `python -m src.supervised.train --config configs\sl.yaml` | SL models 📋 |
| M4 | `python -m src.reinforcement.train --config configs\rl.yaml` | RL model 📋 |
| M5 | `python -m src.hybrid.train --config configs\hybrid.yaml` | Hybrid model 📋 |
| M6 | `python experiments\run_all.py --config configs\final.yaml` | Final results 📋 |

---

## Milestone completion checklist

### M1 ✅

- [x] SUMO 1.28.0 installed and importable from Python.
- [x] Network builds without errors.
- [x] TraCI smoke test prints signal phases and approach counts.
- [x] GUI demo shows moving vehicles.

### M2 ✅

- [x] Five scenario route files generated.
- [x] 20 runs collected (5 scenarios × 4 seeds).
- [x] 57,600 observation rows across 20 CSVs.
- [x] 17-column schema includes `departed_count` and `arrived_count`.
- [x] Manifest shows 20 `ok` runs, 0 teleports, 0 unmapped arrivals.
- [x] Config, route, and network hashes recorded for every run.
- [x] High demand calibrated to 700/approach and confirmed stable.
- [x] Changing scenario includes a recovery phase (400 → 800 → 400).
- [x] Baseline characterisation and `docs/fixed_time_baseline.md` completed.
- [x] Automated M2 tests pass (`14 passed`).
- [x] Reproducibility verified with fixed seeds.

### M3–M6 📋

- [ ] M3: SL dataset, baselines, tuned model, metrics.
- [ ] M4: RL environment, Q-Learning/DQN, evaluation.
- [ ] M5: frozen SL + retrained hybrid agent.
- [ ] M6: fixed-time vs RL-only vs hybrid comparison and report.
