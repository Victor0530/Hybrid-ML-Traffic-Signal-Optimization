# Reproduce the Project — Teammate Quickstart

**Use this file if you only want to run the current system.** It is deliberately
short. `RUNBOOK.md`, `FILES.md` and the milestone reports are reference material,
not required reading before you start.

The current pipeline covers **M1 (network) + M2 (fixed-time baseline and raw
data) + M3 (dataset and EDA)**. The M3 supervised models, M4 RL agents and M5
hybrid are added later; this quickstart will be updated as each milestone lands.

---

## 1. One-time setup (Windows PowerShell)

```powershell
git clone https://github.com/Victor0530/Hybrid-ML-Traffic-Signal-Optimization.git
cd Hybrid-ML-Traffic-Signal-Optimization

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Eclipse SUMO **1.28.0** separately and make sure `SUMO_HOME` is set and
`sumo` is on `PATH`. The Python packages `traci`/`sumolib` must match the SUMO
release.

---

## 2. Reproduce everything with one command

With the virtual environment active, from the repository root:

```powershell
python -m experiments.reproduce
```

The script runs these stages in order and stops on the first real failure:

1. checks Python, packages, SUMO and the network;
2. generates the five route files from `configs/traffic_scenarios.yaml`;
3. collects 20 SUMO runs (5 scenarios × 4 seeds: 42–45), skipping valid
   existing CSVs;
4. runs the M2 fixed-time baseline analysis;
5. rebuilds the M3 dataset, splits, feature map and audit;
6. runs the M3 EDA tables and figures;
7. runs the test suite.

A clean run finishes with something close to:

```text
Rows: 38,400 | runs: 20 | missing cells: 0
...
29 passed
Done in ...
```

The 20 simulations are the slow part. On a typical laptop the first full run is
several minutes; later runs skip the existing CSVs and finish in well under a
minute.

---

## 3. Useful variants

```powershell
# Check the environment only (no simulation, no writes)
python -m experiments.reproduce --check-only

# Data already exists: rebuild the baseline, dataset, EDA and tests only
python -m experiments.reproduce --skip-routes --skip-collection

# Re-run one scenario with one seed
python -m experiments.reproduce --scenarios medium --seeds 42

# Force re-simulation of every selected run
python -m experiments.reproduce --overwrite

# Skip stages you do not need
python -m experiments.reproduce --skip-collection --skip-baseline --skip-tests
```

If a raw CSV exists but its manifest row is missing, its hash does not match the
current config/route/network, or it is incomplete, the script reports it as
invalid. Fix the cause first; use `--overwrite` only when you intend to replace
the run.

---

## 4. Where the outputs go

| Output | Location | Tracked in Git? |
|---|---|---|
| Raw run CSVs and manifest | `data/raw/` | No (regenerated) |
| Route files | `sumo/routes/generated/` | No (regenerated) |
| Fixed-time baseline tables/figures | `results/fixed_time_baseline/` | No |
| Processed dataset and splits | `data/processed/` | No |
| EDA tables/figures | `results/sl/eda/` | No |
| Code, configs and reports | `src/`, `configs/`, `docs/` | Yes |

Everything ignored by Git is reproducible from the command above. Do not commit
raw data, generated routes, model checkpoints or results.

---

## 5. If something fails

| Symptom | Fix |
|---|---|
| `No module named traci` / `yaml` | Activate the virtual environment and reinstall `requirements.txt`. |
| `SUMO could not be found` | Install SUMO 1.28.0, set `SUMO_HOME`, add `%SUMO_HOME%\bin` to `PATH`, reopen the terminal. |
| `configuration changed` / `route changed` / `network changed` | The code or network changed since the raw run was collected. Re-run with `--overwrite` only if you intend to recollect. |
| `Missing SUMO network file` | You are in the wrong folder, or the repository is incomplete. Run from the repo root. |
| Port already in use / TraCI connection error | Another SUMO run is using the default port. Stop it, or run one process at a time. |
| Tests fail after changing shared code | Fix the test failure before collecting new data; do not commit generated outputs. |

---

## 6. Manual commands (if you prefer step by step)

```powershell
python -m src.simulation.scenario_generator --all
python -m src.simulation.data_collector --scenario medium --seed 42
python -m src.analysis.characterise_baseline
python -m src.preprocessing.build_dataset
python -m src.analysis.explore_sl_dataset
python -m pytest -q
```

To collect every run manually, call the collector once per scenario/seed, or use
the orchestrator above. Do not share generated CSVs between machines; rebuild
them so hashes and manifests stay consistent.

---

## 7. Where to read more

- Exact Windows setup, commands and troubleshooting:
  [`../RUNBOOK.md`](../RUNBOOK.md)
- Milestone-ordered file map:
  [`../FILES.md`](../FILES.md)
- M2 fixed-time baseline methodology:
  [`fixed_time_baseline.md`](fixed_time_baseline.md)
- M3 dataset, target, features and splits:
  [`supervised_learning_dataset.md`](supervised_learning_dataset.md)
- M3 EDA findings:
  [`sl_eda.md`](sl_eda.md)
- Assignment scope and milestones:
  [`../PLAN.md`](../PLAN.md)
