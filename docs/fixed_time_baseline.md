# Fixed-Time Baseline Characterisation

Status: complete, 13 November 2026

This report characterises the fixed-time signal controller before any RL or
hybrid controller is trained. It is the M2 input to the RL state, reward, and
action design required by PLAN.md section 6.6.

Related documents:

- `FILES.md` - repository map and milestone status.
- `RUNBOOK.md` - exact commands and expected outputs.
- `results/fixed_time_baseline/` - generated tables and plots (gitignored).

## 1. Fixed-time controller

The network has one four-way junction (`C`) with two incoming lanes per
approach. `netconvert` generated an eight-phase, 90-second fixed-time program:

| Phase | Duration | State | Movement |
|---|---:|---|---|
| 0 | 33 s | `GGgrrrGGgrrr` | North-south through/right, permissive left |
| 1 | 3 s | `yygrrryygrrr` | North-south yellow |
| 2 | 6 s | `rrGrrrrrGrrr` | North-south protected left |
| 3 | 3 s | `rryrrrrryrrr` | North-south left yellow |
| 4 | 33 s | `rrrGGgrrrGGg` | East-west through/right, permissive left |
| 5 | 3 s | `rrryygrrryyg` | East-west yellow |
| 6 | 6 s | `rrrrrGrrrrrG` | East-west protected left |
| 7 | 3 s | `rrrrryrrrrry` | East-west left yellow |

The cycle gives each axis equal green time. This is the fixed-time reference
that RL-only and hybrid controllers will be compared against.

There is **no separate all-red phase**. Each movement ends with a 3-second
yellow clearance and the conflicting axis starts on the next phase.

Lane-to-movement mapping:

- Lane 0 (right lane) serves straight and right turns.
- Lane 1 (left lane) serves the protected left turn.
- Left turns enter outgoing lane 1; straight and right turns enter outgoing lane 0.

The 12-character TLS state follows the network link order: north right, north
straight, north left, east right, east straight, east left, south right, south
straight, south left, west right, west straight, west left. `G` is green, `g`
is permissive green, `y` is yellow, and `r` is red.

Vehicle type `car` uses SUMO's default Krauss car-following model with
`accel=2.6`, `decel=4.5`, `sigma=0.5`, `length=5.0`, `minGap=2.5`, and
`maxSpeed=13.89 m/s`. Vehicles are inserted with `departLane="best"` and
`departSpeed="max"`. `speedFactor` is left at SUMO's default distribution,
which is why an individual observed mean speed can exceed 13.89 m/s.

## 2. Dataset and quality

The analysis uses the canonical 20-run dataset: five scenarios times four seeds
(42-45), one simulated hour per run. The loader validates every run's manifest
configuration, route, and canonical network hashes, and requires exactly this
scenario-by-seed set; stale or mixed data is rejected instead of being analysed.

| Metric | Value |
|---|---:|
| Runs | 20 |
| Observation rows | 57,600 |
| Timestamps per run | 720 (5 s to 3,600 s) |
| Approaches per timestamp | 4 |
| Rows per run | 2,880 |
| Missing values | 0 |
| Missing `mean_speed` values | 0 |
| Zero-vehicle rows | 0 |
| `no_vehicles` rows | 0 |
| `sensor_missing` rows | 0 |
| Duplicate `(run_id, simulation_time, direction)` keys | 0 |
| Unmapped arrivals | 0 |
| Observed `mean_speed` range | 0.028-16.106 m/s |
| Observed `occupancy` range | 0.005-0.403 |

Notes:

- `mean_speed` can exceed the 13.89 m/s speed limit because SUMO applies a
  per-vehicle `speedFactor`; it is not a data error.
- `occupancy` is a fraction from 0 to 1, not a percentage.
- `waiting_time` is the sum of the current waiting times of all vehicles on the
  approach, not a per-vehicle average.
- All runs report `status=ok`, `sumo_version=1.28.0`, and zero teleports.
- The empty-approach and sensor-missing branches exist but are not exercised by
  this dataset. A controlled sensor-loss test belongs in M3 and must only alter
  validation/test features, never fabricate dirty training data.

### Observation dtypes

| Column | Type | Unit / values |
|---|---|---|
| `run_id` | string | `{scenario}_seed{seed}` |
| `scenario` | categorical | `low`, `medium`, `high`, `uneven`, `changing` |
| `seed` | integer | SUMO seed |
| `simulation_time` | integer | simulated seconds, multiples of 5 |
| `direction` | categorical | `north`, `south`, `east`, `west` |
| `vehicle_count` | integer | vehicles on the approach |
| `mean_speed` | float, nullable | m/s; blank when no vehicles or sensor missing |
| `queue_length` | integer | halting vehicles |
| `waiting_time` | float | seconds; sum over current vehicles |
| `occupancy` | float | fraction 0-1 |
| `traffic_light_phase` | integer | TLS phase index 0-7 |
| `phase_duration` | float | programmed seconds |
| `time_in_phase` | float | elapsed seconds |
| `no_vehicles` | integer | 0 or 1 |
| `sensor_missing` | integer | 0 or 1 |
| `departed_count` | integer | vehicles inserted during the 5 s interval |
| `arrived_count` | integer | vehicles that completed trips during the interval |

## 3. Method

The analysis script is `src/analysis/characterise_baseline.py`.

- Samples are grouped into 5-minute bins aligned as `[0, 300)`, `[300, 600)`,
  and so on, so each bin contains 60 five-second samples.
- Queue, waiting time, speed, and vehicle count are averaged across seeds and
  samples within each bin.
- Per-approach throughput comes from the collector's `arrived_count` column:
  arrivals in a 5-minute bin are scaled to vehicles per hour.
- "First 10 min" is `t <= 600 s`; "last 10 min" is `t > 3,000 s`.
- `saturated_end` is flagged when the last-10-minute mean queue is at least
  15 vehicles and mean speed is at most 3.5 m/s.
- `recovery_drop` compares the last full 5-minute bin of the changing
  scenario's peak phase (2,400-2,700 s) with its final bin (3,300-3,600 s).
  The bins are derived from the configured phase boundaries, so they remain
  correct if the bin width or phase times change.

## 4. Scenario results

| Scenario | Demand (veh/h/approach) | Queue mean | Queue last 10 min | Waiting mean (s) | Speed mean (m/s) | Throughput (veh/h/approach) | Completed-trip ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 300 | 1.35-1.40 | 1.35-1.49 | 25.0-25.8 | 9.65-9.79 | 291.5-293.0 | 97.2-97.7% |
| medium | 600 | 3.03-3.22 | 2.98-3.28 | 54.4-58.9 | 8.52-8.76 | 580.5-585.3 | 96.8-97.5% |
| high | 700 | 4.10-4.25 | 4.14-4.58 | 75.9-80.0 | 7.99-8.08 | 676.0-683.3 | 96.6-97.6% |
| uneven - north | 800 | 22.71 | 37.70 | 441.6 | 3.06 | 705.0 | 88.1% |
| uneven - west | 600 | 3.06 | 2.99 | 55.3 | 8.73 | 580.5 | 96.8% |
| uneven - south/east | 250 | 1.08-1.20 | 0.88-1.46 | 19.7-22.4 | 9.81-10.14 | 244.3-245.0 | 97.7-98.0% |
| changing (whole hour) | 400 -> 800 -> 400 | 4.21-4.77 | 1.84-2.14 | 87.0-100.8 | 7.78-8.21 | 488.5-492.0 | 97.7-98.4% |

The ranges cover the four approaches. Full per-approach values are in
`results/fixed_time_baseline/baseline_summary.csv`.

The **completed-trip ratio** is `arrived / demand` over the hour. It excludes
vehicles still running at `t=3600` and is therefore not the same as stop-line
detector throughput.

### High demand is now calibrated

The earlier high scenario used 800 veh/h per approach. At that level the
straight/right lane demand was about 680 veh/h against roughly 660 veh/h of
green capacity, so queues grew to 36-49 vehicles and never recovered.

At 700 veh/h per approach (2,800 veh/h total), the last-10-minute queue is
4.1-4.6 vehicles and mean speed is about 8.0 m/s. The scenario is near capacity
but stable, which is the intended "high" condition.

### Uneven demand remains an intentional stress case

The uneven scenario keeps north at 800 veh/h. North is over capacity and grows
from a 6.7-vehicle mean in the first 10 minutes to 37.7 vehicles in the last
10 minutes, while south and east stay near one vehicle. This is deliberate:
it tests whether an adaptive controller can reallocate green time toward the
busy approach. It is an asymmetric stress case, not a calibration failure.

## 5. Build-up, recovery, saturation, and starvation

### Build-up

- Low and medium demand reach a small steady queue within the first minutes.
- High demand stays stable at about 4-5 vehicles per approach.
- Uneven demand builds continuously on north only.
- Changing demand builds during the 1,800-2,700 s high phase and then falls.

### Recovery

The changing scenario now includes a final 400 veh/h phase. The 5-minute mean
queue peaks at 12.3-15.0 vehicles in the 2,400-2,700 s bin, then falls to
1.8-2.1 vehicles in the last 10 minutes. The recovery drop is 10.4-12.8
vehicles per approach. This confirms that the scenario can test both the
response to a demand step and the ability to discharge the resulting queue.

### Saturation

Only uneven-north is flagged `saturated_end`. High at 700 veh/h is not
saturated, and changing recovers by the end. The earlier high-at-800 dataset
was permanently saturated and has been replaced.

### Starvation

No approach is starved of green: the fixed-time plan gives each axis the same
33 s through phase and 6 s protected-left phase per 90 s cycle. Under uneven
demand, south and east are underloaded rather than starved, while north is
over capacity. This is an allocation mismatch: equal green is not equal
service when demand is asymmetric.

No timestamp has zero vehicles on an approach, so the `no_vehicles` branch was
not exercised. The 5-second sampling interval also aliases the 3-second yellow
and 6-second protected-left phases; some short phases may not appear in the
sample even though SUMO executed them. `traffic_light_phase`,
`phase_duration`, and `time_in_phase` retain the phase context.

## 6. Implications for RL design

### State

- Use per-approach queue, waiting time, mean speed, and vehicle count, not only
  network totals. The uneven scenario shows that a single aggregate would hide
  the north queue behind the empty south and east approaches.
- Include the current phase and time in phase so the agent can reason about
  upcoming yellow and protected-left transitions.
- The changing scenario motivates recent history or the SL 5-minute-ahead
  predictions; the current state alone cannot anticipate the step change.
- Keep the state compact enough for tabular Q-Learning as well as DQN.

### Reward

- A weighted sum of total queue and total waiting time remains appropriate.
- Add a max-queue or fairness term if the project wants to prevent the uneven
  case from being ignored; a pure network total can trade a long north queue
  against several short queues elsewhere.
- Do not optimise throughput alone: high throughput can coexist with long
  queues, and the uneven scenario already serves most demand while one
  approach is saturated.

### Action

- Phase selection plus optional green-duration control is justified by the
  uneven and changing scenarios.
- The fixed-time controller is a strong reference at low, medium, and high
  demand; the RL agent must at least match it there before claiming a benefit.
- The clear opportunity is asymmetric demand: shift green toward north in the
  uneven scenario and react earlier in the changing scenario.

### Evaluation

- Use low and medium as sanity checks, high as the near-capacity condition,
  uneven as the asymmetric stress test, and changing as the recovery test.
- Report queue, waiting time, speed, throughput, and recovery time together.
- Keep the fixed-time baseline numbers in this report as the M2 reference.

## 7. Limitations

- Only four seeds per scenario are available, so seed variability is reported
  but confidence intervals will be wide.
- Throughput is derived from interval departure/arrival counts; it is not a
  detector-level measurement.
- The current data has no sensor-loss or empty-approach examples.
- The fixed-time controller is a generated 90-second program, not a separately
  tuned benchmark.
- The 5-second sampling interval cannot resolve every short signal phase.

## 8. Reproduce

```powershell
python -m src.analysis.characterise_baseline
```

Outputs:

```text
results/fixed_time_baseline/baseline_summary.csv
results/fixed_time_baseline/baseline_timeseries.csv
results/fixed_time_baseline/data_quality_summary.csv
results/fixed_time_baseline/queue_timeseries.png
results/fixed_time_baseline/waiting_timeseries.png
results/fixed_time_baseline/speed_timeseries.png
results/fixed_time_baseline/throughput_timeseries.png
results/fixed_time_baseline/throughput_by_approach.png
```

The script prints the run count, data-quality table, and baseline summary to
the console.
