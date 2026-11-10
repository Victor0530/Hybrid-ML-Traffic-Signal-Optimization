# Hybrid SL + RL Traffic Signal Optimisation

> **Project title:** An Intelligent Hybrid Supervised and Reinforcement
> Learning-Based Adaptive Traffic Signal Optimisation System\
> **Status:** Project plan / implementation specification\
> **Primary language:** Python\
> **Simulation environment:** Eclipse SUMO\
> **Integration interface:** TraCI\
> **Initial scope:** One four-way intersection

------------------------------------------------------------------------

## 1. Project Context

This project develops and evaluates an intelligent traffic signal
controller that combines:

-   **Supervised Learning (SL):** predicts short-term future traffic
    volume from historical and current traffic observations.
-   **Reinforcement Learning (RL):** learns how to control traffic
    signal phases and green durations through interaction with a traffic
    simulation.
-   **Hybrid SL + RL:** supplies the SL model's traffic predictions as
    additional observations to an RL controller, allowing it to consider
    both current and anticipated traffic.

The project is simulation-based. SUMO (Simulation of Urban Mobility)
will model individual vehicles, roads, junctions, routes and traffic
lights. Python will collect observations, train models, control the
simulation and evaluate results.

### Core research question

**Does adding short-term traffic-volume predictions from a Supervised
Learning model improve Reinforcement Learning-based traffic signal
control compared with RL-only control and conventional fixed-time
control?**

The hybrid system is a hypothesis to test, not an assumed winner.
Results must be reported honestly, including cases where prediction does
not improve performance.

------------------------------------------------------------------------

## 2. Assignment Goals

The project should demonstrate a complete applied machine-learning
workflow:

1.  Define a real-world problem and a measurable ML approach.
2.  Build or configure a suitable dataset and simulation environment.
3.  Explore the data and environment.
4.  Prepare features and state representations.
5.  Select and implement appropriate SL and RL algorithms.
6.  Train and evaluate multiple models.
7.  Optimise selected models and compare before/after results.
8.  Integrate the best SL and RL components.
9.  Test the final systems under consistent traffic scenarios.
10. Interpret, visualise and discuss the results.

### Proposed objectives

1.  Develop and evaluate Supervised Learning regression models for
    short-term traffic-volume prediction.
2.  Develop and evaluate Reinforcement Learning agents for adaptive
    traffic signal control.
3.  Optimise selected SL models and RL agents based on initial
    evaluation.
4.  Integrate the best optimised SL model with the best optimised RL
    approach to form a hybrid controller.
5.  Compare fixed-time, RL-only and hybrid control under different
    traffic demand conditions.

------------------------------------------------------------------------

## 3. Scope and Assumptions

### Initial scope

-   One four-way intersection.
-   Four incoming approaches: North, South, East and West.
-   A configurable traffic-light program with safe, non-conflicting
    phases.
-   Simulated vehicles and generated traffic demand.
-   Short-term traffic-volume prediction, initially targeting the number
    of vehicles expected in the next five minutes.
-   Adaptive signal decisions made at defined decision intervals.
-   Comparison against a fixed-time signal baseline.

### Keep the scope manageable

-   Start with one intersection only.
-   Do not add multiple intersections, pedestrians, emergency-vehicle
    priority or multi-agent RL until the core system works.
-   Do not introduce complex deep-learning forecasting unless simpler
    models have been implemented and evaluated first.
-   Do not claim the simulation represents real Malaysian traffic unless
    the network, demand and assumptions are calibrated with appropriate
    real data.

### Important design clarification

The initial RL-only experiments are developed **without SL
predictions**. The RL agent first learns from current traffic
observations. After independent SL and RL evaluation and optimisation,
the selected SL model and selected RL algorithm are integrated. The
hybrid RL agent must then be trained or fine-tuned with the expanded
state that includes SL predictions. Do not simply append new state
features to an already-trained agent and assume it will work.

------------------------------------------------------------------------

## 4. Recommended Technology Stack

  -----------------------------------------------------------------------
  Tool / library                      Purpose
  ----------------------------------- -----------------------------------
  Python                              Main implementation language

  Eclipse SUMO                        Microscopic traffic simulation

  TraCI / sumolib                     Python-SUMO communication and
                                      network utilities

  Pandas                              Dataset preparation and analysis

  NumPy                               Numerical operations

  Matplotlib / Seaborn                Charts and visualisation

  Scikit-learn                        SL models, preprocessing, metrics
                                      and tuning

  XGBoost (optional)                  Gradient-boosted regression model

  PyTorch (if DQN/PPO selected)       Neural-network-based RL
                                      implementation

  Gymnasium (optional)                Standardise a custom RL environment

  Streamlit (optional)                Demonstration dashboard
  -----------------------------------------------------------------------

Use compatible, documented package versions. Check whether SUMO and
TraCI are installed and accessible before writing simulation code. Do
not assume SUMO is installed or that its executable is already on PATH.

### 4.1 Critical evaluation of current technologies

The report must critically evaluate existing approaches, not just list
tools. Summarise and compare:

-   **Fixed-time control:** simple, predictable and easy to audit, but
    cannot respond to changing demand; it is the required baseline.
-   **Actuated control:** responds to detector data using hand-written
    rules, but has limited anticipation and can be brittle under unusual
    demand patterns.
-   **RL-based adaptive control:** learns a policy from interaction and
    can use rich state, but is sample-inefficient and sensitive to
    reward design, state representation and training stability.
-   **SL traffic forecasting:** regression on recent observations can
    provide short-term anticipation, but prediction errors can propagate
    into control decisions.
-   **Hybrid SL + RL:** uses predictions as additional state; the
    research question is whether this improves control, not an assumed
    outcome.

Justify the chosen stack (SUMO, TraCI, scikit-learn, PyTorch) against
alternatives. For example, SUMO is open-source, microscopic,
Python-controllable and reproducible; CityFlow is faster but less
feature-rich, while VISSIM is commercial. Justify regression rather than
classification because the target is a continuous vehicle count, and
justify Q-Learning/DQN because they suit discrete signal actions and
differ in their ability to handle continuous state.

Use references and report this comparison in the related-work section.

------------------------------------------------------------------------

## 5. Overall System Workflow

### Phase A --- Data and simulation setup

1.  Create a four-way SUMO network.
2.  Define valid traffic-light phases and transitions.
3.  Generate vehicle routes and traffic demand.
4.  Run simulation scenarios.
5.  Extract traffic observations at a consistent sampling interval.
6.  Save reproducible datasets and scenario configurations.

### Phase B --- Supervised Learning

1.  Define the prediction target.
2.  Explore and preprocess the data.
3.  Train multiple SL regression models.
4.  Evaluate all models using the same held-out test data.
5.  Select the top two candidates for tuning, based on validation
    results.
6.  Optimise those candidates.
7.  Re-evaluate them on the same untouched test set.
8.  Select the best optimised SL model for hybrid integration.

### Phase C --- Reinforcement Learning

1.  Define the RL environment, observations, actions and reward.
2.  Implement a fixed-time controller as a baseline.
3.  Train multiple RL algorithms independently of SL predictions.
4.  Evaluate agents using repeated simulation runs.
5.  Select the top two candidates for tuning, based on validation
    scenarios.
6.  Optimise those candidates.
7.  Re-evaluate them using consistent test scenarios and multiple seeds.
8.  Select the best optimised RL approach for hybrid integration.

### Phase D --- Hybrid integration

1.  Load the selected optimised SL model.
2.  At each RL decision step, collect current traffic observations.
3.  Use the SL model to predict future traffic volume.
4.  Add predictions to the RL observation/state.
5.  Train or fine-tune the selected RL algorithm using this expanded
    state.
6.  Evaluate the hybrid controller.

### Phase E --- Final comparative testing

Compare:

1.  Fixed-time control.
2.  Optimised RL-only control.
3.  Hybrid SL + RL control.

Use the same traffic demand scenarios, simulation duration, vehicle
demand and random seeds wherever possible. Report averages and
variability across repeated runs.

------------------------------------------------------------------------

## 6. Simulation and Data Generation

### 6.1 SUMO network

Create a simple four-way intersection with four incoming approaches.
Document:

-   Road and lane layout.
-   Lane count and lane length.
-   Speed limits.
-   Allowed movements (straight, left and right, if supported).
-   Traffic-light phases.
-   Yellow and all-red clearance intervals, where applicable.
-   Vehicle types and car-following assumptions.

Use a valid traffic-light phase plan. Never allow conflicting movements
to receive green simultaneously. Enforce minimum and maximum green times
and safe transitions. The RL action should select only valid actions; it
must not bypass traffic-signal safety rules.

### 6.2 Traffic demand scenarios

Create reproducible demand profiles, for example:

-   Low demand.
-   Medium demand.
-   High demand.
-   Uneven demand, where one approach is much busier.
-   Time-varying demand or a sudden demand increase.

The exact vehicle rates should be configurable and selected after
checking that the network is neither empty nor permanently gridlocked.
Treat any example rates as experimental settings, not real-world facts.

Use separate random seeds for scenario generation. Save the seed and
configuration for every run.

### 6.3 Data to collect

Collect at a consistent interval (for example, every 5 or 10 simulation
seconds; choose and document one interval):

  -----------------------------------------------------------------------
  Field                               Meaning
  ----------------------------------- -----------------------------------
  simulation_time                     SUMO simulation time

  direction                           North, South, East or West

  vehicle_count                       Vehicles in the defined approach or
                                      detection region

  mean_speed                          Mean speed of vehicles in the
                                      defined region

  queue_length                        Defined count or length of queued
                                      vehicles

  waiting_time                        Clearly defined accumulated or
                                      interval waiting-time measure

  occupancy / density                 Optional traffic occupancy or
                                      density measure

  traffic_light_phase                 Current active phase

  phase_duration / time_in_phase      Optional phase timing information
  -----------------------------------------------------------------------

Define precisely how each feature is measured. For example, do not mix a
count over an entire incoming road with a queue count over only a short
detector region without documenting the difference.

### 6.4 Dataset generation

Create a repeatable script that:

-   Reads a scenario configuration.
-   Starts SUMO in GUI or headless mode.
-   Runs the simulation.
-   Extracts observations.
-   Writes structured CSV files.
-   Records scenario name, seed, simulation configuration and sampling
    interval.
-   Handles simulation shutdown cleanly, including errors.

Keep raw generated data separate from processed data.

Suggested files:

-   `data/raw/traffic_observations.csv`
-   `data/processed/sl_features.csv`
-   `data/scenarios/` for scenario configurations

Avoid committing huge generated datasets or model checkpoints to version
control unless required. Document how to regenerate them.

### 6.5 Data quality and missing-data handling

SUMO can produce a complete set of rows, but the supervised-learning
dataset still contains structural missingness. Handle it explicitly.

**Sources of missingness**

-   An empty approach has no meaningful mean speed. TraCI commonly
    returns a sentinel such as `-1` for an empty lane or edge; verify the
    behaviour for the installed SUMO version. Do not treat this as a
    real speed.
-   Vehicle count and queue length of zero are valid observations, not
    missing values. Do not drop zero-vehicle rows.
-   The last prediction horizon of each simulation run has no valid
    five-minute-ahead target. The first rows have no lag history.
-   Failed or partial runs, schema mismatches and CSV gaps are
    operational failures. Mark them as failed and exclude them; do not
    impute an entire run.

**Collection policy**

-   Sample on a fixed schedule and write one row per
    `(simulation_time, direction)` even when no vehicles are present.
-   Convert sentinel values to `NaN` and add explicit flags such as
    `no_vehicles` and `sensor_missing`. Do not rely only on automatic
    missing-value indicators, because the training split may contain no
    missing values.
-   Record a run manifest with `run_id`, scenario, seed, status, number
    of steps and sampling interval.

**Dataset construction**

-   Build the target within each run, for example
    `target = groupby(run_id).vehicle_count.shift(-HORIZON_STEPS)`,
    where `HORIZON_STEPS = horizon_seconds / sampling_interval`.
-   Drop rows whose target is undefined at the end of a run, and drop or
    explicitly mark the warm-up rows without lag features.
-   Never impute the target. Only feature columns may be imputed.
-   Split chronologically before fitting any preprocessing step.

**Preprocessing policy**

-   Fit imputers, scalers and encoders on the training data only, then
    transform validation and test data with the fitted objects.
-   Use a documented per-column strategy: for example, median imputation
    plus a `no_vehicles` flag for mean speed; no imputation for count or
    queue length; causal forward-fill only where justified.
-   Keep the same preprocessing pipeline for SL training, hybrid
    inference and any dashboard or demonstration.
-   Save the fitted preprocessing pipeline with the selected model.

**Data-quality audit**

Produce a short audit before modelling:

-   missing count and percentage per column;
-   timestamp continuity and duplicate-row checks;
-   frequency of zero-vehicle intervals by direction;
-   target availability per run;
-   number of failed or excluded runs.

**Optional robustness experiment**

Keep the clean test set as the headline result. On a copy of the
validation/test features only, inject missingness (for example 5% and
10% missing completely at random, plus a block dropout simulating a
failed detector), compare simple strategies such as dropping rows,
median imputation, forward-fill and k-nearest-neighbour imputation, and
report the change in MAE/RMSE. Do not inject missingness into the main
training data or fabricate a dirty dataset. State clearly in the report
that SUMO output is complete by construction and that the experiment
tests robustness to sensor/feature loss.

### 6.6 Environment dynamics characterisation

Before RL training, characterise the environment under the fixed-time
controller:

-   Run the fixed-time baseline across low, medium, high, uneven and
    changing demand.
-   Record queue length, waiting time, mean speed and throughput by
    approach over time.
-   Check that demand is calibrated: the network should be neither empty
    nor permanently gridlocked.
-   Observe queue build-up and recovery, saturation behaviour and
    whether any approach is starved.
-   Document the findings and use them to design the state, reward and
    action space, and to sanity-check RL behaviour.

This is exploratory analysis, not the final controller comparison.

------------------------------------------------------------------------

## 7. Supervised Learning Specification

### 7.1 Task definition

This is a **regression** task.

-   **Inputs:** recent/current traffic observations.
-   **Target:** future vehicle count, initially five minutes ahead.
-   **Prediction granularity:** preferably one target per approach
    (North, South, East, West), or a clearly defined total. Choose one
    approach and use it consistently.

A useful formulation is to create one row per observation time with
features for each direction and a target for the vehicle count five
minutes later. Alternatively, build one model per direction. Document
the chosen design.

### 7.2 Candidate models

Start with 3--4 individual models:

1.  Linear Regression --- simple baseline.
2.  Decision Tree Regressor.
3.  Random Forest Regressor.
4.  XGBoost Regressor (optional).

Then add one combined model:

5.  Ensemble / blended regressor --- a deliberate combination of the
    top two tuned models (for example, averaging their predictions or
    stacking them with a simple meta-learner).

The ensemble is not an extra algorithm added to increase the model
count. It is the model-combination experiment required by the brief:
combine models to improve accuracy and reliability. Keep the blend
simple, evaluate it on the same validation and untouched test data as
the individual models, and report whether it actually improves results.
Do not add many algorithms just to increase the model count. Prioritise
correct evaluation and meaningful comparison.

### 7.3 Data leakage prevention

Because the data is time-dependent:

-   Split data chronologically into training, validation and test sets.
-   Do not randomly mix future observations into training.
-   Fit scalers, imputers and feature transformations on training data
    only; see §6.5 for the missing-data policy.
-   Ensure each target is genuinely five minutes after its input
    observation.
-   Avoid using any feature that would only be known after the
    prediction time.
-   Keep the final test set untouched during model selection and tuning.

Use time-series-aware validation, such as chronological validation or
`TimeSeriesSplit`, where appropriate. Do not use ordinary shuffled
cross-validation for a time-ordered prediction problem.

### 7.4 Initial training and evaluation

Train all selected candidate models. Evaluate them on validation data
for model selection and report final performance on the held-out test
data.

Metrics:

-   MAE --- lower is better.
-   RMSE --- lower is better.
-   R² --- higher is generally better, but interpret carefully.

Include a simple baseline predictor (for example, persistence: future
count equals the latest observed count) if practical. This helps
establish whether ML improves on a naive forecast.

### 7.5 Optimisation strategy

Recommended workload:

1.  Train and evaluate all initial candidates.
2.  Rank candidates using validation metrics, prioritising MAE/RMSE and
    considering R², interpretability and robustness.
3.  Select the top two candidates for hyperparameter optimisation.
4.  Tune with a reasonable search budget (Grid Search or Random Search).
5.  Compare initial and tuned validation performance.
6.  Build a simple blend or stack of the top two tuned models and
    compare it with the best individual model on validation data.
7.  Select the best tuned model or blend.
8.  Evaluate the selected model once on the untouched test set.

Not every model requires complex tuning. Linear Regression may have
little or no useful hyperparameter tuning; do not invent tuning steps
just to make the table look complete.

Keep a record of: - Model name. - Initial parameters. - Search space. -
Best parameters. - Validation metrics before and after tuning. - Final
test metrics. - Training time, if useful.

### 7.6 SL outputs for hybrid use

The selected SL model should expose a clear prediction function, such
as:

`predict_future_traffic(current_observation_window)`

The output must have a documented shape and ordering, for example:

`[predicted_north_count, predicted_south_count, predicted_east_count, predicted_west_count]`

Ensure the feature preparation used at inference is identical to
training. Save the full preprocessing pipeline together with the model
where possible.

If the selected model is an ensemble, the prediction function must
encapsulate the full blend and preprocessing pipeline, so callers see a
single model interface.

### 7.7 Interpretability and key drivers

Explain what the models learn, not only how accurate they are:

-   For Linear Regression, report standardised coefficients and the
    direction of each effect.
-   For tree models, report impurity-based and permutation importance;
    use partial dependence or ICE plots for the most influential
    features.
-   Compare feature importance across models and discuss whether the
    same variables consistently matter.
-   For RL, analyse action distributions conditioned on state (for
    example queue imbalance, current phase and time in phase) and use
    state-feature occlusion or perturbation to show which observations
    influence decisions.
-   Use these findings to refine features and state representations
    (§7.9) and to explain results in the report.

Interpretability is also a model-selection criterion: prefer a model
with similar accuracy but clearer, more stable explanations when the
choice is close.

### 7.8 Prediction error analysis

Go beyond aggregate MAE/RMSE:

-   Residuals by direction, demand level, prediction horizon and time
    period.
-   Over- and under-prediction bias; worst-case errors and their causes.
-   Error distributions and whether errors cluster in particular
    conditions, such as sudden demand changes, empty approaches or
    heavily saturated approaches.
-   Comparison of error patterns across candidate models.
-   Discussion of generalisation, adaptability and potential bias, for
    example a model that favours the busiest approach or performs poorly
    on the minor road.

Use the findings to propose concrete improvements to features,
representations or model structure. Do not report only the mean error.

### 7.9 Feature selection and representation ablation

Run a small, controlled ablation using the same chronological splits,
model and metrics:

-   F0: current vehicle counts only.
-   F1: F0 plus lagged counts and rolling means.
-   F2: F1 plus queue length and mean speed.
-   F3: F2 plus signal phase and time-in-phase features.
-   F4: F3 plus derived directional-imbalance features, if justified.

Report the incremental effect on MAE/RMSE/R² and choose the final
feature set based on validation performance and interpretability. Avoid
adding features that do not improve held-out performance.

For RL, ablate the state representation:

-   S0: current counts, queues, speeds and signal phase.
-   S1: S0 plus time elapsed in the current phase.
-   S2: S1 plus SL predictions (the hybrid state).

Train and evaluate each variant with the same algorithm, scenarios and
seeds. This separates the effect of the extra state from the effect of
the RL algorithm and directly supports the hybrid comparison.

### 7.10 Statistical reliability

Do not rely on single point estimates:

-   Report bootstrap 95% confidence intervals for MAE, RMSE and R² on
    the test set.
-   When comparing models, use paired comparisons over the same
    chronological folds or runs where appropriate.
-   Report the number of runs/seeds and the variability, not only the
    best result.
-   For RL, report learning-curve stability, convergence behaviour and
    variance across seeds (see §10.6).

### 7.11 Cross-distribution robustness

Test whether the SL model generalises beyond the demand profile used for
training:

-   Train on low/medium demand and test on high, uneven and changing
    demand; also test the reverse direction where data permits.
-   Train on a mixture of scenarios and evaluate per scenario.
-   Fit all preprocessing on the training distribution only and report
    degradation relative to in-distribution performance.

Discuss which conditions cause the largest errors and link the findings
to §7.8. For RL, evaluate on held-out scenarios and report recovery time
after a demand change: the time needed to return to a defined baseline
queue or waiting level. This addresses adaptability over time, not just
average performance.

------------------------------------------------------------------------

## 8. Reinforcement Learning Specification

### 8.1 Environment

SUMO is the RL environment. Python communicates with it through TraCI.
Implement a wrapper that has a clear reset/step interface, optionally
compatible with Gymnasium.

The environment should support:

-   Resetting to a known scenario and seed.
-   Returning observations.
-   Applying an action safely.
-   Advancing the simulation by a defined interval.
-   Returning reward, termination status and useful diagnostic
    information.

### 8.2 RL agent and state

For the initial **RL-only** experiments, use current traffic
observations only. Example state features:

-   Current vehicle count per approach.
-   Current queue length per approach.
-   Current mean speed per approach.
-   Current signal phase.
-   Time elapsed in the current phase, if useful.

Normalise or scale state features when required by the chosen algorithm.
Keep the state definition consistent across RL-only training and
evaluation. The state representation will be ablated as described in
§7.9; keep the same feature ordering and definitions across training,
evaluation and hybrid inference.

### 8.3 Actions

Use a small, safe action space initially. Possible design:

-   Select one of the predefined valid signal phases.
-   Optionally choose from a small set of allowed green durations.

For example, actions could represent a valid phase with a selected green
duration. Choose phase definitions and duration options based on the
network. Yellow and clearance transitions must be handled by the
environment, not left to the agent to violate.

Avoid an unnecessarily large action space for the first implementation.

### 8.4 Reward function

A starting reward may penalise total queue length and waiting time:

`reward = -(alpha * total_queue + beta * total_waiting_time)`

Where `alpha` and `beta` are configurable weights.

Important:

-   Define the units and time interval for each component.
-   Consider normalising terms so one does not dominate merely due to
    scale.
-   Avoid double-counting waiting time if the chosen measure is
    cumulative.
-   Log reward components separately.
-   Tune reward weights only with validation scenarios.
-   Do not alter the reward definition between controllers without
    explaining why.

The reward should encourage reduced congestion while preserving safe
signal operation. Throughput can be used as an additional reward term
only if justified and tested.

### 8.5 Candidate algorithms

Start with two algorithms:

1.  Q-Learning --- a simple baseline, likely requiring a discretised
    state.
2.  DQN --- handles a richer numerical state using a neural network.

PPO may be an optional third algorithm if time, compute and
implementation experience permit. Do not attempt all algorithms before
the simulation environment is stable.

For a fair comparison, document the state/action definitions and explain
any algorithm-specific adaptations. Q-Learning may require
discretisation, while DQN uses numeric observations.

### 8.6 Initial training and evaluation

Train each RL algorithm using current observations only. Record:

-   Episode return / cumulative reward.
-   Average waiting time.
-   Average queue length.
-   Average travel time.
-   Throughput (vehicles completed or passed through, with a precise
    definition).
-   Training steps and episode duration.

Evaluate trained agents on validation scenarios not used for training.
RL is stochastic, so use multiple training seeds and multiple evaluation
runs where feasible. Report mean and standard deviation, not just the
single best run. Also report learning-curve stability, convergence
behaviour and variance across seeds (see §10.6). Log diagnostics for RL
inefficiencies such as unnecessary phase switches, green time with no
queued vehicles and starvation of minor approaches (see §8.8).

### 8.7 RL optimisation strategy

Recommended workload:

1.  Train all initial candidate agents.
2.  Evaluate on common validation scenarios and seeds.
3.  Select the top two candidates for optimisation, if two candidates
    are available and meaningful.
4.  Tune appropriate parameters using a manageable search.
5.  Re-train agents with selected parameters.
6.  Compare before/after tuning on the same validation scenarios.
7.  Select the best optimised RL algorithm.
8.  Report its final performance on held-out test scenarios.

Potential parameters:

-   Learning rate.
-   Discount factor.
-   Exploration schedule / epsilon (Q-Learning).
-   Batch size and replay buffer size (DQN).
-   Target-network update interval (DQN).
-   Network size (DQN).
-   Reward weights.
-   Green-duration choices, if treated as a design parameter.

Do not select an RL model solely by cumulative reward if reward
definitions differ. Prioritise traffic outcome metrics such as waiting
time, queue length, travel time and throughput. Cumulative reward is
useful within a consistent reward setup.

### 8.8 RL diagnostics, error and inefficiency analysis

Log and analyse:

-   Phase-switch frequency and unnecessary switches (phase chatter).
-   Green time allocated to approaches with no queued vehicles (wasted
    green).
-   Starvation of minor approaches and maximum waiting time per
    direction.
-   Action saturation, repeated actions and sensitivity to the reward
    weights.
-   Reward-component balance to check that one term does not dominate.

Compare these diagnostics with the fixed-time baseline and use them to
explain performance differences. Report them alongside the headline
traffic metrics.

### 8.9 State-representation ablation

Use the S0/S1/S2 variants from §7.9. Train each variant with the same
algorithm, reward, scenarios and seeds, and evaluate on the same
validation scenarios. Report whether phase elapsed time and SL
predictions add value, and whether any gain is consistent across demand
conditions. This is the controlled evidence for the hybrid design.

------------------------------------------------------------------------

## 9. Hybrid SL + RL Integration

### 9.1 Selected components

Only the best-performing **optimised** SL model and best-performing
**optimised** RL algorithm are used as the starting components for the
main hybrid system. The other candidates remain part of the independent
comparison experiments; they do not all need to be integrated.

### 9.2 Expanded hybrid state

The hybrid agent receives:

-   Current traffic observations.
-   SL-predicted future vehicle counts for the approaches.
-   Any required signal-phase information.

Conceptually:

`hybrid_state = current_traffic_state + predicted_future_traffic`

Ensure the prediction is aligned with the correct direction and time
horizon.

### 9.3 Hybrid training requirement

The RL agent's input/state representation changes when predictions are
added. Therefore:

-   Initialise the selected RL algorithm appropriately for the expanded
    state.
-   Train or fine-tune a hybrid agent using observations that include
    predictions from the selected SL model.
-   Do not claim that the original RL-only agent automatically
    understands the new features.
-   Keep a distinct checkpoint and experiment record for the
    hybrid-trained agent.

The SL model can initially be frozen during hybrid RL training. This is
simpler and makes the experiment easier to interpret. Retraining the SL
model jointly with RL is out of scope unless there is a strong reason.

### 9.4 Avoid information leakage

During hybrid training and testing, the SL model must only receive
information available at that simulation time. Do not feed future
ground-truth vehicle counts to the RL agent. The RL agent receives the
SL model's prediction, not the actual future count.

Use the same prediction model and preprocessing pipeline in training and
evaluation. Do not train the hybrid agent on perfect future values and
test it on noisy predictions.

### 9.5 Optional diagnostic experiment

If time permits, compare:

-   RL-only agent.
-   Hybrid agent using real SL predictions.
-   Optional diagnostic upper bound using ground-truth future traffic
    (clearly labelled as an unrealistic oracle, not a deployable
    system).

The oracle experiment is optional and must never be presented as the
proposed real system.

------------------------------------------------------------------------

## 10. Final Evaluation Design

### 10.1 Systems to compare

  -----------------------------------------------------------------------
  System                              Controller
  ----------------------------------- -----------------------------------
  Fixed-time baseline                 Predefined signal cycle and timings

  RL-only                             Selected optimised RL approach
                                      trained with current observations

  Hybrid SL + RL                      Selected optimised SL predictor +
                                      hybrid-trained RL agent
  -----------------------------------------------------------------------

The fixed-time baseline should use the same intersection, traffic demand
and simulation duration.

### 10.2 Test scenarios

Use held-out traffic scenarios where possible:

-   Low demand.
-   Medium demand.
-   High demand.
-   Uneven directional demand.
-   Changing or sudden demand.

The final test scenarios should not be used to tune hyperparameters or
select models. For SL, add a cross-distribution check: train on some
demand profiles and test on others (see §7.11). For RL, evaluate on
held-out scenarios and report recovery time after a demand change, not
only the average over the episode.

### 10.3 Metrics

  -----------------------------------------------------------------------
  Metric                  Desired direction       Notes
  ----------------------- ----------------------- -----------------------
  Average waiting time    Lower                   Define whether averaged
                                                  per vehicle or over
                                                  vehicles/time

  Average queue length    Lower                   Define
                                                  detector/approach and
                                                  sampling interval

  Average travel time     Lower                   Define completed-trip
                                                  handling

  Throughput              Higher                  Define vehicles that
                                                  pass or complete trips

  Cumulative reward       Higher                  Compare only under the
                                                  same reward definition
  -----------------------------------------------------------------------

For every controller, use consistent measurement definitions. Report
means, standard deviations and 95% bootstrap confidence intervals across
multiple runs. Use paired statistical tests for controller comparisons
where appropriate (see §10.6). Also report recovery time after a demand
change, defined as the time to return to a stated baseline queue or
waiting level.

### 10.4 Fairness and reproducibility

-   Use identical traffic demand files or matched demand seeds across
    controllers.
-   Use equal simulation durations and warm-up periods.
-   Use the same network and vehicle settings.
-   Run each controller across multiple seeds.
-   Do not tune on the final test set.
-   Record code version, model checkpoint, configuration and random
    seed.
-   Report failures and incomplete runs rather than silently dropping
    them.

### 10.5 Interpretation

Answer the research question using the results:

-   Does RL-only outperform fixed-time control?
-   Does hybrid SL + RL outperform RL-only?
-   Under which demand conditions does prediction help?
-   Does the hybrid system ever perform worse?
-   How do SL prediction errors affect signal decisions?
-   Which features most influence SL predictions and RL actions?
-   What types of prediction errors dominate, and under which demand
    conditions?
-   Does the controller treat minor approaches fairly, or starve them?
-   How much does SL performance degrade under demand-distribution
    shift?
-   What are the limitations of simulation-generated data?

Do not invent results. All tables and plots must be generated from
actual experiment outputs. Clearly label any illustrative values as
hypothetical.

### 10.6 Statistical reliability and comparisons

-   Report bootstrap 95% confidence intervals for SL metrics and for
    controller metrics.
-   Use paired tests across matched seeds/scenarios (for example,
    Wilcoxon signed-rank or a paired t-test when assumptions hold) and
    report effect sizes where useful.
-   Report the number of runs and the variability; do not claim a
    difference from a single run.
-   For RL, show learning curves and convergence/variance across seeds,
    not only final mean reward.

------------------------------------------------------------------------

## 11. Visualisations and Demonstration

Recommended visualisations:

### Data exploration

-   Vehicle count over time by direction.
-   Queue length and waiting time over time by direction.
-   Average speed by direction.
-   Correlation heatmap.
-   Traffic demand comparison across scenarios.
-   Missingness/zero-vehicle heatmap and timestamp-continuity check.
-   Fixed-time baseline characterisation: queue build-up/recovery and
    demand calibration.

### SL

-   Actual vs predicted vehicle counts.
-   Prediction residual plot, plus residuals by direction, demand level
    and time period.
-   MAE/RMSE comparison across models, with bootstrap confidence
    intervals.
-   Initial vs optimised model comparison.
-   Feature importance (impurity-based and permutation) and partial
    dependence/ICE plots.
-   Feature-set ablation comparison.
-   Cross-distribution performance matrix (train vs test demand
    profile).

### RL

-   Episode reward / return curves with variability across seeds.
-   Waiting time and queue length by episode.
-   Initial vs optimised agent comparison.
-   Action distribution conditioned on queue imbalance/phase.
-   Phase-switch frequency, wasted green and per-direction waiting
    times.
-   Recovery curve after a demand change.

### Final comparison

-   Waiting time by controller and scenario, with confidence intervals.
-   Queue length by controller and scenario.
-   Travel time by controller and scenario.
-   Throughput by controller and scenario.
-   Recovery time after a demand change.
-   SUMO visual demonstration of the intersection.

Optional: build a Streamlit dashboard showing current counts, SL
predictions, active signal phase, green duration and performance
comparisons. A dashboard is secondary; first ensure the simulation,
models and evaluation work correctly.

------------------------------------------------------------------------

## 12. Suggested Repository Structure

Adapt this structure to the actual project, but keep code modular and
avoid placing everything in one notebook.

``` text
traffic-signal-hybrid-ml/
│
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── simulation.yaml
│   ├── traffic_scenarios.yaml
│   ├── sl_config.yaml
│   ├── rl_config.yaml
│   └── hybrid_config.yaml
│
├── sumo/
│   ├── network/
│   │   ├── intersection.nod.xml
│   │   ├── intersection.edg.xml
│   │   ├── intersection.net.xml
│   │   └── intersection.sumocfg
│   ├── routes/
│   └── traffic_lights/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── scenarios/
│
├── src/
│   ├── simulation/
│   │   ├── run_simulation.py
│   │   ├── data_collector.py
│   │   └── scenario_generator.py
│   │
│   ├── preprocessing/
│   │   ├── build_sl_dataset.py
│   │   └── features.py
│   │
│   ├── supervised/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   └── predict.py
│   │
│   ├── reinforcement/
│   │   ├── environment.py
│   │   ├── controllers/
│   │   │   └── fixed_time.py
│   │   ├── q_learning/
│   │   ├── dqn/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tune.py
│   │
│   ├── hybrid/
│   │   ├── state_builder.py
│   │   ├── train_hybrid.py
│   │   └── evaluate_hybrid.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── run_comparison.py
│   │   ├── report_results.py
│   │   └── aggregate_results.py
│   │
│   └── utils/
│       ├── paths.py
│       ├── reproducibility.py
│       └── logging_utils.py
│
├── experiments/
│   └── run_all.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_sl_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
├── models/
│   ├── supervised/
│   ├── reinforcement/
│   └── hybrid/
│
├── results/
│   ├── sl/
│   ├── rl/
│   ├── hybrid/
│   ├── comparisons/
│   └── figures/
│
└── tests/
    ├── test_data_pipeline.py
    ├── test_signal_safety.py
    ├── test_environment.py
    └── test_prediction_alignment.py
```

This is a suggested structure, not a requirement to create every file
immediately. Create modules incrementally as they become necessary.

### 12.1 Experiment automation

Provide one config-driven entry point that can run the pipeline in
stages:

1.  Generate scenarios and raw observations.
2.  Build and audit the SL dataset.
3.  Train, tune and evaluate SL models.
4.  Train and evaluate RL agents.
5.  Build and train the hybrid agent.
6.  Run the final comparison and produce tables/figures.

`experiments/run_all.py` should support stage selection, skip completed
stages, record the configuration and seed, and fail loudly rather than
silently continuing. `src/evaluation/aggregate_results.py` should
collect metrics and regenerate the report tables from saved outputs.
Also provide a short documented command (or Makefile/PowerShell script)
for the full pipeline so results are reproducible.

------------------------------------------------------------------------

## 13. Recommended Implementation Milestones

### Milestone 1 --- Environment works

-   [ ] Confirm Python environment and dependencies.
-   [ ] Confirm SUMO and TraCI work from Python.
-   [ ] Build and open a simple four-way intersection.
-   [ ] Verify valid signal phases and safe transitions.
-   [ ] Run a small traffic simulation.

**Acceptance check:** A reproducible SUMO simulation runs successfully
and vehicles move through the intersection.

### Milestone 2 --- Data collection works

-   [ ] Generate several traffic scenarios.
-   [ ] Extract counts, speed, queue and waiting-time observations.
-   [ ] Save CSV files with timestamps, directions and scenario
    identifiers.
-   [ ] Verify data types, missing values and feature definitions.
-   [ ] Audit sentinel/undefined values and document the handling policy
    (§6.5).
-   [ ] Characterise the fixed-time baseline and calibrate demand
    (§6.6).

**Acceptance check:** A reproducible script creates a usable dataset
from SUMO runs.

### Milestone 3 --- SL pipeline works

-   [ ] Define the five-minute prediction target.
-   [ ] Create chronological train/validation/test splits.
-   [ ] Perform EDA.
-   [ ] Train candidate regression models.
-   [ ] Evaluate and select top candidates.
-   [ ] Tune selected candidates.
-   [ ] Save the best model and preprocessing pipeline.
-   [ ] Run the feature-set ablation (§7.9).
-   [ ] Report feature importance and prediction-error analysis
    (§7.7--§7.8).
-   [ ] Test cross-distribution robustness (§7.11).
-   [ ] Report bootstrap confidence intervals (§7.10).

**Acceptance check:** The best model can load and produce predictions
from a new observation window without data leakage.

### Milestone 4 --- RL-only pipeline works

-   [ ] Implement a fixed-time controller.
-   [ ] Implement a safe RL environment wrapper.
-   [ ] Implement Q-Learning.
-   [ ] Implement DQN if feasible.
-   [ ] Train and evaluate agents across multiple seeds.
-   [ ] Tune selected candidates.
-   [ ] Save the best RL checkpoint.
-   [ ] Run the state-representation ablation (§8.9).
-   [ ] Report RL diagnostics/inefficiencies and learning-curve stability
    (§8.8, §10.6).

**Acceptance check:** An RL agent can interact with SUMO, select valid
actions and produce logged evaluation metrics.

### Milestone 5 --- Hybrid works

-   [ ] Load the selected SL model.
-   [ ] Add predictions to the RL observation.
-   [ ] Check feature order, units and prediction horizon.
-   [ ] Train or fine-tune the hybrid RL agent.
-   [ ] Evaluate the hybrid agent on held-out scenarios.
-   [ ] Compare S1 and S2 state variants and report recovery time
    (§8.9, §10.2).

**Acceptance check:** The hybrid controller uses model-generated
predictions (not future ground truth) and controls the simulation
safely.

### Milestone 6 --- Final experiments and report

-   [ ] Run fixed-time, RL-only and hybrid comparisons.
-   [ ] Use matched scenarios and multiple seeds.
-   [ ] Calculate aggregate metrics and variability.
-   [ ] Run paired statistical tests and report confidence intervals
    (§10.6).
-   [ ] Regenerate all tables and figures with the one-command pipeline
    (§12.1).
-   [ ] Generate tables and plots from actual logs.
-   [ ] Discuss results, limitations and future work.

**Acceptance check:** All headline results can be reproduced from saved
configurations, checkpoints and scripts.

------------------------------------------------------------------------

## 14. Experiment Tracking Tables

### SL results

  ------------------------------------------------------------------------------------
  Model          Initial   Initial   Initial Tuned MAE     Tuned  Tuned R² Selected?
                     MAE      RMSE        R²                RMSE           
  ------------ --------- --------- --------- --------- --------- --------- -----------
  Linear                                                                   
  Regression                                                               

  Decision                                                                 
  Tree                                                                     

  Random                                                                   
  Forest                                                                   

  XGBoost                                                                  
  (optional)                                                               
  ------------------------------------------------------------------------------------

Only tune the selected top candidates. For models not tuned, mark the
tuned columns as N/A rather than implying they were optimised.

If the model-combination experiment is run (§7.2, §7.5), add an
`Ensemble / blend` row to the table. Evaluate it under the same
validation and untouched-test protocol as the individual models.

### RL results

  ------------------------------------------------------------------------------------------
  Algorithm      Initial   Initial      Initial     Tuned     Tuned        Tuned Selected?
                 waiting     queue   throughput   waiting     queue   throughput 
                    time                             time                        
  ------------ --------- --------- ------------ --------- --------- ------------ -----------
  Q-Learning                                                                     

  DQN                                                                            

  PPO                                                                            
  (optional)                                                                     
  ------------------------------------------------------------------------------------------

Include reward, travel time, seed count and variability in the detailed
experiment logs.

### Final system comparison

  ---------------------------------------------------------------------------
  Controller   Scenario   Mean waiting   Mean queue  Mean travel   Throughput
                                  time       length         time 
  ------------ ---------- ------------ ------------ ------------ ------------
  Fixed-time   Low                                               

  RL-only      Low                                               

  Hybrid SL +  Low                                               
  RL                                                             

  Fixed-time   Medium                                            

  RL-only      Medium                                            

  Hybrid SL +  Medium                                            
  RL                                                             

  ...          ...                                               
  ---------------------------------------------------------------------------

Populate these tables only with actual results.

------------------------------------------------------------------------

## 15. Coding Agent Instructions

You are assisting with a university Machine Learning assignment. Treat
this `PLAN.md` as the project specification and source of truth.

### General working rules

1.  **Inspect before changing.** First inspect the existing repository,
    files, environment and current implementation. Do not overwrite
    working code without understanding it.
2.  **Work incrementally.** Implement one milestone at a time. Do not
    generate the entire project in one untested pass.
3.  **Explain decisions.** Before substantial implementation, briefly
    state the proposed files, design and assumptions. Explain technical
    decisions in beginner-friendly language.
4.  **Prioritise a working baseline.** Get a simple SUMO intersection
    and data collection pipeline working before advanced ML.
5.  **Keep modules reusable.** Avoid duplicated simulation,
    preprocessing, metric and evaluation logic.
6.  **Use configuration files.** Keep paths, seeds, traffic rates,
    simulation duration, sampling interval, model parameters and reward
    weights configurable.
7.  **Reproducibility matters.** Set and record Python, NumPy,
    model-library and SUMO random seeds wherever possible.
8.  **Log experiments.** Save configuration, model name,
    hyperparameters, scenario, seed, metrics and timestamps for each
    run.
9.  **Never fabricate results.** Do not insert made-up performance
    numbers into result tables or claim that an experiment succeeded
    unless it was actually run.
10. **Prevent data leakage.** Use chronological splits for SL. Never
    provide future ground-truth traffic to the hybrid RL agent.
11. **Preserve traffic safety.** Enforce valid signal phases,
    yellow/clearance transitions and minimum/maximum green durations in
    the environment.
12. **Test each stage.** Add small tests and smoke tests before long
    training runs.
13. **Manage compute.** Start with short simulations and small training
    runs. Scale up only after verifying correctness.
14. **Document setup.** Keep README instructions accurate for installing
    dependencies, configuring SUMO and running each experiment.
15. **Do not silently change scope.** Ask before introducing multiple
    intersections, a new prediction target, a substantially different
    reward design or additional complex algorithms.
16. **Automate the pipeline.** Maintain a single config-driven entry
    point that can regenerate data, train/evaluate models and produce
    tables/figures from saved outputs (§12.1).

### Interaction and implementation sequence

At the beginning of work:

1.  Inspect the repository and identify what already exists.
2.  Check Python version, installed dependencies and SUMO/TraCI
    availability.
3.  Summarise the current state and identify the next incomplete
    milestone.
4.  Propose a small, testable next task.
5.  Implement it, run the relevant checks and report changed files and
    outcomes.

When a task is ambiguous, ask one focused clarification rather than
making a major architectural assumption.

### Required code quality

-   Use clear function and variable names.
-   Add type hints where useful.
-   Add docstrings to important modules and functions.
-   Comment on non-obvious logic and ML design choices, not every line.
-   Handle exceptions and close SUMO processes safely.
-   Use paths relative to the project root or a central path utility.
-   Keep raw data, processed data, models and results separate.
-   Ensure evaluation scripts can load saved models and configurations
    without retraining.

### Required experiment integrity

-   Model selection and tuning must use training/validation data, not
    final test results.
-   Use the same held-out test set for SL model comparison.
-   Use matched scenarios and multiple seeds for RL controller
    comparison.
-   Report mean and standard deviation where multiple runs are used.
-   Distinguish RL-only and hybrid checkpoints.
-   Distinguish the initial RL state from the expanded hybrid state.
-   Compare the hybrid controller against the same fixed-time and
    RL-only baselines.
-   Explain any failed runs, missing metrics or deviations from the
    planned experiment.

------------------------------------------------------------------------

## 16. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  SUMO setup takes too long           Start with the smallest possible
                                      network and test TraCI early

  SL dataset does not match           Generate intersection-specific data
  intersection task                   from SUMO

  SL target alignment is wrong        Unit-test that the target timestamp
                                      is exactly five minutes ahead

  Data leakage in forecasting         Use chronological splits and fit
                                      preprocessing only on training data

  RL training is unstable             Start with a small state/action
                                      space, reward logging and multiple
                                      seeds

  Unsafe signal actions               Restrict the action space and
                                      enforce phase-transition rules in
                                      the environment

  Hybrid state mismatch               Build and test a single shared
                                      state builder for training and
                                      evaluation

  Prediction errors harm control      Evaluate prediction quality and
                                      compare hybrid against RL-only

  Training is computationally         Use short smoke tests, then scale
  expensive                           episode count and scenario
                                      complexity

  Results are not reproducible        Save seeds, configs, checkpoints
                                      and run logs
  -----------------------------------------------------------------------

Additional risks introduced by the expanded experiment plan:

-   **Undefined or sentinel feature values corrupt the dataset.** Audit
    these values, convert only true sentinels to missing values, add
    explicit availability flags and document the policy (§6.5).
-   **A blend overfits the validation set.** Keep the blend simple,
    define it before final testing and evaluate it once on the untouched
    test set (§7.5).
-   **Statistical claims are overstated.** Report bootstrap confidence
    intervals, paired tests and effect sizes, and distinguish practical
    from statistical significance (§10.6).
-   **Demand shifts degrade SL predictions.** Report cross-distribution
    results and avoid presenting in-distribution accuracy as evidence of
    general robustness (§7.11).
-   **The pipeline drifts from the reported results.** Regenerate all
    tables and figures from saved outputs through one config-driven
    entry point (§12.1).

------------------------------------------------------------------------

## 17. Expected Final Deliverables

1.  SUMO four-way intersection network and configuration.
2.  Reproducible traffic scenario generation and data collection
    scripts.
3.  Raw and processed traffic datasets, or instructions to regenerate
    them.
4.  EDA notebook and visualisations.
5.  Trained SL candidate models and evaluation results.
6.  Optimised SL model and preprocessing pipeline.
7.  Fixed-time traffic controller.
8.  RL training environment and candidate agents.
9.  Optimised RL-only agent and evaluation results.
10. Hybrid SL + RL controller and trained hybrid agent.
11. Final comparative experiment outputs and visualisations.
12. README with setup and execution instructions.
13. Assignment report covering methodology, experiments, results,
    discussion and conclusion.

### 17.1 Report outline and submission checklist

Use the following structure for the final report. Adapt the depth of each
section to the final results, but keep the methodology and evidence
traceable to the code and saved outputs.

1.  **Title and overview.** Project title, team members, one-paragraph
    summary of the solution and headline findings.
2.  **Problem definition and research question.** The traffic-signal
    problem, why fixed-time control is limited and the precise hybrid
    SL + RL research question.
3.  **Critical evaluation of current technologies and related work.**
    Fixed-time, actuated and RL-based control; forecasting approaches;
    the rationale for the selected tools and algorithms (§4.1).
4.  **Proposed solution and system architecture.** SL, RL and hybrid
    components; how observations, predictions, actions and rewards flow
    through the system.
5.  **Simulation, scenarios and data generation.** Network, demand
    scenarios, observation collection, dataset construction, data-quality
    audit and missing-data handling (§6, especially §6.5--§6.6).
6.  **Supervised learning.** Features and target, leakage prevention,
    candidate models, validation and tuning, feature-set ablation,
    interpretability, prediction-error analysis and cross-distribution
    robustness (§7).
7.  **Reinforcement learning.** Environment, state, actions, reward,
    algorithms, training protocol, learning-curve stability, diagnostics
    and state-representation ablation (§8).
8.  **Hybrid integration.** Expanded state, training or fine-tuning,
    leakage prevention and a controlled comparison of the hybrid state
    variants (§9).
9.  **Evaluation design.** Controllers, test scenarios, metrics,
    statistical reliability, fairness and reproducibility (§10).
10. **Results.** SL, RL and hybrid results and the final controller
    comparison, generated from saved outputs rather than retyped values.
11. **Discussion.** Answer the research questions; examine errors,
    limitations, generalisation, adaptability, fairness and bias; discuss
    threats to validity.
12. **Conclusion and future work.**
13. **References.**
14. **Appendices.** Configurations, additional plots, implementation
    notes and reproducibility instructions.

**Submission checklist:**

-   [ ] Team has 3--4 members and a unique project title/solution.
-   [ ] Report includes a task-allocation and individual-contribution
    section. Add this only after the team agrees the final split; do not
    invent contributions.
-   [ ] Report is written in clear English and compiled as one group
    submission.
-   [ ] One copy of the assessment rubric is submitted separately.
-   [ ] The completed system is ready for the presentation/demo and Q&A
    session.

------------------------------------------------------------------------

## 18. Suggested Official Resources

-   SUMO: https://eclipse.dev/sumo/
-   SUMO documentation: https://eclipse.dev/sumo/docs/
-   TraCI documentation: https://eclipse.dev/sumo/docs/TraCI/
-   SUMO traffic-light tutorial:
    https://eclipse.dev/sumo/docs/Tutorials/TraCI4Traffic_Lights.html
-   Scikit-learn: https://scikit-learn.org/
-   PyTorch: https://pytorch.org/
-   Gymnasium: https://gymnasium.farama.org/

------------------------------------------------------------------------

## 19. Immediate Next Task

**Do not start by implementing all ML algorithms.**

First inspect the development environment and complete Milestone 1:

1.  Verify Python.
2.  Verify SUMO installation and executable access.
3.  Verify TraCI can be imported and communicate with SUMO.
4.  Build or run a minimal four-way intersection.
5.  Confirm that vehicles move and traffic lights switch through safe
    phases.
6.  Document the exact commands needed to reproduce the run.

Once this works, proceed to data collection. The project should advance
through the milestones in order, with a working and tested result at
each stage.
