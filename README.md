# Hybrid Supervised and Reinforcement Learning for Adaptive Traffic Signal Optimisation

A simulation-based machine learning project that investigates whether combining **Supervised Learning (SL)** and **Reinforcement Learning (RL)** can improve traffic signal control at a four-way intersection.

> **Project status:** Planning and development  
> **Primary environment:** Eclipse SUMO  
> **Main implementation language:** Python

---

## 1. Project Overview

Traffic congestion is a common problem at road intersections. Traditional traffic lights often use fixed signal timings, which do not respond well when traffic volume changes between directions.

This project proposes an adaptive traffic signal control system that combines two machine learning approaches:

- **Supervised Learning (SL)** predicts short-term future traffic volume using traffic observations.
- **Reinforcement Learning (RL)** learns how to control traffic lights by interacting with a simulated road environment and receiving feedback.
- **Hybrid SL + RL** provides the RL agent with both current traffic conditions and SL predictions, allowing it to consider expected traffic when choosing signal actions.

The project will use **Eclipse SUMO** to simulate a four-way intersection, generate traffic observations, and provide an interactive environment for training and testing traffic signal controllers.

## 2. Problem Statement

Fixed-time traffic signals operate according to predetermined schedules. When traffic demand changes, a fixed schedule may allocate too much green time to an approach with few vehicles while making busy approaches wait longer.

An RL controller can learn to adapt signal decisions based on current traffic conditions. However, a controller that only observes the current state may not anticipate traffic that is about to arrive.

This project investigates whether adding short-term traffic predictions from an SL model can help an RL controller make better traffic signal decisions.

## 3. Proposed Solution

The project will develop and compare three traffic signal control approaches:

| Approach | Description |
|---|---|
| Fixed-time control | Uses predetermined traffic signal timings as a baseline. |
| RL-only control | Uses current traffic observations to make adaptive signal decisions. |
| Hybrid SL + RL control | Uses current traffic observations and SL-predicted future traffic as input to an RL controller. |

The main research question is:

**Does incorporating supervised traffic-volume predictions into a reinforcement learning traffic signal controller improve traffic performance compared with RL-only and fixed-time control?**

The hybrid approach is a hypothesis to be tested. It is not assumed in advance that it will outperform the other approaches.

## 4. How the System Works

### 4.1 Overall workflow

```text
                  SUMO Traffic Simulation
                           |
                  Traffic Observations
                           |
             +-------------+-------------+
             |                           |
       Supervised Learning          RL Development
             |                           |
       Train and evaluate          Train and evaluate
       traffic predictors          RL controllers
             |                           |
       Select best SL              Select best RL
       model after tuning           algorithm after tuning
             |                           |
             +-------------+-------------+
                           |
                    Hybrid Integration
                           |
              Current traffic observations
                 + SL-predicted traffic
                           |
                Hybrid RL controller
                           |
                Traffic signal action
                           |
                  SUMO executes action
                           |
               Updated traffic conditions
                           |
                 Final system evaluation
```

### 4.2 Supervised Learning (SL)

The SL component learns to predict future traffic volume from traffic observations collected in SUMO.

Possible input features include:

- Current vehicle count for each approach.
- Recent vehicle counts.
- Average vehicle speed.
- Queue length.
- Waiting time.
- Traffic density, where available.
- Time-related features, where appropriate.

The initial prediction target is the **vehicle count approximately five minutes into the future**. The exact prediction horizon and sampling interval must be defined consistently during implementation.

Candidate regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor (optional, depending on time and resources)
5. Ensemble/blend of the top two tuned models (for example, averaging or stacking)

Models will be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The strongest candidates will be tuned, and the best optimised SL model will be selected for hybrid integration.

The ensemble/blend is a deliberate model-combination experiment: it combines the top two tuned models rather than adding another unrelated algorithm. It must be built without using final test data and evaluated on the same validation and untouched-test protocol as the individual models, so the report can show whether combination improves accuracy or reliability.

Beyond accuracy, the SL analysis reports feature importance and key drivers, prediction-error patterns, a feature-set ablation and cross-distribution robustness. The data-quality, sentinel-value and missing-data policy is specified in `PLAN.md` §6.5.

### 4.3 Reinforcement Learning (RL)

The RL component learns to control the traffic lights through repeated interaction with SUMO.

In this project:

- **Agent:** The traffic signal controller.
- **Environment:** The SUMO traffic simulation.
- **State:** Current traffic information, such as vehicle counts, queue lengths, speeds and current signal phase.
- **Action:** A valid traffic signal phase or permitted signal-control decision.
- **Reward:** Feedback designed to encourage lower congestion and waiting time.

An example reward formulation is:

```python
reward = -(alpha * total_queue + beta * total_waiting_time)
```

Here, `alpha` and `beta` are weights that determine the relative importance of queue length and waiting time. Their values must be selected and documented during experiments.

Candidate RL algorithms:

1. Q-Learning
2. Deep Q-Network (DQN)
3. Proximal Policy Optimisation (PPO) (optional)

RL-only models will first be trained and evaluated **without SL predictions**. This provides a fair standalone RL approach for comparison.

RL evaluation also includes learning-curve stability and variance across seeds, a state-representation ablation (base state versus phase-elapsed state versus SL-prediction state) and diagnostics such as phase-switch chatter, wasted green time and starvation of the minor approaches. See `PLAN.md` §8.8--§8.9 and §10.6.

### 4.4 Hybrid Supervised Learning + Reinforcement Learning

After independently developing and optimising the SL and RL components, the best optimised SL model will be integrated with the selected RL algorithm.

The hybrid workflow is:

1. SUMO provides current traffic observations.
2. The SL model predicts future traffic volume for the approaches.
3. The predicted values are added to the RL agent's state.
4. The hybrid RL agent uses current and predicted traffic information to select a signal action.
5. SUMO executes the action and returns updated traffic observations and feedback.

**Important:** The hybrid RL agent must be trained or fine-tuned using the expanded state that includes SL predictions. A previously trained RL-only agent cannot automatically make use of new prediction features without appropriate training.

The hybrid controller must use predictions produced by the SL model. It must not receive future ground-truth traffic values as input.

## 5. Role of SUMO

SUMO is not only used to generate the dataset. It has two major roles:

1. **Traffic data generation:** SUMO simulates traffic and provides observations used to build the SL dataset.
2. **RL environment:** SUMO provides the interactive environment in which RL agents learn and test traffic signal control.

SUMO is also used during hybrid integration and final evaluation.

Python communicates with SUMO through **TraCI (Traffic Control Interface)**. TraCI allows Python programs to retrieve traffic information and control traffic lights while the simulation is running.

The initial simulation scope is one four-way intersection with four incoming approaches:

- North
- South
- East
- West

## 6. Project Scope

### Included

- One simulated four-way intersection.
- Four incoming traffic approaches.
- SUMO-generated traffic observations.
- Short-term traffic-volume prediction.
- Multiple SL model experiments.
- Multiple RL algorithm experiments where feasible.
- Hyperparameter optimisation for selected models.
- Hybrid SL + RL integration.
- Comparison against fixed-time and RL-only control.
- Testing under different traffic demand conditions.

### Not included in the initial scope

To keep the project achievable, the initial implementation will not focus on:

- Multiple interconnected intersections.
- Multi-agent RL.
- Pedestrian signal control.
- Emergency vehicle priority.
- Real-world deployment on physical traffic lights.

These may be considered as future improvements if time and resources permit.

## 7. Traffic Scenarios and Evaluation

The system should be tested under several traffic conditions, such as:

- Low traffic demand.
- Medium traffic demand.
- High traffic demand.
- Uneven traffic demand between approaches.
- Changing or sudden traffic demand.

The final comparison will include:

| Controller | Purpose |
|---|---|
| Fixed-time | Traditional baseline. |
| Optimised RL-only | Measures the performance of RL using current traffic information. |
| Hybrid SL + RL | Measures whether adding predicted traffic information improves RL control. |

### Evaluation metrics

| Metric | Desired direction | What it measures |
|---|---|---|
| Average waiting time | Lower is better | Time vehicles spend waiting. |
| Average queue length | Lower is better | Number of vehicles waiting in queues. |
| Average travel time | Lower is better | Time taken by vehicles to complete their trips. |
| Throughput | Higher is better | Number of vehicles that successfully pass through the intersection. |
| Cumulative reward | Higher is better | RL feedback accumulated during an episode; compare only under the same reward definition. |

For a fair comparison, the controllers should be tested using the same simulation network, traffic demand scenarios, simulation durations and test seeds wherever possible. Use multiple runs and report averages and variability.

Report means, standard deviations and 95% bootstrap confidence intervals, and use paired statistical tests across matched seeds or scenarios where appropriate. Also report recovery time after a demand change to a stated baseline queue/waiting level, and evaluate the SL model on demand distributions that differ from its training distribution. See `PLAN.md` §7.11 and §10.6.

Keep final test scenarios separate from data and experiments used for model selection and tuning.

## 8. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language. |
| Eclipse SUMO | Traffic simulation environment. |
| TraCI | Python interface for interacting with SUMO. |
| Pandas | Dataset handling and preprocessing. |
| NumPy | Numerical operations. |
| Scikit-learn | SL models and evaluation. |
| PyTorch | Neural-network-based RL implementations such as DQN or PPO, if selected. |
| Matplotlib | Charts and result visualisation. |
| Git and GitHub | Version control and team collaboration. |

A critical comparison of fixed-time, actuated and RL-based control, together with the rationale for choosing SUMO, TraCI, scikit-learn and PyTorch, is given in `PLAN.md` §4.1. Summarise the key points in the report's related-work section rather than treating the technology table as the evaluation.

Some tools are optional and should only be added if required by the chosen implementation.

## 9. Suggested Repository Structure

```text
traffic-signal-hybrid-ml/
├── README.md
├── PLAN.md
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
│   ├── preprocessing/
│   ├── supervised/
│   ├── reinforcement/
│   ├── hybrid/
│   ├── evaluation/
│   └── utils/
│
├── experiments/
│   └── run_all.py
│
├── notebooks/
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
```

The structure is a suggested starting point. It can be adjusted as the implementation becomes clearer.

## 10. Getting Started

### Prerequisites

Install the following software:

1. Python (use the version agreed by the team).
2. Eclipse SUMO.
3. Git.
4. A Python IDE or editor, such as Visual Studio Code.

### Setup

1. Clone the project repository.
2. Create and activate a Python virtual environment.
3. Install the project's Python dependencies.
4. Confirm that SUMO is installed and accessible.
5. Confirm that Python can import and communicate with TraCI.
6. Run the basic four-way intersection simulation before implementing ML models.

Example commands (adjust the Python command for your operating system):

```bash
git clone <repository-url>
cd traffic-signal-hybrid-ml

python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies once `requirements.txt` has been prepared:

```bash
pip install -r requirements.txt
```

SUMO is a separate application; installing the Python packages does not install SUMO itself. Follow the official SUMO installation instructions and the project's setup notes.

## 11. Team Collaboration

All team members should use the same project repository and follow the agreed folder structure.

Recommended collaboration practices:

- Agree on compatible Python and SUMO versions.
- Share SUMO network, route and traffic-light configuration files.
- Keep code and configuration changes in GitHub.
- Document dataset columns, units, sampling intervals and prediction horizon.
- Record experiment settings, random seeds and results.
- Avoid committing large generated datasets or model files unless the team has agreed to store them in the repository.
- Coordinate changes to shared interfaces, such as traffic-state features, actions and reward definitions.

Not every member must generate the dataset personally. A designated member can generate and share the dataset for SL development. However, members responsible for RL training, hybrid integration or simulation testing need access to a working SUMO environment.

## 12. Development Roadmap

The project will be developed in the following stages:

1. **Environment setup:** Install and verify Python, SUMO and TraCI; create the four-way intersection.
2. **Traffic data generation:** Configure traffic scenarios, collect observations and prepare the dataset.
3. **SL development:** Train, evaluate and optimise traffic-volume prediction models.
4. **RL development:** Implement, train, evaluate and optimise RL controllers without SL predictions.
5. **Hybrid integration:** Add predictions from the selected SL model to the RL state, then train or fine-tune the hybrid agent.
6. **Final evaluation:** Compare fixed-time, RL-only and hybrid controllers under matched traffic scenarios.
7. **Reporting:** Analyse results, discuss limitations and prepare the final report.

For the detailed methodology, experiment plan, implementation guidance and milestones, refer to **[PLAN.md](PLAN.md)**.

## 13. Important Experimental Guidelines

- Use chronological data splits for traffic forecasting to avoid training on future observations.
- Do not use the final test set for model tuning or selection.
- Ensure the SL prediction target is correctly aligned with its input timestamp.
- Train and evaluate RL-only controllers without SL predictions.
- Train the hybrid RL agent with actual SL model predictions as part of its state.
- Apply safe traffic signal phase transitions, including appropriate yellow and clearance intervals.
- Use the same test scenarios and comparable simulation settings for final controller comparisons.
- Record experiment configurations and random seeds.
- Report measured results honestly. Do not present hypothetical or illustrative results as actual findings.
- Audit sentinel/undefined values and apply the missing-data policy in `PLAN.md` §6.5.
- Report bootstrap confidence intervals and paired statistical tests, not only point estimates (`PLAN.md` §10.6).
- Run the SL feature-set ablation and the RL state-representation ablation (`PLAN.md` §7.9 and §8.9).
- Test demand-distribution shift, adaptation and recovery time (`PLAN.md` §7.11).
- Regenerate final tables and figures through the one-command pipeline (`PLAN.md` §12.1).

## 14. Useful Resources

- [Eclipse SUMO](https://eclipse.dev/sumo/)
- [SUMO TraCI Documentation](https://eclipse.dev/sumo/docs/TraCI/)
- [SUMO Traffic Light Tutorial](https://eclipse.dev/sumo/docs/Tutorials/TraCI4Traffic_Lights.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)

## 15. Report Outline and Submission Checklist

The detailed report outline is in `PLAN.md` §17.1. The report should cover:

1. Title, team members and a brief overview.
2. Problem definition and research question.
3. Critical evaluation of current technologies and related work.
4. Proposed solution, architecture and how it works.
5. Simulation, traffic scenarios, data generation and data quality.
6. SL methodology, results, interpretability and error analysis.
7. RL methodology, results, training stability and diagnostics.
8. Hybrid integration, training and results.
9. Evaluation design, statistical comparisons and final results.
10. Discussion, limitations, fairness/bias and future work.
11. References and appendices.

**Submission checklist:**

- [ ] Team has 3–4 members and a unique title/solution with no duplication.
- [ ] Report includes task allocations and each member's individual contribution. Add this after the team agrees the final split; do not invent contributions.
- [ ] Report is written in clear English and compiled as one group submission.
- [ ] One copy of the assessment rubric is submitted separately from the report.
- [ ] The completed project is ready for the presentation/demo and Q&A in the practical class from week 7.

---

**Note:** This README introduces the project and helps team members understand its purpose and workflow. The official assignment brief and marking rubric remain the authority for submission requirements. Refer to `PLAN.md` for the detailed development plan.
