# Lane C notes — HistGradientBoosting & XGBoost

Branch `lane-c` · stage 2 (validation) · F4 feature set · test split **not** used.

## 1. Command

```powershell
python -m src.supervised.train --models hist_gradient_boosting xgboost persistence --feature-set F4 --seed 42
```

For all feature sets (section 5.3):

```powershell
foreach ($fs in 'F0','F1','F2','F3','F4') {
    python -m src.supervised.train --models hist_gradient_boosting xgboost persistence --feature-set $fs --seed 42
}
```

This command reproduces the optimized results exactly. The original results
come from the same command with the lines marked `# Lane C validation tuning`
removed from `configs/sl.yaml`.

## 2. Parameter search (validation, F4)

**How the best value was chosen.**

- Each table changes **one** parameter. The other parameters stay at their
  selected values.
- The value with the lowest validation MAE is chosen.
- If another value is within 0.01 MAE, the two count as equal, and the
  simpler or faster one is chosen.
- A value is rejected if it memorises the training data or costs much more
  training time for no gain.

### HGB — `max_leaf_nodes` (other parameters: `max_iter=300`, `learning_rate=0.08`)

| `max_leaf_nodes` | Train MAE | Val MAE | Val RMSE | Fit time | |
|---:|---:|---:|---:|---:|---|
| 5 | 1.123 | 2.710 | 6.035 | 0.4 s | |
| 7 | 1.060 | 2.661 | 5.964 | 0.4 s | |
| **10** | 0.986 | **2.626** | **5.895** | 0.5 s | best |
| 12 | 0.945 | 2.630 | 5.949 | 0.5 s | equal to 10, but more leaves |
| 15 | 0.890 | 2.639 | 5.999 | 0.6 s | |
| 20 | 0.836 | 2.703 | 6.206 | 0.7 s | |
| 31 (default) | 0.742 | 2.705 | 6.260 | 0.9 s | original |
| 63 | 0.619 | 2.787 | 6.428 | 1.3 s | |

**Reason:** `max_leaf_nodes` sets how many rules one tree can learn.

- **5 leaves — underfitting.** Train MAE (1.123) and val MAE (2.710) are both
  higher than at 10. The trees are too small to learn the traffic pattern.
- **63 leaves — overfitting.** Train MAE is the lowest (0.619), but val MAE is
  the worst (2.787). The train–val gap grows from 1.64 (10 leaves) to 2.17
  (63 leaves). The extra leaves memorise the training runs.
- **10 vs 12.** Val MAE differs by only 0.004, which is under the 0.01 rule.
  10 also has the lower RMSE (5.895 vs 5.949), so we chose 10 because it is
  simpler and faster.

Other HGB parameters were also tried one at a time from the default:

- `max_iter`
- `learning_rate`
- `max_depth`
- `min_samples_leaf`
- `l2_regularization`

None did better than `max_leaf_nodes`. HGB `max_depth=24` gave exactly the
default result, because the default 31-leaf limit already stops the tree
before it reaches depth 24.

### XGB — `min_child_weight` (other parameters: `subsample=0.6`, `max_depth=6`)

| `min_child_weight` | Train MAE | Val MAE | Val RMSE | |
|---:|---:|---:|---:|---|
| 1 (default) | 0.773 | 2.779 | 6.353 | original |
| 10 | 0.807 | 2.674 | 6.128 | |
| 20 | 0.838 | 2.718 | 6.247 | |
| 35 | 0.873 | 2.652 | 6.081 | |
| **50** | 0.896 | **2.633** | **5.977** | best |
| 75 | 0.927 | 2.746 | 6.237 | |
| 100 | 0.954 | 2.823 | 6.409 | |

**Reason:** `min_child_weight` is the smallest amount of data a leaf may hold.
With the squared-error loss used here, this is the number of training rows in
the leaf.

- **1 (default) — too small.** A leaf can be built from very few rows, so the
  tree learns noise from single runs. Train MAE is the lowest (0.773), but val
  MAE is high (2.779).
- **100 — too large.** The tree cannot make small leaves, so it misses patterns
  that only a small group of rows shows. Train MAE rises to 0.954, and val MAE
  (2.823) is even worse than the default.
- **50 — the balance.** It has the lowest val MAE and RMSE.
- Train MAE rises steadily as the value grows, but val MAE does not move
  smoothly (10 → 20 → 35 gives 2.674, 2.718, 2.652). Part of the difference
  between neighbouring values is noise (see section 7).

### XGB — `subsample` (other parameters: `min_child_weight=50`, `max_depth=6`)

| `subsample` | Train MAE | Val MAE | Val RMSE | |
|---:|---:|---:|---:|---|
| 0.4 | 0.928 | 2.772 | 6.307 | |
| 0.5 | 0.914 | 2.678 | 6.102 | |
| **0.6** | 0.896 | **2.633** | **5.977** | best |
| 0.7 | 0.885 | 2.641 | 6.028 | equal to 0.6 |
| 0.8 | 0.879 | 2.649 | 6.073 | |
| 0.9 | 0.882 | 2.677 | 6.164 | |
| 1.0 (default) | 0.900 | 2.673 | 6.132 | original |

**Reason:** `subsample` is the share of training rows that each tree sees. Each
tree gets a different random sample.

- **Why less than 1.0 helps.** Because each tree sees different rows, the trees
  make different errors. When the trees are added together, many of these
  errors cancel out, which reduces variance. This is the same idea as bagging
  in a random forest. Val MAE improves from 2.673 (1.0, all rows) to 2.633
  (0.6).
- **0.4 — too little data.** Each tree sees too few rows to learn the pattern
  well, so val MAE rises to 2.772, worse than using all rows.
- **0.6 vs 0.7.** Val MAE differs by only 0.007, which is under the 0.01 rule.
  0.6 has the lower MAE and RMSE (5.977 vs 6.028), and each tree trains on
  fewer rows, so we chose 0.6.
- Unlike the other parameters, train MAE barely changes here (0.879–0.928).
  The gain does not come from making each tree simpler. It comes from making
  the trees different from each other.

### XGB — `max_depth` (other parameters: `min_child_weight=50`, `subsample=0.6`)

| `max_depth` | Train MAE | Val MAE | Val RMSE | Fit time | |
|---:|---:|---:|---:|---:|---|
| 4 | 1.046 | 2.681 | 6.013 | 0.4 s | |
| **6 (default)** | 0.896 | **2.633** | **5.977** | 0.5 s | best — default kept |
| 8 | 0.795 | 2.648 | 6.092 | 0.6 s | |
| 24 | 0.586 | 2.645 | 6.095 | 1.8 s | rejected: no gain, 3.4× slower |

**Reason:** `max_depth` is how many levels of splits (questions) one tree can
ask.

- **4 — too shallow (underfitting).** Train MAE (1.046) and val MAE (2.681) are
  both higher than at 6. The trees cannot combine enough conditions, for
  example a lag, the phase and the queue together.
- **8 and 24 — deeper, but no better.** Train MAE keeps falling (0.795, then
  0.586), but val MAE does not improve (2.648, 2.645). The extra depth only
  fits the training data better. Depth 24 also takes 3.4× longer to train
  (1.8 s vs 0.5 s), so it is rejected.
- **6 (default) — the best.** It has the lowest val MAE and RMSE, so the
  default was kept. Keeping the default is also a valid tuning result.
- **The parameters interact.** When `max_depth=24` was tried alone (with
  `min_child_weight=1`), train MAE was 0.0005, which means the model memorised
  the training data. With `min_child_weight=50`, train MAE at depth 24 is
  0.586. The leaf-size limit stops the deep tree from growing tiny leaves, so
  the depth limit matters much less.

Other XGB parameters were also tried one at a time from the default:

- `n_estimators`
- `learning_rate`
- `colsample_bytree`
- `reg_lambda`

None did better than the three above.

**Final selected parameters (now in `configs/sl.yaml`):**

| Model | Parameter | Original | Optimized |
|---|---|---:|---:|
| HGB | `max_leaf_nodes` | 31 | 10 |
| XGB | `min_child_weight` | 1 | 50 |
| XGB | `subsample` | 1.0 | 0.6 |
| XGB | `max_depth` | 6 | 6 (unchanged) |

## 3. Original vs Optimized (validation, 9,600 rows)

| Metric | HGB original |  HGB optimized | Change | XGB original | XGB optimized | Change |
|---|---:|---:|---:|---:|---:|---:|
| MAE ↓ | 2.705 | **2.626** | −0.079 (−2.9%) | 2.788 | **2.633** | −0.155 (−5.6%) |
| RMSE ↓ | 6.260 | **5.895** | −0.365 (−5.8%) | 6.363 | **5.977** | −0.386 (−6.1%) |
| R² ↑ | 0.790 | **0.813** | +0.024 | 0.782 | **0.808** | +0.026 |
| MAPE ↓ | 25.3% | **25.2%** | −0.1 pt | 25.3% | **24.5%** | −0.8 pt |
| Train MAE | 0.742 | 0.986 | +0.244 | 0.807 | 0.896 | +0.089 |

**Result:** both optimized models are better than the original on MAE, RMSE,
R² and MAPE.

## 4. Comparison

### 4.1 Optimized models vs persistence baseline

| Metric | Persistence | HGB optimized | vs persistence | XGB optimized | vs persistence |
|---|---:|---:|---:|---:|---:|
| MAE ↓ | 4.983 | **2.626** | −47.3% | **2.633** | −47.2% |
| RMSE ↓ | 6.702 | **5.895** | −12.0% | **5.977** | −10.8% |
| R² ↑ | 0.759 | **0.813** | +0.055 | **0.808** | +0.049 |
| MAPE ↓ | 58.2% | **25.2%** | −33.0 pt | **24.5%** | −33.7 pt |

**Result: both models clearly beat persistence** on all four metrics.

### 4.2 HGB vs XGB (same algorithm family: gradient-boosted trees)

| Metric | HGB optimized | XGB optimized | Difference |
|---|---:|---:|---|
| MAE ↓ | 2.626 | 2.633 | 0.007 (0.3%) |
| RMSE ↓ | 5.895 | 5.977 | 0.082 (1.4%), HGB lower |
| R² ↑ | 0.813 | 0.808 | 0.005, HGB higher |
| MAPE ↓ | 25.2% | 24.5% | 0.7 pt, XGB lower |

**Result: tie.**

- The MAE difference is only 0.007 (0.3%). This is much smaller than the
  0.05–0.1 change between neighbouring parameter values in section 2.
- No model wins on every metric.
- Before optimization, HGB led by 0.083 MAE. After both were optimized, the
  gap disappeared.

## 5. Breakdown (validation)

### 5.1 MAE by scenario

| Scenario | HGB original | HGB optimized | XGB original | XGB optimized | Persistence |
|---|---:|---:|---:|---:|---:|
| low | 0.528 | 0.613 | 0.527 | 0.536 | 2.512 |
| medium | 0.581 | 0.595 | 0.580 | 0.581 | 4.472 |
| high | 0.910 | 1.028 | 0.882 | 0.966 | 4.837 |
| uneven | 2.770 | 2.800 | 3.224 | 3.031 | 3.857 |
| changing | 8.737 | **8.096** | 8.730 | **8.054** | 9.238 |

### 5.2 MAE by direction

| Direction | HGB original | HGB optimized | XGB original | XGB optimized | Persistence |
|---|---:|---:|---:|---:|---:|
| north | 4.105 | **3.978** | 4.408 | **4.157** | 5.772 |
| south | 2.147 | **2.075** | 2.145 | **2.028** | 4.626 |
| east | 2.279 | **2.197** | 2.296 | **2.138** | 4.522 |
| west | 2.290 | **2.255** | 2.304 | **2.211** | 5.012 |

After optimization, all four directions got better for both models.

**Why north is the worst.** The extra error of north comes from `uneven`. In
this scenario north has the highest demand (800 veh/h; west 600, south and
east 250).

| Optimized HGB / XGB MAE | north | west | south | east |
|---|---:|---:|---:|---:|
| `uneven` | 9.19 / 10.34 | 0.68 / 0.63 | 0.65 / 0.57 | 0.68 / 0.58 |
| mean vehicle count in `uneven` | 64.0 | 10.2 | 3.8 | 4.2 |

North in `uneven` has 64 cars on average, about 6 times the other directions.
The fixed-time signal cannot clear this many cars, so the queue keeps getting
longer during the run:

- In training, the count goes from about 20 to 43 (max 53).
- In validation, the count goes from 54 to 64 (max 71).

So most of the validation counts are higher than anything the model saw in
training. A tree cannot predict a value higher than what it saw in training,
so the error becomes large. In the other scenarios, north is about the same as
the other directions.

### 5.3 MAE by feature set

The parameters were selected on F4 and then used unchanged on F0–F3.

| Feature set | Columns | HGB original | HGB optimized | XGB original | XGB optimized |
|---|---:|---:|---:|---:|---:|
| F0 | 7 | 5.377 | 5.377 | 5.700 | 5.388 |
| F1 | 19 | 2.693 | 2.728 | 2.780 | 2.814 |
| F2 | 26 | 2.726 | 2.714 | 2.797 | 2.830 |
| F3 | 36 | 2.680 | 2.758 | 2.828 | 2.803 |
| F4 | 41 | 2.705 | **2.626** | 2.788 | **2.633** |

Persistence MAE is 4.983 for every set, because it uses no features.

- **F0: both models are worse than persistence.** F0 only has the current
  count, with no history and no signal phase.
  - Persistence just copies the current count, so it can follow any level,
    even one never seen in training (north in `uneven`, the peak in
    `changing`).
  - The tree can only give values it saw in training, so on these rows it
    loses to persistence.
- **F1: big improvement** (about 5.4 → 2.7). F1 adds the count history (lags
  and rolling means), and this is the most useful information.
- **F1 to F4: almost the same.** The difference is within about 0.1, so the
  extra features do not help much.
- **The optimized parameters only help on F4.** On F1–F3 some results are
  better and some are worse.
  - Each feature set needs its own best parameters, so F1–F3 would need to be
    tuned again.
  - Part of the F4 improvement may also come from choosing the parameters on
    the same validation data (see section 7).

## 6. Overall reasons

Some background first:

- **Each scenario has the same weight.** Validation has 1,920 rows for each
  scenario (4 seeds × 4 directions × 120 samples). So the overall MAE is just
  the average of the 5 scenario MAEs. For example, HGB optimized:
  (0.613 + 0.595 + 1.028 + 2.800 + 8.096) / 5 = 2.626.
- **Validation and training are in different time periods.** Training uses
  305–1,800 s and validation uses 2,105–2,700 s
  (`docs/supervised_learning_dataset.md`).
- **In `changing`, the demand goes 400 → 800 → 400 veh/h.** It changes at
  1,800 s and at 2,700 s.

1. **Why did optimization make both models better?**
   - The default models were overfitting. For example, the default HGB has
     train MAE 0.742 but val MAE 2.705, so it is much better on the training
     data than on new data.
   - All the parameters we chose make the model simpler or add randomness:
     - HGB: fewer leaves.
     - XGB: bigger minimum leaf size, and each tree uses only part of the rows.
   - In section 3, train MAE went **up** for both models (HGB +0.244, XGB
     +0.089), but val MAE went **down**. This means the models memorise less of
     the training data and work better on new data.

2. **Why did most of the improvement come from `changing`, while `low`/`medium`/`high` got a little worse?**
   - **`changing` has the biggest error.** Its MAE is about 8.7, while the
     other scenarios are below 3. Before tuning it was about 63–65% of the
     total error.
     - For HGB, `changing` improved by 0.641, so the overall average goes down
       by 0.128 (0.641 / 5).
     - The other 4 scenarios got worse by 0.247 in total, so the average goes
       up by only 0.049.
     - Overall it is −0.079.
   - **In `changing`, validation is different from training.** Training only
     has the 400 veh/h period, but validation is in the 800 veh/h peak.
     - The complex model learned very detailed rules from the training data.
       On new data that it never saw, these rules give extreme predictions.
     - The simpler model takes the average of more rows, so its predictions
       are less extreme and the error is smaller.
   - **`low`/`medium`/`high` are the same in training and validation**
     (the demand does not change). Here the detailed rules of the default
     model were actually correct. When we make the model simpler, we lose a
     bit of this detail, so the error goes up a little.

3. **Why is `changing` still the worst after tuning?**
   - **The model never saw a drop in training.**
     - The target is the count 300 s later. For validation inputs from 2,400
       to 2,700 s, the target time is after the drop from 800 to 400 veh/h at
       2,700 s. So about half of the validation rows ask the model to predict
       a drop.
     - But no training data has a drop in demand. Training ends at 1,800 s,
       and the other scenarios have constant demand.
   - **The features cannot tell the model that a drop is coming.** All
     features only use the past (lags and rolling means of up to 300 s).
     Nothing at time *t* says the demand will drop at 2,700 s, so the model
     keeps predicting peak-level counts. Tuning makes it better (8.7 → 8.1),
     but it cannot fix this missing information.
   - **So this is a data and feature problem, not a tuning problem.** To fix
     it, the training data needs to include demand changes. For example, we
     could train on `changing` runs that cover the whole hour, or add more
     scenarios with different demand.

4. **Why do HGB and XGB tie?**
   - **Same method, same data.** Both are gradient-boosted trees, both use the
     same F4 features, and after tuning both models have a similar size.
   - **The difference is too small to matter.**
     - The MAE difference is only 0.007. Just changing one parameter to the
       next value in section 2 moves MAE by 0.05–0.1, which is 7 to 14 times
       bigger.
     - No model wins on every metric. HGB has lower RMSE and higher R², but
       XGB has lower MAPE.
   - **Why HGB was better before tuning.**
     - Most of the gap came from `uneven`, where the default XGB was 0.454
       worse (3.224 vs 2.770). This adds about 0.09 to the overall average,
       which is close to the 0.083 gap.
     - After tuning, the XGB error in `uneven` went down (3.224 → 3.031), and
       both models improved by about the same amount in `changing`. So the
       gap closed.

5. **Why do both models beat persistence by so much?**
   - **Why persistence is bad.**
     - Persistence just copies the current count. But the signal is
       fixed-time with a 90 s cycle, so the queue goes up during red and down
       during green.
     - 300 s later is 3 cycles + 30 s, which is a different point in the
       cycle. So the current count is often a bad guess, even when the demand
       does not change (persistence MAE 2.5–4.8 in `low`/`medium`/`high`).
   - **Why the models are good.** They use rolling means (the average level)
     and the signal-phase features (`time_in_phase`, phase), so they can learn
     this repeating cycle. Their MAE in these scenarios is about 1 or lower.
   - **Why RMSE improves much less than MAE** (about −11 to −12% vs −47%).
     - RMSE squares the errors, so a few very large errors control it.
     - These large errors mostly come from `changing`. There, both the models
       (about 8.1) and persistence (9.2) fail at the demand drop.
     - The models remove many small errors, but not the large ones.

## 7. Notes for the owner

- The parameter changes follow the owner's approval on 2026-11-17. Contract
  §5 still says "no tuning".
- Parameters were selected on the validation split, so the optimized scores are
  slightly optimistic. Neighbouring values can move MAE by about 0.05–0.1 (see
  `min_child_weight`).
- The optimized parameters do not consistently help F1–F3 (section 5.3). The
  test split will give the unbiased number.
- Every result uses one seed (42). HGB has no randomness in this setup, and
  XGB `subsample` uses `random_state=42`, so reruns give identical numbers.
  Seed-to-seed variation was not measured.
- Lane C is now tuned. If other lanes keep their defaults, the stage-3 ranking
  will favour Lane C.
- Full trial results are in `results/sl/tuning/` (gitignored).
- **Run-level bootstrap CIs (contract §2.5) are not computed yet.** They are
  left for the final evaluation stage. The comparisons in sections 3–4 use the
  point values only.
- **Metrics JSON:** `results/` is gitignored, so the JSON files are not in this
  tag. The command in section 1 regenerates them exactly
  (`results/sl/metrics/<F>/{hist_gradient_boosting,xgboost,persistence}.json`).
