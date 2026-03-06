# AgentTwin — Reproducible Code Package 

This package reproduces the AgentTwin experiments on the **Tennessee Eastman Process (TEP)** using:
- A **real** TEP simulator vendored under `third_party/tep/` (source and license preserved).
- A hierarchical MARL architecture:
  - **Continuous control**: residual RL on top of the standard TE decentralized PI controller.
  - **Scheduling**: discrete operating-mode selection on a slower time scale.
  - **Safety shield**: QP-based residual filter (OSQP) with a heuristic fallback.

The training and evaluation scripts generate:
- Raw per-episode metrics (`results/raw/`)
- CSV + LaTeX tables (`results/tables/`)
- B/W figures with a single accent color (`results/plots/`) saved as both PNG and PDF

## 1) Installation

Recommended: Python **3.10+** (3.11 works well).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Verify the simulator

```bash
make verify
```

To additionally confirm that **scenarios are distinct** (different demand schedules,
disturbances, biases, etc.) and that disturbances become active as expected:

```bash
make verify_scenarios
```

## 3) Quick end-to-end run (smoke test)

Uses `configs/quick.yaml` (short horizons, few timesteps):

```bash
make reproduce_quick
```

## 4) Paper-style run (hours+)

Uses `configs/paper.yaml` (multi-seed, longer horizons):

```bash
make reproduce
```

Optional stronger configuration:

```bash
make reproduce_strong
```

Adjust seeds, timesteps, and horizons directly in `configs/paper.yaml`.

## 5) Outputs

After evaluation:

- `results/raw/all_episode_metrics.csv`: one row per episode (seed × scenario × method)

Tables (CSV + LaTeX):
- `results/tables/table1_performance_S2.csv` and `.tex`
- `results/tables/table2_safety_S2.csv` and `.tex`
- `results/tables/table3_scenarios.csv` and `.tex`
- `results/tables/table4_fairness_S2.csv` and `.tex`
- `results/tables/table5_scheduling_S2.csv` and `.tex`  (demand adherence + switching)


Plots (PNG + PDF):
- `results/plots/figure1_reward_S2.(png|pdf)`
- `results/plots/figure2_violations_S2.(png|pdf)`
- `results/plots/figure3_scenario_robustness.(png|pdf)`
- `results/plots/figure4_fairness_S2.(png|pdf)`

An executive summary is written to:
- `results/executive_summary.md`

## 6) Scenario set (S1–S5)

Scenarios are defined in `envs/scenarios.py`:
- **S1** nominal grade changes
- **S2** dynamic demand
- **S3** fault/disturbance injection (TE IDV)
- **S4** sensor bias (observation corruption)
- **S5** model/actuator mismatch (MV gain/bias)

You can switch between the minimal scenario set and the full set by editing:
- `experiment.scenario_set` in the YAML config.

## 6.1) Scheduling reward terms

The scheduling environment includes optional penalties that are important for
integrated scheduling-control evaluation:

- `reward.demand_mismatch`: penalty when the selected operating mode differs from
  the (exogenous) demand mode at that time.
- `reward.mode_switch`: penalty for switching operating modes (discourages
  unnecessary chattering).

These are configured in the YAML files under `env.reward`.

## FAQ: Why do I see `DummyVecEnv`?

Stable-Baselines3 wraps any Gymnasium environment in a *vectorized* wrapper (often `DummyVecEnv`) so that algorithms have a consistent interface.
This does **not** mean the plant is a dummy simulation — it is just a batching wrapper.

## License and attribution

The Tennessee Eastman simulator code under `third_party/tep/` is vendored from an open-source implementation.
See `third_party/tep/LICENSE` for license terms and attribution.

## 7) Linear data-driven MPC baseline (LinMPC)

To address reviewer requests for a *data-driven MPC* baseline, this package includes a **mode-dependent linear MPC** controller that operates in the same residual-control structure as the RL controllers:
- The decentralized PI controller remains in the loop.
- LinMPC computes a **continuous residual correction** (12-D) every control interval.
- The residual is solved via a condensed QP with **OSQP**.

### Step-by-step

```bash
# 1) Collect identification data (one dataset per operating mode)
python -m baselines.linmpc.collect_id_data --config configs/paper.yaml --out_dir artifacts/linmpc --seed 0

# 2) Fit a mode-dependent linear model
python -m baselines.linmpc.fit_model --data_dir artifacts/linmpc/data --out_path artifacts/linmpc/model_linmpc.npz

# 3) Evaluate linMPC and merge into the existing results (tables/figures regenerated)
python -m baselines.linmpc.eval_linmpc --config configs/paper.yaml --model artifacts/linmpc/model_linmpc.npz --results_dir results
```

### One-shot target

```bash
make linmpc_all
```

Notes:
- LinMPC does **not** require any logs from the RL training run. It generates its own identification dataset using the real simulator.
- If you already have high-frequency trajectories saved (e.g., full (state, action) traces), you can reuse them for identification, but the default pipeline is self-contained.
