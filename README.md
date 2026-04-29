# AgentTwin for TEP
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JPC](https://img.shields.io/badge/JPC-2026.xxxxxx-b31b1b.svg)](https://www.sciencedirect.com/journal/journal-of-process-control)

**AgentTwin}: a multi-agent digital-twin testbed for supervisory operating-mode scheduling and residual regulatory control on the Tennessee Eastman Process with a solver-backed safety shield**
![agenttwin_graphical_abstract_generated (1)](https://github.com/user-attachments/assets/80283cd6-640a-4732-86e5-4457f6092cbf)

This repository provides a **fully reproducible** experimental pipeline for the *AgentTwin* architecture described in the accompanying Journal of Process Control manuscript. The codebase is designed for **fair, auditable comparisons** of learning-based and classical controllers on a simulation-based digital twin of the **Tennessee Eastman Process (TEP)**.

AgentTwin integrates three components:

- **Slow-time scheduling (discrete):** a PPO policy selects the operating mode (m_k \in {1,...,6}) every (T_s = 300) s.
- **Fast-time regulatory control (continuous):** residual controllers (centralized SAC or **multi-agent SAC**) adjust manipulated variables every \(T_c = 6\) s **on top of** the benchmark decentralized PI controller shipped with the TEP simulator.
- **Solver-backed safety layer:** a **QP projection shield** (OSQP) filters residual actions and falls back to a conservative safe action if the projection is infeasible or fails.

The repository includes:
- A **real** TEP simulator vendored under `third_party/tep/` (original source and license preserved).
- One-command reproduction of training + evaluation + publication figures/tables.
- A linear **data-driven MPC baseline (LinMPC)** requested by reviewers, implemented in the same residual-control structure.

---

## Table of contents
- [Quick start](#quick-start)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Scenario suite](#scenario-suite)
- [Baselines](#baselines)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Linear data-driven MPC baseline (LinMPC)](#linear-data-driven-mpc-baseline-linmpc)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [License and attribution](#license-and-attribution)
- [Citation](#citation)

---

## Quick start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Sanity-check the TEP simulator and the environment wrappers
make verify
make verify_scenarios

# 4) Run a short smoke test (minutes)
make reproduce_quick
```

---

## Installation

**Recommended:** Python **3.10+** (Python 3.11 is supported).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- The default configuration uses the **pure-Python** TEP backend (`env.tep_backend: python`).  
- If you maintain a compiled backend (e.g., Fortran), you may extend `third_party/tep/` and set `env.tep_backend` accordingly.

---

## Repository layout

```
.
├── configs/                 # YAML experiment configs (quick / paper / paper_strong)
├── control/                 # benchmark PI controllers and residual interfacing
├── envs/                    # TEP gymnasium envs, scenarios, QP safety shield
├── train/                   # training pipeline (SAC controller(s), PPO scheduler)
├── eval/                    # evaluation + table/figure generation
├── baselines/linmpc/         # linear data-driven MPC baseline (ID, fit, eval)
├── third_party/tep/          # vendored TEP simulator (license preserved)
├── verify_tep.py             # simulator sanity checks
├── verify_scenarios.py       # scenario fingerprints + short rollouts
└── Makefile                  # convenience targets for full reproduction
```

---

## Reproducing the paper experiments

### 1) Verify simulator and scenarios

```bash
make verify
make verify_scenarios
```

`verify_scenarios` prints a **scenario fingerprint** and runs a short rollout to confirm that each scenario actually triggers the intended demand schedule / disturbance / mismatch.

### 2) Smoke test (fast)

```bash
make reproduce_quick
```

Uses `configs/quick.yaml` (small training budgets) to validate the full end-to-end pipeline.

### 3) Paper configuration

```bash
make reproduce
```

Uses `configs/paper.yaml` and performs:
1. Training of controller baselines (centralized SAC and multi-agent SAC).
2. Training of the PPO scheduler on the closed loop with the fixed controller layer.
3. Evaluation of all methods on scenarios **S1–S5** across seeds and episodes.
4. Generation of all plots and LaTeX-ready tables.

### 4) Stronger configuration (recommended when resources allow)

```bash
make reproduce_strong
```

Uses `configs/paper_strong.yaml` with more seeds / episodes and larger training budgets.

> Runtime depends on CPU/GPU, BLAS libraries, and the chosen budgets (`sac_steps`, `ppo_steps`, number of seeds). Use `configs/quick.yaml` to validate correctness before launching a multi-seed run.

---

## Scenario suite

Scenarios are defined in `envs/scenarios.py`.

- **S1 (nominal):** scripted mode sequence, nominal plant.
- **S2 (dynamic demand):** stochastic demand/mode changes.
- **S3 (fault/disturbance):** standard TEP disturbance injection (e.g., an IDV step).
- **S4 (sensor bias):** measurement bias injected into selected observations.
- **S5 (actuator / model mismatch):** per-MV gain/bias perturbations (digital-twin drift proxy).

Select the scenario set via:
- `experiment.scenario_set: minimal` (quick checks), or
- `experiment.scenario_set: paper` (full S1–S5 suite).

---

## Baselines

The evaluation suite reports:

- **PI (benchmark):** decentralized PI controller provided with the simulator.
- **Centralized SAC (C-SAC):** one SAC policy outputs a 12-D residual action.
- **Multi-agent SAC (MA-SAC):** three SAC policies output residuals for disjoint MV groups.
- **AgentTwin:** PPO scheduler + MA-SAC controller (two-time-scale integration).
- **Ablation (no shield):** AgentTwin without the QP projection (residual clipping only).
- **LinMPC (data-driven):** linear model identified from safe excitation data; online QP residual MPC.

All learning-based methods share:
- the same observation normalizations,
- the same reward definition and constraints,
- the same scenario suite and evaluation protocol,
so differences are attributable to decomposition and the scheduling layer (plus the safety shield ablation).

---

## Outputs

After any `make reproduce*` target, outputs are written under `results/`:

### Raw metrics
- `results/raw/all_episode_metrics.csv`  
  One row per episode (seed × scenario × method) with reward, safety, fairness, and scheduling metrics.

### Tables (CSV + LaTeX)
- `results/tables/table1_performance_S2.(csv|tex)`
- `results/tables/table2_safety_S2.(csv|tex)`
- `results/tables/table3_scenarios.(csv|tex)`
- `results/tables/table4_fairness_S2.(csv|tex)`
- `results/tables/table5_scheduling_S2.(csv|tex)`

### Figures (PNG + PDF; grayscale + one accent color)
- `results/plots/figure1_reward_S2.(png|pdf)`
- `results/plots/figure2_violations_S2.(png|pdf)`
- `results/plots/figure3_scenario_robustness.(png|pdf)`
- `results/plots/figure4_fairness_S2.(png|pdf)`
- `results/plots/figure5_scheduling_S2.(png|pdf)`

### Executive summary
- `results/executive_summary.md`

> The `results/` and `artifacts/` directories are **generated** and typically should not be committed to git. See `.gitignore`.

---

## Configuration

All experiment settings are controlled via YAML (see `configs/*.yaml`). The main blocks are:

- `experiment.seeds`: list of RNG seeds (controls training + evaluation).
- `experiment.n_eval_episodes`: evaluation episodes per seed and scenario.
- `env.*`: simulator timing, episode duration, residual magnitude limits, and safety shield settings.
- `training.controller.*`: SAC budgets and multi-agent best-response rounds.
- `training.scheduler.*`: PPO budgets.
- `evaluation.*`: ablations and optional trace saving.

Example knobs (from `configs/paper.yaml`):
- `env.control_interval_sec: 6`
- `env.scheduling_interval_sec: 300`
- `env.episode_length_sec: 28800` (8 hours)
- `training.controller.sac_steps` and `training.scheduler.ppo_steps`

---

## Linear data-driven MPC baseline (LinMPC)

To address reviewer requests for a **data-driven MPC** comparison, we include a mode-dependent linear MPC controller that:

- identifies a linear model from **safe excitation data** generated with the real simulator,
- solves a condensed QP online (OSQP),
- outputs a residual correction (12-D) on top of the PI controller,
- can optionally use the same QP safety shield for comparable constraint handling.

### One-shot baseline run
```bash
make linmpc_all
```

### Step-by-step (explicit)
```bash
# 1) Collect identification data (one dataset per operating mode)
python -m baselines.linmpc.collect_id_data --config configs/paper.yaml --out_dir artifacts/linmpc --seed 0   --episodes 1 --steps_per_episode 4800 --burn_in_steps 200 --action_std 0.10 --action_clip 0.25 --use_safety_shield

# 2) Fit a mode-dependent linear model
python -m baselines.linmpc.fit_model --data_dir artifacts/linmpc/data --out_path artifacts/linmpc/model_linmpc.npz

# 3) Evaluate LinMPC and merge results into the same results directory
python -m baselines.linmpc.eval_linmpc --config configs/paper.yaml --model artifacts/linmpc/model_linmpc.npz --results_dir results
```

Notes:
- LinMPC is **self-contained** and does not require RL artifacts or logs (it collects its own identification dataset).
- You may override the configuration used by `make linmpc_all`:
  ```bash
  make linmpc_all CONFIG=configs/paper_strong.yaml
  ```

---

## Reproducibility notes

- All scripts use explicit RNG seeding (NumPy / PyTorch / environment).
- Exact bitwise reproducibility can still vary across platforms due to low-level numeric libraries and PyTorch nondeterminism. For the most stable results:
  - prefer CPU-only runs, or
  - pin versions in a lockfile/conda env, and
  - use the same OS + hardware stack when comparing runs.

---

## Troubleshooting

### Why do I see `Monitor` and `DummyVecEnv` in logs?
Stable-Baselines3 wraps Gymnasium environments in standard wrappers (often `Monitor` and `DummyVecEnv`) to provide a consistent vectorized API. This does **not** mean the plant is a dummy simulation; it is a wrapper around the real environment.

### OSQP installation issues
If OSQP fails to install/build on your platform, upgrade pip and build tools:
```bash
pip install --upgrade pip setuptools wheel
pip install osqp
```

---

## License and attribution

- The repository code is released under the **MIT License** (see `LICENSE`).
- The vendored TEP simulator under `third_party/tep/` retains its **original license and attribution** (see `third_party/tep/LICENSE`).

---

## Citation

If you use this repository, please cite the software and/or the accompanying paper. A machine-readable citation file is provided:

- `CITATION.cff`

You may also reference the repository URL:
- https://github.com/alqithami/AgentTwin
