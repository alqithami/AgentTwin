# AgentTwin — Multi‑Agent Digital Twin for Joint Scheduling & Control (Tennessee Eastman)

This repository contains the code to reproduce the **AgentTwin** results for the Tennessee Eastman (TE) process: multi‑agent reinforcement learning with a runtime safety shield, plus figure/table generation for the paper.

---

## 1) Clone

```bash
git clone https://github.com/alqithami/AgentTwin.git
cd AgentTwin
```

---

## 2) Install

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate agenttwin
pip install -e .
```

### Option B — venv + pip
```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .
```

> **Note**: Apple Silicon users may additionally run the optimized setup script:
> ```bash
> bash setup_apple_m4.sh
> ```

---

## 3) Tennessee Eastman (TE) Simulator — external dependency

The TE simulator is a **third‑party** resource and is **not redistributed** in this repository.

- **Source (original):** N. Lawrence Ricker’s *Tennessee Eastman Challenge Archive* (University of Washington) — <https://depts.washington.edu/control/LARRY/TE/download.html>
- **Install & path setup:** download/unzip the TE archive locally, then point the code to its root via an environment variable:
  ```bash
  export TEP_HOME=/path/to/te-simulator       # Windows (PowerShell): $env:TEP_HOME="C:\te-simulator"
  ```
- **Wrappers/configs here:** our Python wrappers live under `envs/`, and experiments are configured by YAML files in `configs/`.
- **Quick check:** after activating your environment, verify the variable is set:
  ```bash
  python -c "import os; print('TEP_HOME:', os.getenv('TEP_HOME'))"
  ```

---

## 4) Quick smoke test (few episodes)

Runs a small end‑to‑end pipeline to verify the installation and TE integration.

### Using Make (if available)
```bash
make reproduce-quick
```

### Or directly with Python
```bash
python execute_complete_pipeline.py --config configs/quick.yaml --out results/quick
```

Outputs (examples):
- Logs and intermediate artifacts → `results/quick/`
- Any generated preview figures/tables → inside `results/quick/`

---

## 5) Full reproduction

Runs all scenarios/seeds used in the paper and exports **camera‑ready figures** and **CSV tables**.

### Using Make (if available)
```bash
make reproduce
```

### Or directly with Python
```bash
python execute_complete_pipeline.py --config configs/main.yaml --out results/paper --export-figures
```

Expected outputs:
- **Figures (vector PDF + PNG)** exported under `results/paper/` (subfolder `figures/` if configured).
- **Aggregated tables (CSV)** exported under `tables/` (and/or `results/paper/` depending on config).
- A reproducibility log containing commit hash, config, and seed list.

---

## 6) Reproducibility settings (as used in the paper)

- **Seeds × episodes:** 5 seeds × 50 episodes per scenario (5 scenarios) = 1,250 evaluation episodes.
- **Primary metrics:** episode cost (−reward), safety shield **pre‑shield interventions**, **post‑shield hard violations** (target 0).
- **Statistical tests:** Welch’s *t*‑test with Bonferroni correction across 15 planned comparisons (adjusted α = 0.0033).

You can adjust seeds/episodes via the YAML files in `configs/`.

---

## 7) Useful Make targets (if `Makefile` is present)

```bash
# create environment (printed guidance)
make setup

# quick smoke run (few episodes per scenario)
make reproduce-quick

# full results + export camera‑ready figures
make reproduce

# export figures from existing results
make figures

# housekeeping
make clean
```

> If `make` is not available on your system, use the equivalent Python commands shown above.

---

## 8) Troubleshooting

- **TEP_HOME not set / simulator not found**  
  Ensure you downloaded the TE archive (link above) and set `TEP_HOME` to its root folder.

- **Apple Silicon (M‑series) performance**  
  Use the provided `setup_apple_m4.sh` to configure accelerated math backends; verify PyTorch MPS availability:
  ```bash
  python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
  ```

- **Clean and re‑install**  
  ```bash
  conda env remove -n agenttwin || true
  conda env create -f environment.yml
  conda activate agenttwin
  pip install -e .
  ```

---

## 9) Data availability

All reproduction scripts and aggregated tables/figures are provided in this repository. Large raw logs may be generated under `results/` during runs and can be re‑created by executing the pipelines above.

---

## 10) Repository layout (key paths)

```
agents/           # agent definitions and policies
configs/          # experiment YAML configs (quick.yaml, main.yaml, ...)
control/          # control logic components
envs/             # Tennessee Eastman environment wrappers and helpers
eval/             # evaluation utilities
results/          # outputs (created by runs)
logs/             # logs (created by runs)
shield/           # safety shield implementation
tables/           # aggregated CSV tables (exported)
train/            # training helpers
execute_complete_pipeline.py  # end‑to‑end runner
environment.yml   # Conda environment (if using conda)
requirements.txt  # pip requirements (if using venv)
Makefile          # convenience targets (optional but included)
README.md         # this file
```

---

### Citation
If you use this code in your research, please cite the AgentTwin manuscript (and this repository). A `CITATION.cff` is included in the repo.

---

**Happy experimenting!** 🚀
