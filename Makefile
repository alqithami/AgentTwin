PY=python

# Default configuration files
CONFIG_QUICK=configs/quick.yaml
CONFIG_PAPER=configs/paper.yaml
CONFIG_PAPER_STRONG=configs/paper_strong.yaml

# Default config used by baseline targets (can be overridden), e.g.:
#   make linmpc_all CONFIG=configs/paper_strong.yaml
CONFIG?=$(CONFIG_PAPER)

ARTIFACTS?=artifacts
RESULTS?=results

# linMPC baseline (data-driven linear MPC)
LINMPC_DIR?=$(ARTIFACTS)/linmpc
LINMPC_MODEL?=$(LINMPC_DIR)/model_linmpc.npz

.PHONY: help verify verify_scenarios reproduce_quick reproduce reproduce_strong reproduce_full linmpc_collect linmpc_fit linmpc_eval linmpc_all

help:
	@echo "Targets:"
	@echo "  make verify               - run TE simulator sanity check"
	@echo "  make verify_scenarios      - print scenario fingerprints and a short rollout per scenario"
	@echo "  make reproduce_quick      - quick end-to-end run (smoke test)"
	@echo "  make reproduce            - paper-style end-to-end run (full paper config)"
	@echo "  make reproduce_strong     - stronger stats (more steps/seeds) config"
	@echo "  make reproduce_full       - reproduce + linMPC baseline + regenerated figures/tables"
	@echo "  make linmpc_all           - collect ID data, fit linMPC model, evaluate+merge"
	@echo "Variables:"
	@echo "  ARTIFACTS=<dir> RESULTS=<dir>"
	@echo "  LINMPC_DIR=<dir> LINMPC_MODEL=<file>"

verify:
	$(PY) verify_tep.py

verify_scenarios:
	$(PY) verify_scenarios.py --config $(CONFIG_QUICK)

reproduce_quick:
	$(PY) -m train.training_pipeline --config $(CONFIG_QUICK) --out_root $(ARTIFACTS) --seeds 0
	$(PY) -m eval.generate_results --config $(CONFIG_QUICK) --artifacts_root $(ARTIFACTS) --results_dir $(RESULTS)

reproduce:
	$(PY) -m train.training_pipeline --config $(CONFIG_PAPER) --out_root $(ARTIFACTS)
	$(PY) -m eval.generate_results --config $(CONFIG_PAPER) --artifacts_root $(ARTIFACTS) --results_dir $(RESULTS)

reproduce_strong:
	$(PY) -m train.training_pipeline --config $(CONFIG_PAPER_STRONG) --out_root $(ARTIFACTS)
	$(PY) -m eval.generate_results --config $(CONFIG_PAPER_STRONG) --artifacts_root $(ARTIFACTS) --results_dir $(RESULTS)

reproduce_full: reproduce linmpc_all

# -----------------------------------------------------------------------------
# linMPC baseline
# -----------------------------------------------------------------------------

linmpc_collect:
	$(PY) -m baselines.linmpc.collect_id_data \
		--config $(CONFIG) \
		--out_dir $(LINMPC_DIR) \
		--seed 0 \
		--episodes 1 \
		--steps_per_episode 4800 \
		--burn_in_steps 200 \
		--action_std 0.10 \
		--action_clip 0.25 \
		--use_safety_shield

linmpc_fit:
	$(PY) -m baselines.linmpc.fit_model --data_dir $(LINMPC_DIR)/data --out_path $(LINMPC_MODEL)

linmpc_eval:
	$(PY) -m baselines.linmpc.eval_linmpc --config $(CONFIG) --model $(LINMPC_MODEL) --results_dir $(RESULTS)

linmpc_all: linmpc_collect linmpc_fit linmpc_eval
