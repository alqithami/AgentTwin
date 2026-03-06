"""Evaluate the linear data-driven MPC baseline and (optionally) merge into results.

This script is designed to be run **after** the main AgentTwin evaluation has
produced `results/raw/all_episode_metrics.csv`.

It will:
1) Load the fitted linMPC model (see `fit_model.py`).
2) Evaluate linMPC across the same scenario set and seeds as in the YAML config.
3) Append the new rows to `results/raw/all_episode_metrics.csv`.
4) Regenerate tables and plots using the existing result-generation utilities.

Usage
-----
# (1) collect data
python -m baselines.linmpc.collect_id_data --config configs/paper.yaml --out_dir artifacts/linmpc --seed 0

# (2) fit model
python -m baselines.linmpc.fit_model --data_dir artifacts/linmpc/data --out_path artifacts/linmpc/model_linmpc.npz

# (3) evaluate + merge
python -m baselines.linmpc.eval_linmpc --config configs/paper.yaml --model artifacts/linmpc/model_linmpc.npz --results_dir results

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import json
import hashlib

from baselines.linmpc.mpc_controller import LinearResidualMPC, LinMPCConfig

from envs import (
    TEPConfig,
    TEPSchedulingEnv,
    ScenarioType,
    apply_scenario_to_config,
    get_scenario_set,
)


def _scenario_fingerprint(scenario) -> str:
    payload = {
        "scenario": str(scenario.scenario.value),
        "initial_mode": int(scenario.initial_mode),
        "demand_schedule": [(int(t), int(m)) for (t, m) in scenario.demand_schedule],
        "disturbances": [(int(i), int(on), int(dur)) for (i, on, dur) in scenario.disturbances],
        "sensor_bias": {str(k): float(v) for k, v in scenario.sensor_bias.items()},
        "mv_gain_sigma": float(scenario.mv_gain_sigma),
        "mv_bias_sigma": float(scenario.mv_bias_sigma),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _greedy_demand_scheduler(obs: np.ndarray) -> int:
    # Obs layout: [xmeas(41), xmv(12), mode_onehot(6), demand_onehot(6), readiness(3)]
    demand = obs[41 + 12 + 6 : 41 + 12 + 6 + 6]
    return int(np.argmax(demand))


def evaluate_linmpc_on_scenario(
    *,
    method_name: str,
    base_cfg: TEPConfig,
    scenario,
    linmpc: LinearResidualMPC,
    n_episodes: int,
    seed_offset: int,
    train_seed: int,
) -> pd.DataFrame:
    cfg = apply_scenario_to_config(base_cfg, scenario)
    env = TEPSchedulingEnv(cfg, control_policy=linmpc)

    scen_fp = _scenario_fingerprint(scenario)

    rows = []
    for ep in range(int(n_episodes)):
        ep_seed = int(seed_offset + ep)
        obs, _ = env.reset(seed=ep_seed)

        done = False
        tot_reward = 0.0
        violations = 0
        pre_breaches = 0
        shield_acts = 0
        solve_ms = 0.0
        mismatch_seconds = 0
        mode_switches = 0
        group_cost_sum = {"feed": 0.0, "separator": 0.0, "utility": 0.0}

        t_end = 0
        done_reason = "truncated"

        while not done:
            a_sched = _greedy_demand_scheduler(obs)
            obs, r, terminated, truncated, info = env.step(a_sched)

            tot_reward += float(r)
            violations += int(info.get("violations", 0))
            pre_breaches += int(info.get("pre_shield_breaches", 0))
            shield_acts += int(info.get("shield_activations", 0))
            solve_ms += float(info.get("shield_solve_time_ms", 0.0))

            mismatch_seconds += int(info.get("demand_mismatch_seconds", 0))
            mode_switches += int(info.get("mode_switch", 0))
            t_end = int(info.get("t", t_end))

            gtc = info.get("group_tracking_costs", {})
            for g in group_cost_sum.keys():
                if g in gtc:
                    group_cost_sum[g] += float(gtc[g])

            done = bool(terminated or truncated)
            if terminated:
                done_reason = "terminated"

        costs = np.asarray(list(group_cost_sum.values()), dtype=float)
        fairness = float(np.nan)
        if np.all(np.isfinite(costs)) and np.sum(costs) > 0:
            fairness = float((np.sum(costs) ** 2) / (len(costs) * np.sum(costs**2) + 1e-12))

        rows.append(
            {
                "method": method_name,
                "scenario": scenario.scenario.value,
                "episode": int(ep),
                "seed": int(train_seed),
                "eval_seed": int(ep_seed),
                "scenario_fp": str(scen_fp),
                "total_reward": float(tot_reward),
                "violations": int(violations),
                "pre_shield_breaches": int(pre_breaches),
                "shield_activations": int(shield_acts),
                "shield_solve_time_ms": float(solve_ms),
                "demand_mismatch_seconds": int(mismatch_seconds),
                "mode_switches": int(mode_switches),
                "t_end": int(t_end),
                "done_reason": str(done_reason),
                "fairness_jain": float(fairness),
                **{f"cost_{k}": float(v) for k, v in group_cost_sum.items()},
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate linMPC baseline and merge into results")
    ap.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config")
    ap.add_argument("--model", type=str, default="artifacts/linmpc/model_linmpc.npz", help="Fitted model NPZ")
    ap.add_argument("--results_dir", type=str, default="results", help="Results directory")
    ap.add_argument("--method_name", type=str, default="LinMPC (data-driven)", help="Method name in tables")

    # Optional MPC hyperparameters
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--u_bound", type=float, default=1.0)
    ap.add_argument("--q_xmeas", type=float, default=1.0)
    ap.add_argument("--q_xmv", type=float, default=1e-2)
    ap.add_argument("--r_u", type=float, default=1e-1)

    args = ap.parse_args()

    cfg_dict = _load_yaml(Path(args.config))

    seeds: List[int] = [int(x) for x in cfg_dict.get("experiment", {}).get("seeds", [0])]
    n_eval_episodes = int(cfg_dict.get("experiment", {}).get("n_eval_episodes", 10))
    scenario_set_name = str(cfg_dict.get("experiment", {}).get("scenario_set", "paper"))

    env_cfg = cfg_dict.get("env", {})
    base_cfg = TEPConfig(
        seed=0,
        tep_backend=env_cfg.get("tep_backend", "python"),
        control_interval_sec=int(env_cfg.get("control_interval_sec", 6)),
        scheduling_interval_sec=int(env_cfg.get("scheduling_interval_sec", 300)),
        episode_length_sec=int(env_cfg.get("episode_length_sec", 8 * 3600)),
        residual_max=float(env_cfg.get("residual_max", 10.0)),
        warmup_sec=int(env_cfg.get("warmup_sec", 0)),
        pi_update_interval_sec=int(env_cfg.get("pi_update_interval_sec", 1)),
        use_safety_shield=bool(env_cfg.get("use_safety_shield", True)),
        safety_shield_type=str(env_cfg.get("safety_shield_type", "qp")),
    )
    if "qp_shield" in env_cfg:
        try:
            base_cfg.qp_shield = base_cfg.qp_shield.__class__(**env_cfg["qp_shield"])
        except Exception:
            pass

    # Optional reward shaping (kept consistent with the main pipeline)
    rw = env_cfg.get("reward", {})
    if isinstance(rw, dict):
        for k, v in rw.items():
            if hasattr(base_cfg.weights, str(k)):
                try:
                    setattr(base_cfg.weights, str(k), float(v))
                except Exception:
                    pass

    # Load MPC
    mpc_cfg = LinMPCConfig(
        horizon=int(args.horizon),
        u_bound=float(args.u_bound),
        q_xmeas=float(args.q_xmeas),
        q_xmv=float(args.q_xmv),
        r_u=float(args.r_u),
    )
    linmpc = LinearResidualMPC(model_path=args.model, cfg=mpc_cfg)

    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Existing results
    all_path = raw_dir / "all_episode_metrics.csv"
    df_existing = None
    if all_path.exists():
        df_existing = pd.read_csv(all_path)

    all_rows = []

    for seed in seeds:
        scenario_defs = get_scenario_set(
            name=scenario_set_name,
            seed=int(seed),
            episode_length_sec=int(base_cfg.episode_length_sec),
        )
        for scen in scenario_defs:
            df_l = evaluate_linmpc_on_scenario(
                method_name=str(args.method_name),
                base_cfg=base_cfg,
                scenario=scen,
                linmpc=linmpc,
                n_episodes=n_eval_episodes,
                seed_offset=seed * 10_000 + 100 * 5,
                train_seed=int(seed),
            )
            all_rows.append(df_l)

    df_linmpc = pd.concat(all_rows, ignore_index=True)
    df_linmpc.to_csv(raw_dir / "linmpc_episode_metrics.csv", index=False)

    if df_existing is not None:
        df_all = pd.concat([df_existing, df_linmpc], ignore_index=True)
    else:
        df_all = df_linmpc

    df_all.to_csv(all_path, index=False)

    # Regenerate tables/plots using existing utilities
    try:
        from eval.generate_results import make_tables, make_plots, write_executive_summary

        tables_dir = results_dir / "tables"
        plots_dir = results_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        make_tables(df_all, tables_dir)
        make_plots(df_all, plots_dir)
        write_executive_summary(df_all, results_dir / "executive_summary.md")
        print(f"[linmpc] merged results and regenerated tables/plots in: {results_dir}")
    except Exception as e:
        print(f"[linmpc] merged metrics written, but could not regenerate tables/plots: {e}")


if __name__ == "__main__":
    main()
