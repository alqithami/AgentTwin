"""Lightweight scenario sanity-check.

This script is intended to answer a recurring question in review/reproduction:
"Are the scenarios actually different and are disturbances being applied?"

It runs ONE short episode per scenario with a greedy demand-following scheduler
and the default PI controller (residual=0), then prints:
- scenario fingerprint
- the first few demand schedule entries
- whether any disturbances became active
- scheduling adherence metrics from the environment

Run:
    python verify_scenarios.py --config configs/quick.yaml

"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from envs import TEPConfig, TEPSchedulingEnv, apply_scenario_to_config, get_scenario_set


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


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


def _greedy_demand_scheduler(obs: np.ndarray) -> int:
    # Obs layout: [xmeas(41), xmv(12), mode_onehot(6), demand_onehot(6), readiness(3)]
    demand = obs[41 + 12 + 6 : 41 + 12 + 6 + 6]
    return int(np.argmax(demand))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scenario sanity check")
    ap.add_argument("--config", type=str, default="configs/quick.yaml", help="YAML config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    cfg_dict = _load_yaml(Path(args.config))

    env_cfg = cfg_dict.get("env", {})
    exp_cfg = cfg_dict.get("experiment", {})

    base_cfg = TEPConfig(
        seed=int(args.seed),
        tep_backend=str(env_cfg.get("tep_backend", "python")),
        control_interval_sec=int(env_cfg.get("control_interval_sec", 6)),
        scheduling_interval_sec=int(env_cfg.get("scheduling_interval_sec", 300)),
        episode_length_sec=int(env_cfg.get("episode_length_sec", 3600)),
        residual_max=float(env_cfg.get("residual_max", 10.0)),
        warmup_sec=int(env_cfg.get("warmup_sec", 0)),
        pi_update_interval_sec=int(env_cfg.get("pi_update_interval_sec", 1)),
        use_safety_shield=bool(env_cfg.get("use_safety_shield", True)),
        safety_shield_type=str(env_cfg.get("safety_shield_type", "qp")),
    )

    # Optional reward shaping
    rw = env_cfg.get("reward", {})
    if isinstance(rw, dict):
        for k, v in rw.items():
            if hasattr(base_cfg.weights, str(k)):
                try:
                    setattr(base_cfg.weights, str(k), float(v))
                except Exception:
                    pass

    scenario_set = str(exp_cfg.get("scenario_set", "minimal"))
    scenarios = get_scenario_set(name=scenario_set, seed=int(args.seed), episode_length_sec=int(base_cfg.episode_length_sec))

    print(f"Scenario set: {scenario_set} (n={len(scenarios)})")

    for scen in scenarios:
        cfg = apply_scenario_to_config(base_cfg, scen)
        env = TEPSchedulingEnv(cfg, control_policy=None)

        fp = _scenario_fingerprint(scen)
        sched_preview = scen.demand_schedule[:5]
        print("\n---")
        print(f"{scen.scenario.value}: fp={fp}")
        print(f"  initial_mode: {scen.initial_mode}")
        print(f"  demand_schedule (first 5): {sched_preview}{' ...' if len(scen.demand_schedule)>5 else ''}")
        print(f"  disturbances: {scen.disturbances}")

        for ep in range(int(args.episodes)):
            obs, _ = env.reset(seed=int(args.seed + 1000 * ep))
            done = False
            tot_reward = 0.0
            mismatch_sec = 0
            switches = 0
            active = set()

            while not done:
                a = _greedy_demand_scheduler(obs)
                obs, r, terminated, truncated, info = env.step(a)
                tot_reward += float(r)
                mismatch_sec += int(info.get("demand_mismatch_seconds", 0))
                switches += int(info.get("mode_switch", 0))
                for d in info.get("active_disturbances", []) or []:
                    active.add(int(d))
                done = bool(terminated or truncated)

            print(
                f"  ep{ep}: R={tot_reward:.2f}, mismatch_sec={mismatch_sec}, mode_switches={switches}, active_disturbances={sorted(active)}"
            )

    print("\nOK: scenarios are instantiated and exercised.")


if __name__ == "__main__":
    main()
