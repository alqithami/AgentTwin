"""Quick smoke test.

Run this after installing requirements to verify that:
- The vendored Tennessee Eastman simulator loads.
- Gymnasium environments step without errors.

This test is intentionally lightweight; it does not train models.
"""

from __future__ import annotations

import numpy as np

from envs.tep_env import TEPConfig, TEPSchedulingEnv
from envs.scenarios import ScenarioType


def greedy_scheduler_policy(obs: np.ndarray) -> int:
    """Pick the current demand mode (greedy baseline)."""
    # TEPSchedulingEnv obs = [xmeas(41), xmv(12), mode_one_hot(6), demand_one_hot(6), readiness(3)]
    demand_one_hot = obs[41 + 12 + 6 : 41 + 12 + 12]
    return int(np.argmax(demand_one_hot))


def main() -> None:
    cfg = TEPConfig(
        seed=0,
        episode_length_sec=30 * 60,  # 30 minutes
        control_interval_sec=6,
        scheduling_interval_sec=300,
        scenario=ScenarioType.S2_DYNAMIC_DEMAND,
        use_safety_shield=True,
    )

    env = TEPSchedulingEnv(config=cfg, control_policy=None)
    obs, info = env.reset(seed=0)

    total_reward = 0.0
    total_violations = 0
    total_shields = 0
    n_steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = greedy_scheduler_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_violations += int(info.get("violations", 0))
        total_shields += int(info.get("shield_activations", 0))
        n_steps += 1

    print("Quick test completed")
    print(f"  scheduling_steps : {n_steps}")
    print(f"  total_reward     : {total_reward:.3f}")
    print(f"  final_mode       : {info.get('mode')}")
    print(f"  final_demand_mode: {info.get('demand_mode')}")
    print(f"  total_violations : {total_violations}")
    print(f"  shield_activations: {total_shields}")


if __name__ == "__main__":
    main()
