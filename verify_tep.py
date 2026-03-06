"""Verify that the Tennessee Eastman (TE) simulator and environments are runnable.

This script is intentionally lightweight:
- No Stable-Baselines3 required
- No training

Run:
    python verify_tep.py

Expected:
- measurements: (41,)
- manipulated variables: (12,)
- internal states: (50,)

The presence of large-magnitude flows/pressures/temperatures in the printed
measurements is normal for the canonical TE benchmark.
"""

from __future__ import annotations

import numpy as np

from third_party.tep.simulator import TEPSimulator, ControlMode

from envs.tep_env import (
    TEPConfig,
    TEPContinuousControlEnv,
    TEPSchedulingEnv,
    ScenarioType,
)


def main() -> None:
    print("[1/3] TEPSimulator smoke test")
    sim = TEPSimulator(backend="python", control_mode=ControlMode.CLOSED_LOOP, random_seed=0)
    sim.initialize()

    y0 = sim.get_measurements().copy()
    u0 = sim.get_manipulated_vars().copy()
    x0 = sim.get_states().copy()

    print("  measurements shape        :", y0.shape)
    print("  manipulated vars shape    :", u0.shape)
    print("  internal states shape     :", x0.shape)
    print("  initial measurements[:10] :", np.array2string(y0[:10], precision=4))
    print("  initial MVs (12)          :", np.array2string(u0, precision=4))

    # Run 10 minutes of simulation (600 seconds)
    for _ in range(600):
        sim.step()

    y = sim.get_measurements()
    print("  t (hours) after 10 min    :", f"{sim.time:.4f}")
    print("  reactor pressure (y[6])   :", f"{y[6]:.2f}")
    print("  reactor temperature (y[8]):", f"{y[8]:.2f}")

    print("\n[2/3] TEPContinuousControlEnv smoke test")
    cfg = TEPConfig(
        seed=0,
        episode_length_sec=10 * 60,
        control_interval_sec=6,
        scheduling_interval_sec=300,
        scenario=ScenarioType.S1_NOMINAL,
        use_safety_shield=True,
    )

    # Some earlier variants of the env constructors accepted `initial_mode`.
    # In this package we set it through the config for compatibility.
    cfg.initial_mode = 1
    env = TEPContinuousControlEnv(config=cfg)
    obs, _ = env.reset(seed=0)

    for _ in range(10):
        # zero residual action
        obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        if terminated or truncated:
            break

    print("  last reward               :", f"{reward:.4f}")
    print("  violations                :", info.get("violations"))
    print("  shield_activations              :", info.get("shield_activations"))

    print("\n[3/3] TEPSchedulingEnv smoke test")
    sched_env = TEPSchedulingEnv(config=cfg, control_policy=None)
    obs, _ = sched_env.reset(seed=0)

    obs, reward, terminated, truncated, info = sched_env.step(0)  # choose mode 1
    print("  one step reward           :", f"{reward:.4f}")
    print("  mode / demand_mode        :", info.get("mode"), "/", info.get("demand_mode"))
    print("  violations / shield_activations :", info.get("violations"), "/", info.get("shield_activations"))

    print("\nOK: TE simulator + environments are running.")


if __name__ == "__main__":
    main()
