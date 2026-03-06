"""Scenario definitions for AgentTwin experiments.

The reproduction pipeline evaluates AgentTwin under five scenarios (S1--S5).

Each scenario can specify:
- A *demand profile* (desired TE operating mode / grade)
- A list of *TE disturbances* (IDV index, start time, duration)
- Optional *sensor bias* injected into the observation stream
- Optional *actuator mismatch* (gain/bias) applied to commanded MVs

We keep backwards-compatible aliases ("normal", "dynamic_demand", ...)
by mapping them to the S1--S5 identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ScenarioType(str, Enum):
    # Paper scenarios
    S1_NOMINAL = "S1_nominal"
    S2_DYNAMIC_DEMAND = "S2_dynamic_demand"
    S3_FAULT_DISTURBANCE = "S3_fault_disturbance"
    S4_SENSOR_BIAS = "S4_sensor_bias"
    S5_MODEL_MISMATCH = "S5_model_mismatch"

    # Backwards-compatible aliases
    NORMAL = "S1_nominal"
    DYNAMIC_DEMAND = "S2_dynamic_demand"
    DISTURBANCE = "S3_fault_disturbance"
    FAULT_DISTURBANCE = "S3_fault_disturbance"
    SENSOR_BIAS = "S4_sensor_bias"
    MODEL_MISMATCH = "S5_model_mismatch"


@dataclass
class ScenarioDefinition:
    """A fully specified evaluation scenario."""

    scenario: ScenarioType
    initial_mode: int = 1

    # Exogenous demand profile: list of (t_sec, demanded_mode)
    # The demanded mode is held constant from its time until the next change.
    demand_schedule: List[Tuple[int, int]] = field(default_factory=list)

    # If demand_schedule is empty, we can define a stochastic demand change law.
    demand_change_prob: float = 0.0
    demand_interval_sec: int = 1800

    # Process disturbances: list of (idv_idx, onset_sec, duration_sec)
    disturbances: List[Tuple[int, int, int]] = field(default_factory=list)

    # Sensor bias: dict {xmeas_index (0-based): bias_value}
    sensor_bias: Dict[int, float] = field(default_factory=dict)

    # Actuator mismatch: multiplicative gain and additive bias on *applied* MVs.
    # If mv_gain_sigma > 0 or mv_bias_sigma > 0, per-episode gains/biases are
    # sampled i.i.d. for each MV on reset.
    mv_gain_sigma: float = 0.0
    mv_bias_sigma: float = 0.0


def make_random_demand_schedule(
    *,
    seed: int,
    episode_length_sec: int,
    interval_sec: int = 1800,
    p_change: float = 0.35,
    modes: Sequence[int] = (1, 2, 3, 4, 5, 6),
    start_mode: int = 1,
) -> List[Tuple[int, int]]:
    """Generate a deterministic random demand schedule.

    The schedule is deterministic given `seed` and is intended for repeatable
    evaluations.
    """

    rng = np.random.default_rng(int(seed))
    schedule: List[Tuple[int, int]] = [(0, int(start_mode))]

    cur = int(start_mode)
    for t in range(interval_sec, int(episode_length_sec) + 1, interval_sec):
        if rng.random() < float(p_change):
            # Choose a new mode different from current
            candidates = [m for m in modes if int(m) != cur]
            cur = int(rng.choice(candidates))
            schedule.append((int(t), cur))

    # Guardrail for reproducibility: ensure S2/S3/... are truly "dynamic".
    # With small probability, the Bernoulli process may yield no changes; in
    # that case we force a single change at mid-episode.
    if len(schedule) == 1:
        t_mid = int(max(interval_sec, episode_length_sec // 2))
        candidates = [m for m in modes if int(m) != cur]
        cur = int(rng.choice(candidates))
        schedule.append((t_mid, cur))

    return schedule


def scenario_set_paper(seed: int, episode_length_sec: int = 8 * 3600) -> List[ScenarioDefinition]:
    """Return the S1--S5 scenario set used for the paper-style evaluation."""

    # S1: scripted nominal grade/mode changes (no disturbances)
    s1 = ScenarioDefinition(
        scenario=ScenarioType.S1_NOMINAL,
        initial_mode=1,
        demand_schedule=[
            (0, 1),
            (2 * 3600, 2),
            (4 * 3600, 3),
            (6 * 3600, 1),
        ],
    )

    # S2: stochastic demand profile (deterministic given seed)
    sched_s2 = make_random_demand_schedule(
        seed=seed,
        episode_length_sec=episode_length_sec,
        interval_sec=1800,
        p_change=0.35,
        start_mode=1,
    )

    s2 = ScenarioDefinition(
        scenario=ScenarioType.S2_DYNAMIC_DEMAND,
        initial_mode=1,
        demand_schedule=sched_s2,
    )

    # S3: same demand profile + representative fault/disturbance
    # IDV(4): reactor cooling water inlet temperature (step)
    s3 = ScenarioDefinition(
        scenario=ScenarioType.S3_FAULT_DISTURBANCE,
        initial_mode=1,
        demand_schedule=sched_s2,
        disturbances=[(4, 2 * 3600, 2 * 3600)],
    )

    # S4: same demand profile + sensor bias injected into a subset of measurements
    # Bias values are conservative and can be tuned.
    s4 = ScenarioDefinition(
        scenario=ScenarioType.S4_SENSOR_BIAS,
        initial_mode=1,
        demand_schedule=sched_s2,
        sensor_bias={
            8: +1.0,   # reactor temperature (degC)
            39: +2.0,  # product G composition (mol%)
            40: -2.0,  # product H composition (mol%)
        },
    )

    # S5: same demand profile + actuator mismatch (valve gain/bias)
    s5 = ScenarioDefinition(
        scenario=ScenarioType.S5_MODEL_MISMATCH,
        initial_mode=1,
        demand_schedule=sched_s2,
        mv_gain_sigma=0.05,
        mv_bias_sigma=0.5,
    )

    return [s1, s2, s3, s4, s5]


def scenario_set_minimal(seed: int, episode_length_sec: int = 8 * 3600) -> List[ScenarioDefinition]:
    """A smaller scenario set for quick iteration."""

    sched = make_random_demand_schedule(
        seed=seed,
        episode_length_sec=episode_length_sec,
        interval_sec=1800,
        p_change=0.35,
        start_mode=1,
    )

    s1 = ScenarioDefinition(scenario=ScenarioType.S1_NOMINAL, initial_mode=1, demand_schedule=[(0, 1)])
    s2 = ScenarioDefinition(scenario=ScenarioType.S2_DYNAMIC_DEMAND, initial_mode=1, demand_schedule=sched)
    s3 = ScenarioDefinition(
        scenario=ScenarioType.S3_FAULT_DISTURBANCE,
        initial_mode=1,
        demand_schedule=sched,
        disturbances=[(4, 2 * 3600, 2 * 3600)],
    )

    return [s1, s2, s3]


def get_scenario_set(name: str, seed: int, episode_length_sec: int) -> List[ScenarioDefinition]:
    """Return a scenario set by name."""

    key = str(name).strip().lower()
    if key in {"paper", "s1-s5", "s1_s5"}:
        return scenario_set_paper(seed=seed, episode_length_sec=episode_length_sec)
    if key in {"minimal", "quick"}:
        return scenario_set_minimal(seed=seed, episode_length_sec=episode_length_sec)

    raise ValueError(f"Unknown scenario_set: {name!r}. Use 'paper' or 'minimal'.")
