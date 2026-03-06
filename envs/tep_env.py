"""Tennessee Eastman Process (TEP) Gymnasium environments.

This module provides **real** Tennessee Eastman Process dynamics via the vendored
`third_party.tep` simulator.

Environment design
------------------
- Continuous control env: residual RL (continuous action) on top of the
  standard decentralized TE PI controller.
- Scheduling env: discrete operating-mode switching (modes 1--6) on a slower
  time scale, combined with the residual controllers.

The goal is to provide a runnable, end-to-end reproduction pipeline with:
- deterministic seeds
- scenario-driven evaluation (S1--S5)
- optional QP-based safety shield

Note on DummyVecEnv
-------------------
Stable-Baselines3 uses vectorized environments. `DummyVecEnv` is simply a
single-process vector wrapper and does not imply a "fake" environment.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required. Install dependencies via: pip install -r requirements.txt"
    ) from e

from third_party.tep.simulator import TEPSimulator, ControlMode
from third_party.tep.controllers import DecentralizedController
from third_party.tep.constants import MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES, OPERATING_MODES

from envs.scenarios import ScenarioType, ScenarioDefinition
from envs.shield import SafetyShieldHeuristic, SafetyShieldQP, QPShieldConfig


# =============================================================================
# MV groups (for multi-agent control)
# =============================================================================
# 0-based MV indices; group names should match evaluation scripts.
MV_GROUPS: Dict[str, List[int]] = {
    "feed": [0, 1, 2, 3],          # feed valves/flows
    "separator": [4, 5, 6, 7],     # separator + purge related
    "utility": [8, 9, 10, 11],     # utilities (cooling water, steam, etc.)
}


# =============================================================================
# Configuration dataclasses
# =============================================================================
@dataclass
class SafetyLimits:
    """Key safety limits used by the reproduction package.

    These limits are intentionally conservative and are enforced through:
    - hard termination when the TE simulator reports shutdown
    - soft penalties in the reward
    - optional safety shield for residual actions
    """

    reactor_pressure_max: float = 3000.0      # kPa
    reactor_temp_max: float = 150.0          # degC
    sep_pressure_max: float = 3500.0         # kPa
    stripper_pressure_max: float = 3300.0    # kPa
    reactor_level_max: float = 90.0          # %
    reactor_level_min: float = 10.0          # %


@dataclass
class RewardWeights:
    """Reward/cost weights."""

    tracking: float = 1.0
    utilities: float = 1e-3
    mv_move: float = 1e-3
    constraint_violation: float = 50.0

    # Scheduling-level penalties (used in TEPSchedulingEnv)
    # NOTE: defaults are 0.0 to preserve backwards compatibility unless
    # explicitly enabled from YAML.
    demand_mismatch: float = 0.0
    mode_switch: float = 0.0


@dataclass
class TEPConfig:
    """Unified configuration for TEP control + scheduling experiments."""

    # Reproducibility
    seed: int = 0

    # Simulator
    tep_backend: str = "python"  # 'python' | 'fortran' | None('auto')

    # Episode timing
    episode_length_sec: int = 8 * 3600
    control_interval_sec: int = 6
    scheduling_interval_sec: int = 300
    warmup_sec: int = 0

    # Policy structure
    residual_max: float = 10.0
    pi_update_interval_sec: int = 1

    # Scenario / initial condition
    scenario: ScenarioType = ScenarioType.S1_NOMINAL
    initial_mode: int = 1
    randomize_initial_mode: bool = False
    follow_demand_in_training: bool = False

    # Demand profile
    demand_schedule: List[Tuple[int, int]] = field(default_factory=list)
    demand_change_prob: float = 0.0
    demand_interval_sec: int = 1800

    # Disturbances
    disturbances: List[Tuple[int, int, int]] = field(default_factory=list)  # (idv_idx, onset_sec, duration_sec)

    # Sensor bias (observation stream)
    sensor_bias: Dict[int, float] = field(default_factory=dict)

    # Actuator mismatch (applied MV gain/bias)
    mv_gain_sigma: float = 0.0
    mv_bias_sigma: float = 0.0

    # Safety
    use_safety_shield: bool = True
    safety_shield_type: str = "qp"  # 'qp' | 'heuristic' | 'off'
    qp_shield: QPShieldConfig = field(default_factory=QPShieldConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    # Reward weights
    weights: RewardWeights = field(default_factory=RewardWeights)

    # Logging/diagnostics
    record_trace: bool = False

    # Backwards-compatible alias (some earlier scripts used this name)
    scheduler_interval_sec: Optional[int] = None

    def __post_init__(self) -> None:
        if self.scheduler_interval_sec is not None:
            # Maintain backwards compatibility
            self.scheduling_interval_sec = int(self.scheduler_interval_sec)

        self.control_interval_sec = int(self.control_interval_sec)
        self.scheduling_interval_sec = int(self.scheduling_interval_sec)
        self.episode_length_sec = int(self.episode_length_sec)
        self.warmup_sec = int(self.warmup_sec)
        self.pi_update_interval_sec = int(self.pi_update_interval_sec)

        if self.tep_backend in {"auto", "none"}:
            self.tep_backend = None  # let simulator auto-select

        # Normalize shield type
        self.safety_shield_type = str(self.safety_shield_type).strip().lower()
        if not self.use_safety_shield or self.safety_shield_type in {"off", "none", "false"}:
            self.use_safety_shield = False
            self.safety_shield_type = "off"


# =============================================================================
# Helpers
# =============================================================================
_DEF_MAX_XMEAS = np.array([
    1.0, 10000.0, 10000.0, 20.0, 1.0, 1.0, 5000.0, 100.0, 200.0, 1.0,
    1.0, 100.0, 5000.0, 1.0, 100.0, 5000.0, 1.0, 200.0, 1.0, 1.0,
    1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0
], dtype=np.float32)

_DEF_MAX_XMV = np.array([100.0] * 12, dtype=np.float32)


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[int(idx)] = 1.0
    return v


# =============================================================================
# Plant wrapper
# =============================================================================
class TEPPlant:
    """A thin wrapper around the TEPSimulator for residual control."""

    def __init__(self, config: TEPConfig):
        self.config = config

        self.sim = TEPSimulator(
            control_mode=ControlMode.OPEN_LOOP,
            backend=config.tep_backend,
            random_seed=int(config.seed),
        )
        self.sim.initialize()

        self.base_ctrl = DecentralizedController(mode=int(config.initial_mode))
        self.base_ctrl.reset()

        self.current_mode = int(config.initial_mode)
        self.t = 0  # seconds since episode start

        # Disturbance bookkeeping
        self._disturbances = list(config.disturbances)
        self._active_disturbances: Dict[int, int] = {}

        # Sensor bias vector (41)
        self._sensor_bias = np.zeros(41, dtype=np.float32)
        for k, v in config.sensor_bias.items():
            if 0 <= int(k) < 41:
                self._sensor_bias[int(k)] = float(v)

        # Actuator mismatch parameters (sampled on reset)
        self.mv_gain = np.ones(12, dtype=np.float32)
        self.mv_bias = np.zeros(12, dtype=np.float32)
        self._rng = np.random.default_rng(int(config.seed))
        self._sample_actuator_mismatch()

        # Safety shield
        self.shield = None
        if config.use_safety_shield:
            if config.safety_shield_type == "heuristic":
                self.shield = SafetyShieldHeuristic(config.safety)
            elif config.safety_shield_type == "qp":
                try:
                    self.shield = SafetyShieldQP(
                        limits=config.safety,
                        residual_max=config.residual_max,
                        cfg=config.qp_shield,
                        backend=(config.tep_backend or "python"),
                        seed=int(config.seed),
                        dt_sec=1,
                    )
                except Exception as e:
                    # Fail gracefully to heuristic rather than hard-crashing
                    print(f"[WARN] QP shield unavailable ({e}); falling back to heuristic shield.")
                    self.shield = SafetyShieldHeuristic(config.safety)
            else:
                self.shield = None

        # Cache for MV move penalty
        self._prev_xmv = self.sim.process.get_xmv().copy()

        # Optional trace
        self._trace: Dict[str, List[Any]] = {
            "t": [],
            "mode": [],
            "xmeas": [],
            "xmv": [],
            "residual": [],
            "reward": [],
        }

        if config.warmup_sec > 0:
            self._warmup(int(config.warmup_sec))

    def _warmup(self, seconds: int) -> None:
        # Run the baseline PI controller for a short warm-up to settle.
        for _ in range(int(seconds)):
            xmeas = self.get_xmeas_measured()
            xmv = self.sim.process.get_xmv()
            base_xmv = self.base_ctrl.calculate(xmeas, xmv, self.t)
            for i in range(12):
                self.sim.set_mv(i + 1, float(np.clip(base_xmv[i], 0.0, 100.0)))
            ok = self.sim.step(1)
            self.t += 1
            if not ok:
                break

    def _sample_actuator_mismatch(self) -> None:
        cfg = self.config
        if cfg.mv_gain_sigma > 0:
            self.mv_gain = (1.0 + self._rng.normal(0.0, float(cfg.mv_gain_sigma), size=12)).astype(np.float32)
        else:
            self.mv_gain = np.ones(12, dtype=np.float32)

        if cfg.mv_bias_sigma > 0:
            self.mv_bias = self._rng.normal(0.0, float(cfg.mv_bias_sigma), size=12).astype(np.float32)
        else:
            self.mv_bias = np.zeros(12, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, initial_mode: Optional[int] = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        self._rng = np.random.default_rng(int(self.config.seed))

        self.sim = TEPSimulator(
            control_mode=ControlMode.OPEN_LOOP,
            backend=self.config.tep_backend,
            random_seed=int(self.config.seed),
        )
        self.sim.initialize()

        mode = int(initial_mode) if initial_mode is not None else int(self.config.initial_mode)
        self.current_mode = mode

        self.base_ctrl = DecentralizedController(mode=mode)
        self.base_ctrl.reset()

        self.t = 0
        self._active_disturbances.clear()
        self.sim.clear_disturbances()
        self._sample_actuator_mismatch()

        self._prev_xmv = self.sim.process.get_xmv().copy()

        for k in self._trace:
            self._trace[k] = []

        if self.config.warmup_sec > 0:
            self._warmup(int(self.config.warmup_sec))

    def set_mode(self, mode: int) -> None:
        mode = int(mode)
        if mode not in OPERATING_MODES:
            raise ValueError(f"Invalid mode {mode}; expected 1..6")
        if mode == self.current_mode:
            return
        self.current_mode = mode
        self.base_ctrl.set_mode(mode)

    def get_xmeas_true(self) -> np.ndarray:
        return self.sim.process.get_xmeas().copy().astype(np.float32)

    def get_xmeas_measured(self) -> np.ndarray:
        # Observation stream (true + bias)
        return (self.get_xmeas_true() + self._sensor_bias).astype(np.float32)

    def get_xmv(self) -> np.ndarray:
        return self.sim.process.get_xmv().copy().astype(np.float32)

    def _apply_disturbances(self) -> None:
        # Disturbances are specified as (idv_idx, onset_sec, duration_sec)
        for idv_idx, onset, dur in self._disturbances:
            idv = int(idv_idx)
            start = int(onset)
            end = int(onset) + int(dur)

            if start <= self.t < end:
                if self._active_disturbances.get(idv, 0) == 0:
                    self.sim.set_disturbance(idv, 1)
                    self._active_disturbances[idv] = 1
            else:
                if self._active_disturbances.get(idv, 0) == 1:
                    self.sim.set_disturbance(idv, 0)
                    self._active_disturbances[idv] = 0

    def _tracking_costs_by_group(self, xmeas_true: np.ndarray) -> Dict[str, float]:
        # Map a subset of setpoint indices to agent groups.
        # This is used for fairness/payoff analysis; it does not affect training.
        group_meas = {
            "feed": {0, 1, 2, 3, 4, 5, 9, 39, 40},
            "separator": {11, 12, 14, 15, 7},
            "utility": {6, 8, 10, 17, 18, 19, 20, 21},
        }
        mode_cfg = OPERATING_MODES[int(self.current_mode)]
        sp = mode_cfg.xmeas_setpoints
        costs: Dict[str, float] = {k: 0.0 for k in group_meas}

        for idx, sp_val in sp.items():
            idx_i = int(idx)
            err = float(xmeas_true[idx_i] - float(sp_val)) / float(_DEF_MAX_XMEAS[idx_i])
            e2 = err * err
            for g, idx_set in group_meas.items():
                if idx_i in idx_set:
                    costs[g] += e2

        return costs

    def run(self, seconds: int, residual: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run the plant forward for `seconds` at 1s resolution.

        Parameters
        ----------
        seconds:
            Number of 1-second integration steps.
        residual:
            Residual action (12,) applied additively to the baseline PI.

        Returns
        -------
        obs_measured : np.ndarray
            The measured observation at the end of the interval.
        reward_sum : float
            Accumulated reward over the interval.
        terminated : bool
            True if simulator shuts down.
        truncated : bool
            True if horizon reached (handled by gym env).
        info : dict
            Step diagnostics.
        """

        cfg = self.config
        residual = np.asarray(residual, dtype=np.float32).reshape(12)
        residual = np.clip(residual, -cfg.residual_max, cfg.residual_max)

        reward_sum = 0.0
        violations = 0
        pre_shield_breaches = 0
        shield_activations = 0
        shield_solve_time_ms = 0.0

        group_costs_accum = {"feed": 0.0, "separator": 0.0, "utility": 0.0}

        terminated = False

        # For logging/identification: average residual after shielding over the interval
        res_applied_sum = np.zeros(12, dtype=np.float32)
        res_applied_n = 0

        for _ in range(int(seconds)):
            self._apply_disturbances()

            xmeas_meas = self.get_xmeas_measured()
            xmeas_true = self.get_xmeas_true()
            xmv = self.get_xmv()

            # Baseline PI action (update every pi_update_interval_sec)
            if (self.t % cfg.pi_update_interval_sec) == 0:
                base_xmv = self.base_ctrl.calculate(xmeas_meas, xmv, self.t)
            else:
                base_xmv = xmv.copy()

            # Safety shield acts on residual only
            res_filt = residual.copy()
            if self.shield is not None:
                res_filt, s_info = self.shield.filter(xmeas_meas, residual, cfg.residual_max)
                shield_activations += int(float(s_info.get("activated", 0.0)) > 0.5)
                shield_solve_time_ms += float(s_info.get("solve_time_ms", 0.0))
                # pre-shield breach proxy: constraints violated for nominal residual
                if float(s_info.get("activated", 0.0)) > 0.5:
                    pre_shield_breaches += 1

            res_applied_sum += res_filt.astype(np.float32)
            res_applied_n += 1

            # Commanded MV, with mismatch on *applied* MV
            cmd = np.clip(base_xmv + res_filt, 0.0, 100.0)
            applied = np.clip(self.mv_gain * cmd + self.mv_bias, 0.0, 100.0)

            for i in range(12):
                self.sim.set_mv(i + 1, float(applied[i]))

            ok = self.sim.step(1)
            self.t += 1

            xmeas_true_next = self.get_xmeas_true()
            xmv_next = self.get_xmv()

            # Costs
            mode_cfg = OPERATING_MODES[int(self.current_mode)]
            sp = mode_cfg.xmeas_setpoints

            tracking_cost = 0.0
            for idx, sp_val in sp.items():
                i = int(idx)
                err = float(xmeas_true_next[i] - float(sp_val)) / float(_DEF_MAX_XMEAS[i])
                tracking_cost += err * err

            # Utility proxy: penalize utilities MV deviation from nominal setpoints
            util_idx = MV_GROUPS["utility"]
            utilities_cost = float(np.mean(np.abs(xmv_next[util_idx] - mode_cfg.xmv_setpoints[util_idx])))

            mv_move_cost = float(np.mean((xmv_next - self._prev_xmv) ** 2))
            self._prev_xmv = xmv_next.copy()

            # Constraint violations (hard)
            lim = cfg.safety
            if (
                xmeas_true_next[6] > lim.reactor_pressure_max
                or xmeas_true_next[8] > lim.reactor_temp_max
                or xmeas_true_next[12] > lim.sep_pressure_max
                or xmeas_true_next[15] > lim.stripper_pressure_max
                or xmeas_true_next[7] > lim.reactor_level_max
                or xmeas_true_next[7] < lim.reactor_level_min
            ):
                violations += 1

            # Soft constraint penalty
            constraint_cost = 0.0
            constraint_cost += max(0.0, float(xmeas_true_next[6] - lim.reactor_pressure_max)) / lim.reactor_pressure_max
            constraint_cost += max(0.0, float(xmeas_true_next[8] - lim.reactor_temp_max)) / lim.reactor_temp_max
            constraint_cost += max(0.0, float(xmeas_true_next[7] - lim.reactor_level_max)) / max(1.0, lim.reactor_level_max)
            constraint_cost += max(0.0, float(lim.reactor_level_min - xmeas_true_next[7])) / max(1.0, lim.reactor_level_min)

            w = cfg.weights
            reward = -(
                w.tracking * tracking_cost
                + w.utilities * utilities_cost
                + w.mv_move * mv_move_cost
                + w.constraint_violation * constraint_cost
            )
            reward_sum += float(reward)

            # Group tracking costs (for fairness analysis)
            gc = self._tracking_costs_by_group(xmeas_true_next)
            for k in group_costs_accum:
                group_costs_accum[k] += float(gc.get(k, 0.0))

            if cfg.record_trace:
                self._trace["t"].append(self.t)
                self._trace["mode"].append(self.current_mode)
                self._trace["xmeas"].append(xmeas_true_next.copy())
                self._trace["xmv"].append(xmv_next.copy())
                self._trace["residual"].append(res_filt.copy())
                self._trace["reward"].append(float(reward))

            if not ok:
                terminated = True
                break

        obs_measured = self.get_xmeas_measured()

        res_applied_mean = (res_applied_sum / max(1, res_applied_n)).astype(np.float32)

        info = {
            "t": int(self.t),
            "mode": int(self.current_mode),
            "reward_sum": float(reward_sum),
            "violations": int(violations),
            "pre_shield_breaches": int(pre_shield_breaches),
            "shield_activations": int(shield_activations),
            "shield_solve_time_ms": float(shield_solve_time_ms),
            # Residual bookkeeping (useful for identification / debugging)
            "residual_requested": residual.copy(),
            "residual_applied": res_applied_mean.copy(),
            "residual_requested_norm": (residual / float(cfg.residual_max)).copy(),
            "residual_applied_norm": (res_applied_mean / float(cfg.residual_max)).copy(),
            "group_tracking_costs": group_costs_accum,
            "active_disturbances": self.sim.get_active_disturbances(),
            "mv_gain": self.mv_gain.copy(),
            "mv_bias": self.mv_bias.copy(),
        }

        if cfg.record_trace:
            info["trace"] = self._trace

        return obs_measured, float(reward_sum), bool(terminated), False, info


# =============================================================================
# Gym environments
# =============================================================================
class TEPContinuousControlEnv(gym.Env):
    """Continuous residual control for the TEP."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[TEPConfig] = None, initial_mode: Optional[int] = None):
        super().__init__()
        self.config = config or TEPConfig()
        # Backwards compatibility: some scripts pass `initial_mode` directly.
        if initial_mode is not None:
            self.config.initial_mode = int(initial_mode)
        self.plant = TEPPlant(self.config)

        self._steps = 0
        self._max_steps = max(1, self.config.episode_length_sec // self.config.control_interval_sec)

        # Action is residual on 12 MVs
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        # Observation: [xmeas_norm(41), xmv_norm(12), mode_onehot(6)]
        obs_dim = 41 + 12 + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(int(self.config.seed))
        self._demand_mode = int(self.config.initial_mode)

    def _update_demand(self) -> None:
        cfg = self.config
        t = int(self.plant.t)

        # 1) Scheduled demand
        if cfg.demand_schedule:
            # Find last schedule entry with time <= t
            mode = int(cfg.demand_schedule[0][1])
            for ts, m in cfg.demand_schedule:
                if int(ts) <= t:
                    mode = int(m)
                else:
                    break
            self._demand_mode = mode
            return

        # 2) Stochastic demand changes
        if cfg.demand_change_prob > 0 and cfg.demand_interval_sec > 0:
            if t > 0 and (t % int(cfg.demand_interval_sec) == 0):
                if self._rng.random() < float(cfg.demand_change_prob):
                    choices = [m for m in range(1, 7) if m != self._demand_mode]
                    self._demand_mode = int(self._rng.choice(choices))

    def _get_obs(self) -> np.ndarray:
        xmeas = self.plant.get_xmeas_measured()
        xmv = self.plant.get_xmv()

        xmeas_norm = xmeas / _DEF_MAX_XMEAS
        xmv_norm = xmv / _DEF_MAX_XMV
        mode_onehot = _one_hot(self.plant.current_mode - 1, 6)

        return np.concatenate([xmeas_norm, xmv_norm, mode_onehot]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.config.seed = int(seed)
        self._rng = np.random.default_rng(int(self.config.seed))

        # Optionally randomize initial mode (training convenience)
        init_mode = int(self.config.initial_mode)
        if self.config.randomize_initial_mode:
            init_mode = int(self._rng.integers(1, 7))

        self._demand_mode = init_mode
        self.plant.reset(seed=int(self.config.seed), initial_mode=init_mode)

        self._steps = 0
        obs = self._get_obs()
        info = {"t": int(self.plant.t), "mode": int(self.plant.current_mode), "demand_mode": int(self._demand_mode)}
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1

        # Scale normalized action to physical residual units
        a = np.asarray(action, dtype=np.float32).reshape(12)
        a = np.clip(a, -1.0, 1.0)
        residual = a * float(self.config.residual_max)

        # Optional training shortcut: force mode to follow demand
        if self.config.follow_demand_in_training:
            self._update_demand()
            self.plant.set_mode(self._demand_mode)

        _, reward_sum, terminated, _, info = self.plant.run(self.config.control_interval_sec, residual)

        obs = self._get_obs()

        truncated = self._steps >= self._max_steps
        info["demand_mode"] = int(self._demand_mode)
        return obs, float(reward_sum), bool(terminated), bool(truncated), info


class TEPSchedulingEnv(gym.Env):
    """Scheduling + continuous control environment.

    Scheduler selects a mode (1..6) every `scheduling_interval_sec`. Between
    scheduler decisions, the control policy (residual controller) is applied at
    `control_interval_sec`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[TEPConfig] = None,
        control_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        initial_mode: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or TEPConfig()
        # Backwards compatibility: some scripts pass `initial_mode` directly.
        if initial_mode is not None:
            self.config.initial_mode = int(initial_mode)
        self.plant = TEPPlant(self.config)

        self.control_policy = control_policy

        self._rng = np.random.default_rng(int(self.config.seed))
        self._demand_mode = int(self.config.initial_mode)
        self._steps = 0

        self._max_steps = max(1, self.config.episode_length_sec // self.config.scheduling_interval_sec)

        # Scheduler action: choose one of 6 modes
        self.action_space = spaces.Discrete(6)

        # Observation: [xmeas_norm(41), xmv_norm(12), current_mode_onehot(6), demand_onehot(6), readiness(3)]
        obs_dim = 41 + 12 + 6 + 6 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _update_demand(self) -> None:
        cfg = self.config
        t = int(self.plant.t)

        if cfg.demand_schedule:
            mode = int(cfg.demand_schedule[0][1])
            for ts, m in cfg.demand_schedule:
                if int(ts) <= t:
                    mode = int(m)
                else:
                    break
            self._demand_mode = mode
            return

        if cfg.demand_change_prob > 0 and cfg.demand_interval_sec > 0:
            if t > 0 and (t % int(cfg.demand_interval_sec) == 0):
                if self._rng.random() < float(cfg.demand_change_prob):
                    choices = [m for m in range(1, 7) if m != self._demand_mode]
                    self._demand_mode = int(self._rng.choice(choices))

    def _compute_readiness(self, xmeas_true: np.ndarray) -> np.ndarray:
        # Readiness is a soft signal: closer to setpoints => higher readiness.
        mode_cfg = OPERATING_MODES[int(self.plant.current_mode)]
        sp = mode_cfg.xmeas_setpoints

        group_meas = {
            "feed": {0, 1, 2, 3, 4, 5, 9, 39, 40},
            "separator": {11, 12, 14, 15, 7},
            "utility": {6, 8, 10, 17, 18, 19, 20, 21},
        }

        errs = []
        for g in ["feed", "separator", "utility"]:
            e = 0.0
            n = 0
            for idx, sp_val in sp.items():
                i = int(idx)
                if i in group_meas[g]:
                    err = float(xmeas_true[i] - float(sp_val)) / float(_DEF_MAX_XMEAS[i])
                    e += err * err
                    n += 1
            if n > 0:
                e /= n
            errs.append(e)

        # Map error to readiness in (0,1]
        readiness = np.exp(-10.0 * np.asarray(errs, dtype=np.float32))
        return readiness.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        xmeas_meas = self.plant.get_xmeas_measured()
        xmeas_true = self.plant.get_xmeas_true()
        xmv = self.plant.get_xmv()

        xmeas_norm = xmeas_meas / _DEF_MAX_XMEAS
        xmv_norm = xmv / _DEF_MAX_XMV

        mode_onehot = _one_hot(self.plant.current_mode - 1, 6)
        demand_onehot = _one_hot(self._demand_mode - 1, 6)
        readiness = self._compute_readiness(xmeas_true)

        return np.concatenate([xmeas_norm, xmv_norm, mode_onehot, demand_onehot, readiness]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.config.seed = int(seed)
        self._rng = np.random.default_rng(int(self.config.seed))

        init_mode = int(self.config.initial_mode)
        self._demand_mode = init_mode
        self.plant.reset(seed=int(self.config.seed), initial_mode=init_mode)

        self._steps = 0
        obs = self._get_obs()
        info = {"t": int(self.plant.t), "mode": int(self.plant.current_mode), "demand_mode": int(self._demand_mode)}
        return obs, info

    def step(self, action: int):
        self._steps += 1

        # Update exogenous demand
        self._update_demand()

        # Scheduler decision: mode in 1..6
        prev_mode = int(self.plant.current_mode)
        mode = int(action) + 1
        self.plant.set_mode(mode)

        # Scheduling penalties (demand adherence + switching)
        mismatch = 1 if int(mode) != int(self._demand_mode) else 0
        switched = 1 if int(mode) != int(prev_mode) else 0

        # Run control for one scheduling interval
        interval = int(self.config.scheduling_interval_sec)
        n_ctrl = max(1, interval // int(self.config.control_interval_sec))

        reward_total = 0.0
        terminated = False
        violations_total = 0
        pre_breach_total = 0
        shield_act_total = 0
        shield_time_total = 0.0

        group_costs = {"feed": 0.0, "separator": 0.0, "utility": 0.0}

        for _ in range(n_ctrl):
            # Control obs for the residual controller: same as continuous env obs
            obs_control = np.concatenate([
                (self.plant.get_xmeas_measured() / _DEF_MAX_XMEAS),
                (self.plant.get_xmv() / _DEF_MAX_XMV),
                _one_hot(self.plant.current_mode - 1, 6),
            ]).astype(np.float32)

            if self.control_policy is None:
                u = np.zeros(12, dtype=np.float32)
            else:
                u = np.asarray(self.control_policy(obs_control), dtype=np.float32).reshape(12)
                u = np.clip(u, -1.0, 1.0)

            residual = u * float(self.config.residual_max)

            _, r, term, _, info = self.plant.run(self.config.control_interval_sec, residual)
            reward_total += float(r)
            terminated = bool(terminated or term)

            violations_total += int(info.get("violations", 0))
            pre_breach_total += int(info.get("pre_shield_breaches", 0))
            shield_act_total += int(info.get("shield_activations", 0))
            shield_time_total += float(info.get("shield_solve_time_ms", 0.0))

            gc = info.get("group_tracking_costs", {})
            for k in group_costs:
                group_costs[k] += float(gc.get(k, 0.0))

            if terminated:
                break

        # Apply scheduling penalties once per decision.
        w = self.config.weights
        reward_total -= float(w.demand_mismatch) * float(mismatch)
        reward_total -= float(w.mode_switch) * float(switched)

        obs = self._get_obs()

        truncated = self._steps >= self._max_steps

        info_out = {
            "t": int(self.plant.t),
            "mode": int(self.plant.current_mode),
            "demand_mode": int(self._demand_mode),
            "demand_mismatch": int(mismatch),
            "demand_mismatch_seconds": int(mismatch) * int(interval),
            "mode_switch": int(switched),
            "violations": int(violations_total),
            "pre_shield_breaches": int(pre_breach_total),
            "shield_activations": int(shield_act_total),
            "shield_solve_time_ms": float(shield_time_total),
            "group_tracking_costs": group_costs,
            "active_disturbances": self.plant.sim.get_active_disturbances(),
        }

        return obs, float(reward_total), bool(terminated), bool(truncated), info_out


# =============================================================================
# Helper: create scenario-specific configs
# =============================================================================
def apply_scenario_to_config(base: TEPConfig, scenario: ScenarioDefinition) -> TEPConfig:
    """Return a copy of base config with scenario fields applied."""

    cfg = TEPConfig(**{k: getattr(base, k) for k in base.__dataclass_fields__.keys()})

    cfg.scenario = scenario.scenario
    cfg.initial_mode = int(scenario.initial_mode)
    cfg.demand_schedule = list(scenario.demand_schedule)
    cfg.disturbances = list(scenario.disturbances)
    cfg.sensor_bias = dict(scenario.sensor_bias)
    cfg.mv_gain_sigma = float(scenario.mv_gain_sigma)
    cfg.mv_bias_sigma = float(scenario.mv_bias_sigma)

    return cfg
