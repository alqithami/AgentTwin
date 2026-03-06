"""Training pipeline for AgentTwin (real TEP reproduction).

This module trains:
1) Centralized SAC residual controller (12-D action)
2) Multi-agent SAC residual controllers (3 groups)
3) PPO scheduler over TE operating modes (6 discrete actions)

All training uses the *real* Tennessee Eastman Process simulator vendored under
`third_party/tep`.

Entry points
------------
- As a module: `python -m train.training_pipeline --config configs/paper.yaml --seed 0`
- From the end-to-end orchestrator: `python -m pipeline.reproduce --config configs/paper.yaml`

Artifacts
---------
Saved under:
  artifacts/seed_<seed>/models/
  artifacts/seed_<seed>/logs/

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from envs import (
    MV_GROUPS,
    TEPConfig,
    TEPContinuousControlEnv,
    TEPSchedulingEnv,
)


from envs.shield import QPShieldConfig
# Stable-Baselines3
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for training. Install dependencies via: pip install -r requirements.txt"
    ) from e


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: Path, obj) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


# -----------------------------------------------------------------------------
# Multi-agent wrapper
# -----------------------------------------------------------------------------

class PartialControlEnv(TEPContinuousControlEnv):
    """A single-agent view of the 12-D residual control env.

    The agent controls only the MVs in `group_name`. Other groups are provided
    by fixed models (if given) or zeros.
    """

    def __init__(
        self,
        base_cfg: TEPConfig,
        group_name: str,
        other_models: Optional[Dict[str, SAC]] = None,
        deterministic_others: bool = True,
    ):
        cfg = TEPConfig(**{k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__.keys()})
        super().__init__(config=cfg)

        if group_name not in MV_GROUPS:
            raise KeyError(f"Unknown group_name={group_name}. Expected one of: {list(MV_GROUPS)}")
        self.group_name = group_name
        self.group_idx = list(MV_GROUPS[group_name])
        self.other_models = other_models or {}
        self.deterministic_others = bool(deterministic_others)

        # Override action_space to group dimension
        from gymnasium import spaces

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.group_idx),), dtype=np.float32)

    def step(self, action):
        # Build full 12-D normalized action
        full_action = np.zeros(12, dtype=np.float32)
        a = np.asarray(action, dtype=np.float32).reshape(len(self.group_idx))
        full_action[self.group_idx] = np.clip(a, -1.0, 1.0)

        # Query other models
        if self.other_models:
            obs = self._get_obs()
            for other_name, other_model in self.other_models.items():
                if other_name == self.group_name:
                    continue
                idx = MV_GROUPS[other_name]
                try:
                    other_act, _ = other_model.predict(obs, deterministic=self.deterministic_others)
                    other_act = np.asarray(other_act, dtype=np.float32).reshape(len(idx))
                    full_action[idx] = np.clip(other_act, -1.0, 1.0)
                except Exception:
                    # If model not ready, keep zeros.
                    pass

        obs, reward, terminated, truncated, info = super().step(full_action)
        return obs, reward, terminated, truncated, info


def build_group_policy(group_models: Dict[str, SAC], deterministic: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """Return a policy(obs)->action(12,) that concatenates the group actions."""

    def _pi(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(12, dtype=np.float32)
        for group_name, model in group_models.items():
            idx = MV_GROUPS[group_name]
            act, _ = model.predict(obs, deterministic=deterministic)
            act = np.asarray(act, dtype=np.float32).reshape(len(idx))
            a[idx] = np.clip(act, -1.0, 1.0)
        return a

    return _pi


# -----------------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------------

def train_centralized_sac(cfg: TEPConfig, out_dir: Path, total_timesteps: int) -> SAC:
    env = Monitor(TEPContinuousControlEnv(cfg))

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=int(cfg.seed),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        gamma=0.99,
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)

    model_path = out_dir / "centralized_sac.zip"
    model.save(str(model_path))
    return model


def train_multiagent_sac(cfg: TEPConfig, out_dir: Path, total_timesteps: int, rounds: int) -> Dict[str, SAC]:
    groups = list(MV_GROUPS.keys())

    steps_per_group = int(total_timesteps)
    steps_per_round = max(1, steps_per_group // max(1, rounds))

    # Initialize models
    group_models: Dict[str, SAC] = {}
    for g in groups:
        env_g = Monitor(PartialControlEnv(cfg, g, other_models=None))
        group_models[g] = SAC(
            policy="MlpPolicy",
            env=env_g,
            verbose=0,
            seed=int(cfg.seed),
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            gamma=0.99,
        )

    # Iterated best-response style training
    for r in range(int(rounds)):
        for g in groups:
            others = {k: v for k, v in group_models.items() if k != g}
            env_g = Monitor(PartialControlEnv(cfg, g, other_models=others))
            group_models[g].set_env(env_g)
            group_models[g].learn(total_timesteps=int(steps_per_round), reset_num_timesteps=False, progress_bar=True)

    # Save
    for g, model in group_models.items():
        model.save(str(out_dir / f"multiagent_sac_{g}.zip"))

    return group_models


def train_scheduler_ppo(cfg: TEPConfig, out_dir: Path, total_timesteps: int, control_policy: Callable[[np.ndarray], np.ndarray]) -> PPO:
    env = Monitor(TEPSchedulingEnv(cfg, control_policy=control_policy))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=int(cfg.seed),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)
    model.save(str(out_dir / "scheduler_ppo.zip"))
    return model


def train_all_for_seed(
    base_cfg: TEPConfig,
    seed: int,
    controller_train: dict,
    scheduler_train: dict,
    out_root: Path,
) -> Dict[str, Path]:
    """Train all requested components for a given seed.

    Returns a mapping of component name -> saved model path.
    """

    cfg = TEPConfig(**{k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__.keys()})
    cfg.seed = int(seed)

    # For controller training we want exposure to mode changes.
    cfg.randomize_initial_mode = True
    cfg.follow_demand_in_training = True
    if cfg.demand_change_prob <= 0.0 and not cfg.demand_schedule:
        cfg.demand_change_prob = 0.35
        cfg.demand_interval_sec = 1800

    set_random_seed(int(seed))

    seed_dir = _ensure_dir(out_root / f"seed_{seed}")
    models_dir = _ensure_dir(seed_dir / "models")
    logs_dir = _ensure_dir(seed_dir / "logs")

    _save_json(seed_dir / "config_used.json", asdict(cfg))

    paths: Dict[str, Path] = {}

    centralized_model: Optional[SAC] = None
    if bool(controller_train.get("train_centralized", True)):
        centralized_model = train_centralized_sac(cfg, models_dir, int(controller_train.get("sac_steps", 200000)))
        paths["centralized_sac"] = models_dir / "centralized_sac.zip"

    group_models: Optional[Dict[str, SAC]] = None
    if bool(controller_train.get("train_multi_agent", True)):
        group_models = train_multiagent_sac(
            cfg,
            models_dir,
            int(controller_train.get("sac_steps", 200000)),
            int(controller_train.get("rounds", 3)),
        )
        for g in MV_GROUPS.keys():
            paths[f"multiagent_sac_{g}"] = models_dir / f"multiagent_sac_{g}.zip"

    # Scheduler (requires a control policy). Prefer multi-agent controllers if available.
    if bool(scheduler_train.get("enabled", True)):
        if group_models is not None:
            control_policy = build_group_policy(group_models, deterministic=True)
        elif centralized_model is not None:
            control_policy = lambda obs: np.asarray(centralized_model.predict(obs, deterministic=True)[0], dtype=np.float32)
        else:
            control_policy = lambda obs: np.zeros(12, dtype=np.float32)

        _ = train_scheduler_ppo(cfg, models_dir, int(scheduler_train.get("ppo_steps", 100000)), control_policy)
        paths["scheduler_ppo"] = models_dir / "scheduler_ppo.zip"

    return paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())



def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgentTwin models (real TEP)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (configs/*.yaml)")
    parser.add_argument(
        "--out_root",
        type=str,
        default="artifacts",
        help="Directory where seed_* artifacts will be written",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list. If omitted, uses experiment.seeds from the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fallback seed if neither --seeds nor experiment.seeds are provided.",
    )
    args = parser.parse_args()

    # NOTE: earlier internal versions used a `load_yaml()` helper name.
    # Keep this robust by using the local implementation.
    cfg_dict = _load_yaml(Path(args.config))

    exp_dict = cfg_dict.get("experiment", {})
    env_dict = cfg_dict.get("env", {})
    train_dict = cfg_dict.get("training", {})
    ctrl_train = train_dict.get("controller", {})
    sched_train = train_dict.get("scheduler", {})

    seeds: List[int]
    if args.seeds is not None and len(args.seeds) > 0:
        seeds = list(args.seeds)
    else:
        seeds = list(exp_dict.get("seeds", [args.seed]))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        base_cfg = TEPConfig(
            seed=int(seed),
            tep_backend=str(env_dict.get("tep_backend", "python")),
            control_interval_sec=int(env_dict.get("control_interval_sec", 6)),
            scheduling_interval_sec=int(env_dict.get("scheduling_interval_sec", 300)),
            episode_length_sec=int(env_dict.get("episode_length_sec", 8 * 3600)),
            residual_max=float(env_dict.get("residual_max", 10.0)),
            pi_update_interval_sec=int(env_dict.get("pi_update_interval_sec", 1)),
            warmup_sec=int(env_dict.get("warmup_sec", 0)),
            use_safety_shield=bool(env_dict.get("use_safety_shield", True)),
            safety_shield_type=str(env_dict.get("safety_shield_type", "qp")),
            qp_shield=QPShieldConfig(
                alpha=float(env_dict.get("qp_shield", {}).get("alpha", 0.20)),
                lambda_reg=float(env_dict.get("qp_shield", {}).get("lambda_reg", 0.001)),
            ),
        )

        # Optional reward shaping weights (incl. scheduling penalties)
        rw = env_dict.get("reward", {})
        if isinstance(rw, dict):
            for k, v in rw.items():
                if hasattr(base_cfg.weights, str(k)):
                    try:
                        setattr(base_cfg.weights, str(k), float(v))
                    except Exception:
                        pass

        train_all_for_seed(base_cfg=base_cfg, seed=int(seed), controller_train=ctrl_train, scheduler_train=sched_train, out_root=out_root)


if __name__ == "__main__":
    main()
