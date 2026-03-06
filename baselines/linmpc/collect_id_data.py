"""Collect identification data for the linear MPC baseline.

We collect (s_t, u_t, s_{t+1}) tuples at the **control interval** (default 6 s)
using the real TEP simulator:
- State s := [xmeas_norm(41), xmv_norm(12)]  (53 dims)
- Input u := residual action in normalized units [-1, 1] (12 dims)

The simulator is operated under the decentralized PI controller; we inject
bounded random residuals to excite the dynamics.

Outputs
-------
One NPZ file per operating mode:
  <out_dir>/data/id_mode<k>_seed<seed>.npz

Each NPZ contains:
  X : (N, 53)  state at time t
  U : (N, 12)  residual action applied at time t
  Y : (N, 53)  next state at time t+1
  meta : dict  (saved as npz object)

Usage
-----
python -m baselines.linmpc.collect_id_data --config configs/paper.yaml --out_dir artifacts/linmpc --seed 0

"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from envs import TEPConfig, TEPContinuousControlEnv


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _build_base_cfg(cfg_dict: dict) -> TEPConfig:
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

    return base_cfg


def collect_mode_data(
    *,
    base_cfg: TEPConfig,
    mode: int,
    out_dir: Path,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    burn_in_steps: int,
    action_std: float,
    action_clip: float,
    use_safety_shield: bool,
) -> Path:
    """Collect identification tuples for a single operating mode."""

    cfg = TEPConfig(**{k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__.keys()})

    cfg.seed = int(seed)
    cfg.initial_mode = int(mode)

    # Keep the plant in a fixed mode for identification.
    cfg.follow_demand_in_training = True
    cfg.demand_schedule = [(0, int(mode))]
    cfg.randomize_initial_mode = False

    # Nominal identification (no explicit scenario stressors).
    cfg.disturbances = []
    cfg.sensor_bias = {}
    cfg.mv_gain_sigma = 0.0
    cfg.mv_bias_sigma = 0.0

    # Optional shield during data collection.
    cfg.use_safety_shield = bool(use_safety_shield)

    env = TEPContinuousControlEnv(cfg)

    rng = np.random.default_rng(int(seed))

    X_list: List[np.ndarray] = []
    U_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed + 10_000 * ep))

        # Burn-in: let PI settle before collecting tuples.
        for _ in range(int(burn_in_steps)):
            u0 = np.zeros(12, dtype=np.float32)
            obs, _, term, trunc, _ = env.step(u0)
            if term or trunc:
                break

        for _ in range(int(steps_per_episode)):
            x = obs[: 41 + 12].copy()  # (53,)

            u = rng.normal(0.0, float(action_std), size=12).astype(np.float32)
            u = np.clip(u, -float(action_clip), float(action_clip)).astype(np.float32)

            obs2, _, term, trunc, info = env.step(u)
            y = obs2[: 41 + 12].copy()

            # If a safety shield is active, the applied residual can differ from
            # the requested one. Prefer the applied value for identification.
            u_used = u
            if isinstance(info, dict) and ("residual_applied_norm" in info):
                try:
                    u_used = np.asarray(info["residual_applied_norm"], dtype=np.float32).reshape(12)
                except Exception:
                    u_used = u

            X_list.append(x)
            U_list.append(u_used)
            Y_list.append(y)

            obs = obs2
            if term or trunc:
                break

    X = np.asarray(X_list, dtype=np.float32)
    U = np.asarray(U_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    meta: Dict[str, object] = {
        "mode": int(mode),
        "seed": int(seed),
        "episodes": int(episodes),
        "steps_per_episode": int(steps_per_episode),
        "burn_in_steps": int(burn_in_steps),
        "action_std": float(action_std),
        "action_clip": float(action_clip),
        "use_safety_shield": bool(use_safety_shield),
        "control_interval_sec": int(cfg.control_interval_sec),
        "state_dim": int(X.shape[1]) if X.ndim == 2 else 53,
        "input_dim": int(U.shape[1]) if U.ndim == 2 else 12,
        "base_cfg": asdict(cfg),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / f"id_mode{int(mode)}_seed{int(seed)}.npz"
    np.savez_compressed(out_path, X=X, U=U, Y=Y, meta=meta)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect identification data for linear MPC baseline")
    ap.add_argument("--config", type=str, default="configs/paper.yaml", help="YAML config used for env settings")
    ap.add_argument("--out_dir", type=str, default="artifacts/linmpc", help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Operating modes to collect")
    ap.add_argument("--episodes", type=int, default=1, help="Episodes per mode")
    ap.add_argument("--steps_per_episode", type=int, default=2000, help="Control steps per episode")
    ap.add_argument("--burn_in_steps", type=int, default=100, help="Burn-in control steps before logging")
    ap.add_argument("--action_std", type=float, default=0.20, help="Std dev of random residual (normalized)")
    ap.add_argument("--action_clip", type=float, default=0.50, help="Clip bound for random residual (normalized)")
    ap.add_argument("--use_safety_shield", action="store_true", help="Enable safety shield during collection")

    args = ap.parse_args()

    cfg_dict = _load_yaml(Path(args.config))
    base_cfg = _build_base_cfg(cfg_dict)

    out_dir = Path(args.out_dir)

    for mode in args.modes:
        path = collect_mode_data(
            base_cfg=base_cfg,
            mode=int(mode),
            out_dir=out_dir,
            seed=int(args.seed),
            episodes=int(args.episodes),
            steps_per_episode=int(args.steps_per_episode),
            burn_in_steps=int(args.burn_in_steps),
            action_std=float(args.action_std),
            action_clip=float(args.action_clip),
            use_safety_shield=bool(args.use_safety_shield),
        )
        print(f"[linmpc] wrote: {path}")


if __name__ == "__main__":
    main()
