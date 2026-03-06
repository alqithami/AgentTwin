"""Evaluate trained AgentTwin models and generate tables/figures.

Outputs
-------
- results/raw/*.csv: per-episode metrics (one row per episode)
- results/tables/*.csv: aggregated tables (mean ± std)
- results/tables/*.tex: LaTeX-ready versions of the tables
- results/plots/*.png, *.pdf: B/W figures with a single accent color

Typical usage
-------------
1) Train:
    python -m train.training_pipeline --config configs/paper.yaml --seed 0
2) Evaluate:
    python -m eval.generate_results --config configs/paper.yaml --artifacts_root artifacts --results_dir results

Note
----
This evaluation uses the **vendored real TE simulator** under `third_party/tep`.
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import (
    MV_GROUPS,
    TEPConfig,
    TEPSchedulingEnv,
    ScenarioDefinition,
    ScenarioType,
    apply_scenario_to_config,
    get_scenario_set,
)

# Stable-Baselines3
try:
    from stable_baselines3 import PPO, SAC
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for evaluation. Install via: pip install -r requirements.txt"
    ) from e


ACCENT_COLOR = "tab:red"  # single non-grayscale highlight


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _scenario_fingerprint(scenario: ScenarioDefinition) -> str:
    """Stable fingerprint to help verify that scenarios actually differ."""

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

def _save_figure(plots_dir: Path, basename: str) -> None:
    """Save the current Matplotlib figure as PNG and PDF."""
    plt.savefig(plots_dir / f"{basename}.png", dpi=300)
    plt.savefig(plots_dir / f"{basename}.pdf")





def _method_style(name: str) -> Dict[str, object]:
    """Return style primitives for B/W plots with one accent color.

    Keys returned:
      - facecolor, edgecolor, hatch: for bar charts
      - linecolor, linestyle, marker, linewidth: for line charts
    """
    key = str(name).lower().strip()

    # Highlight the proposed method with the single accent color
    if key.startswith("agenttwin"):
        return {
            "facecolor": ACCENT_COLOR,
            "edgecolor": "black",
            "hatch": None,
            "linecolor": ACCENT_COLOR,
            "linestyle": "-",
            "marker": "D",
            "linewidth": 2.6,
        }

    if key.startswith("pid"):
        return {
            "facecolor": "white",
            "edgecolor": "black",
            "hatch": "",
            "linecolor": "black",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.8,
        }

    if "centralized" in key:
        return {
            "facecolor": "white",
            "edgecolor": "black",
            "hatch": "//",
            "linecolor": "black",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 1.6,
        }

    if "multi-agent" in key or "multiagent" in key:
        return {
            "facecolor": "white",
            "edgecolor": "black",
            "hatch": "xx",
            "linecolor": "black",
            "linestyle": ":",
            "marker": "^",
            "linewidth": 1.6,
        }

    if "no shield" in key:
        return {
            "facecolor": "white",
            "edgecolor": "black",
            "hatch": "\\",
            "linecolor": "black",
            "linestyle": "-.",
            "marker": "v",
            "linewidth": 1.6,
        }

    if "linmpc" in key:
        return {
            "facecolor": "white",
            "edgecolor": "black",
            "hatch": "..",
            "linecolor": "black",
            "linestyle": "-",
            "marker": "x",
            "linewidth": 1.6,
        }

    # Default fallback
    return {
        "facecolor": "white",
        "edgecolor": "black",
        "hatch": "..",
        "linecolor": "black",
        "linestyle": "-",
        "marker": ".",
        "linewidth": 1.2,
    }


def _greedy_demand_scheduler(obs: np.ndarray) -> int:
    # Obs layout: [xmeas(41), xmv(12), mode_onehot(6), demand_onehot(6), readiness(3)]
    demand = obs[41 + 12 + 6 : 41 + 12 + 6 + 6]
    return int(np.argmax(demand))


def _load_models_for_seed(seed_dir: Path) -> Dict[str, object]:
    models_dir = seed_dir / "models"

    out: Dict[str, object] = {}

    c_path = models_dir / "centralized_sac.zip"
    if c_path.exists():
        out["centralized_sac"] = SAC.load(str(c_path))

    group_models: Dict[str, SAC] = {}
    for g in MV_GROUPS.keys():
        p = models_dir / f"multiagent_sac_{g}.zip"
        if p.exists():
            group_models[g] = SAC.load(str(p))
    if group_models:
        out["multiagent"] = group_models

    s_path = models_dir / "scheduler_ppo.zip"
    if s_path.exists():
        out["scheduler_ppo"] = PPO.load(str(s_path))

    return out


def _build_group_control_policy(group_models: Dict[str, SAC], deterministic: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    def _pi(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(12, dtype=np.float32)
        for g, model in group_models.items():
            idx = MV_GROUPS[g]
            act, _ = model.predict(obs, deterministic=deterministic)
            act = np.asarray(act, dtype=np.float32).reshape(len(idx))
            a[idx] = np.clip(act, -1.0, 1.0)
        return a

    return _pi


# -----------------------------------------------------------------------------
# Evaluation core
# -----------------------------------------------------------------------------

def evaluate_method_on_scenario(
    method_name: str,
    base_cfg: TEPConfig,
    scenario: ScenarioDefinition,
    scheduler_policy: Callable[[np.ndarray], int],
    control_policy: Optional[Callable[[np.ndarray], np.ndarray]],
    n_episodes: int,
    seed_offset: int,
    deterministic: bool = True,
    train_seed: int = 0,
) -> pd.DataFrame:
    cfg = apply_scenario_to_config(base_cfg, scenario)

    scen_fp = _scenario_fingerprint(scenario)

    env = TEPSchedulingEnv(cfg, control_policy=control_policy)

    rows: List[dict] = []
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
        group_cost_sum = {k: 0.0 for k in MV_GROUPS.keys()}

        t_end = 0
        done_reason = "truncated"  # overwritten if terminated

        while not done:
            a_sched = scheduler_policy(obs)
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

        # Jain fairness on *costs* (positive): 1.0 is perfectly balanced
        costs = np.asarray([group_cost_sum[g] for g in MV_GROUPS.keys()], dtype=float)
        fairness = float(np.nan)
        if np.all(np.isfinite(costs)) and np.sum(costs) > 0:
            fairness = float((np.sum(costs) ** 2) / (len(costs) * np.sum(costs**2) + 1e-12))

        rows.append(
            {
                "method": method_name,
                "scenario": scenario.scenario.value,
                "episode": ep,
                "seed": int(train_seed),
                "eval_seed": int(ep_seed),
                "scenario_fp": str(scen_fp),
                "total_reward": tot_reward,
                "violations": violations,
                "pre_shield_breaches": pre_breaches,
                "shield_activations": shield_acts,
                "shield_solve_time_ms": solve_ms,
                "demand_mismatch_seconds": int(mismatch_seconds),
                "mode_switches": int(mode_switches),
                "t_end": int(t_end),
                "done_reason": str(done_reason),
                "fairness_jain": fairness,
                **{f"cost_{g}": float(group_cost_sum[g]) for g in MV_GROUPS.keys()},
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Tables/figures
# -----------------------------------------------------------------------------

def _mean_std(x: pd.Series) -> str:
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def make_tables(all_df: pd.DataFrame, tables_dir: Path) -> None:
    _ensure_dir(tables_dir)

    # Focus scenario S2 as the primary comparison table
    s2 = all_df[all_df["scenario"] == ScenarioType.S2_DYNAMIC_DEMAND.value]
    if s2.empty:
        s2 = all_df

    perf = (
        s2.groupby("method")
        .agg(
            total_reward_mean=("total_reward", "mean"),
            total_reward_std=("total_reward", "std"),
            violations_mean=("violations", "mean"),
            violations_std=("violations", "std"),
        )
        .reset_index()
    )
    perf.to_csv(tables_dir / "table1_performance_S2.csv", index=False)
    perf.to_latex(tables_dir / "table1_performance_S2.tex", index=False, float_format="%.2f")

    safety = (
        s2.groupby("method")
        .agg(
            pre_shield_breaches_mean=("pre_shield_breaches", "mean"),
            pre_shield_breaches_std=("pre_shield_breaches", "std"),
            shield_activations_mean=("shield_activations", "mean"),
            shield_activations_std=("shield_activations", "std"),
            shield_solve_time_ms_mean=("shield_solve_time_ms", "mean"),
            shield_solve_time_ms_std=("shield_solve_time_ms", "std"),
        )
        .reset_index()
    )

    # Conditional QP solve time (per activation)
    if "shield_activations" in s2.columns and "shield_solve_time_ms" in s2.columns:
        tmp = s2.copy()
        tmp["solve_ms_when_active"] = tmp["shield_solve_time_ms"] / tmp["shield_activations"].clip(lower=1)
        cond = (
            tmp.groupby("method")["solve_ms_when_active"]
            .agg([("solve_ms_when_active_mean", "mean"), ("solve_ms_when_active_std", "std")])
            .reset_index()
        )
        safety = safety.merge(cond, on="method", how="left")
    safety.to_csv(tables_dir / "table2_safety_S2.csv", index=False)
    safety.to_latex(tables_dir / "table2_safety_S2.tex", index=False, float_format="%.2f")

    scen = (
        all_df.groupby(["scenario", "method"])
        .agg(total_reward_mean=("total_reward", "mean"), total_reward_std=("total_reward", "std"))
        .reset_index()
    )
    scen.to_csv(tables_dir / "table3_scenarios.csv", index=False)
    scen.to_latex(tables_dir / "table3_scenarios.tex", index=False, float_format="%.2f")

    fairness = (
        s2.groupby("method")
        .agg(fairness_jain_mean=("fairness_jain", "mean"), fairness_jain_std=("fairness_jain", "std"))
        .reset_index()
    )
    fairness.to_csv(tables_dir / "table4_fairness_S2.csv", index=False)
    fairness.to_latex(tables_dir / "table4_fairness_S2.tex", index=False, float_format="%.3f")

    # Scheduling adherence (only meaningful when demand is nontrivial)
    if "demand_mismatch_seconds" in s2.columns:
        sched = (
            s2.groupby("method")
            .agg(
                demand_mismatch_seconds_mean=("demand_mismatch_seconds", "mean"),
                demand_mismatch_seconds_std=("demand_mismatch_seconds", "std"),
                mode_switches_mean=("mode_switches", "mean"),
                mode_switches_std=("mode_switches", "std"),
            )
            .reset_index()
        )
        sched.to_csv(tables_dir / "table5_scheduling_S2.csv", index=False)
        sched.to_latex(tables_dir / "table5_scheduling_S2.tex", index=False, float_format="%.2f")


def make_plots(all_df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate B/W plots (PNG + PDF) from the episode-level dataframe.

    The plotting code is deliberately conservative (journal-friendly):
    - grayscale + one accent color (AgentTwin)
    - error bars are standard deviation across episodes
    - large performance gaps (PID vs RL) are handled via split panels
    """

    _ensure_dir(plots_dir)

    # Consistent scenario ordering (if present)
    scenario_order = [
        ScenarioType.S1_NOMINAL.value,
        ScenarioType.S2_DYNAMIC_DEMAND.value,
        ScenarioType.S3_FAULT_DISTURBANCE.value,
        ScenarioType.S4_SENSOR_BIAS.value,
        ScenarioType.S5_MODEL_MISMATCH.value,
    ]

    def _short_s(s: str) -> str:
        if isinstance(s, str) and s.startswith("S") and "_" in s:
            return s.split("_", 1)[0]
        return str(s)

    def _method_sort_key(m: str) -> int:
        k = str(m).lower()
        if k.startswith("pid"):
            return 0
        if "linmpc" in k:
            return 1
        if "centralized" in k:
            return 2
        if "multi-agent" in k or "multiagent" in k:
            return 3
        if k.startswith("agenttwin"):
            return 4
        if "no shield" in k:
            return 5
        return 99

    def _split_methods(methods: List[str]) -> Tuple[List[str], List[str]]:
        learning: List[str] = []
        classical: List[str] = []
        for m in methods:
            k = str(m).lower()
            if k.startswith("pid") or ("linmpc" in k):
                classical.append(m)
            else:
                learning.append(m)
        learning = sorted(learning, key=_method_sort_key)
        classical = sorted(classical, key=_method_sort_key)
        return learning, classical

    # Focus S2 as primary comparison scenario
    s2 = all_df[all_df["scenario"] == ScenarioType.S2_DYNAMIC_DEMAND.value]
    if s2.empty:
        s2 = all_df

    # ------------------------------------------------------------------
    # Figure 1: Reward on S2 (split panels to keep RL readable)
    # ------------------------------------------------------------------
    stats_r = s2.groupby("method")["total_reward"].agg(["mean", "std", "count"]).copy()
    stats_r = stats_r.sort_index()
    methods_order = sorted(list(stats_r.index), key=_method_sort_key)
    learning_methods, classical_methods = _split_methods(methods_order)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1.0, 1.0]})

    def _bar(ax, methods: List[str], title: str, y_label: str, zoom: bool = False) -> None:
        for i, m in enumerate(methods):
            row = stats_r.loc[m]
            st = _method_style(m)
            y = float(row["mean"])
            ystd = float(row["std"]) if float(row["count"]) > 1 else 0.0
            ax.bar(
                i,
                y,
                yerr=ystd,
                capsize=3,
                color=st["facecolor"],
                edgecolor=st["edgecolor"],
                hatch=st["hatch"],
                linewidth=1.0,
            )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.85")

        if zoom and len(methods) > 0:
            vals = []
            for m in methods:
                row = stats_r.loc[m]
                y = float(row["mean"])
                ystd = float(row["std"]) if float(row["count"]) > 1 else 0.0
                vals.extend([y - ystd, y + ystd])
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            rng = max(1e-6, vmax - vmin)
            ax.set_ylim(vmin - 0.10 * rng, vmax + 0.10 * rng)

    _bar(axes[0], learning_methods, "(a) Learning-based controllers (zoom)", r"Return $R$", zoom=True)
    _bar(axes[1], classical_methods, "(b) Classical baselines", r"Return $R$", zoom=False)

    plt.tight_layout()
    _save_figure(plots_dir, "figure1_reward_S2")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Safety + compute breakdown on S2
    #   (a) violations  (symlog)
    #   (b) shield activations (symlog)
    #   (c) QP solve time when active (ms)
    # ------------------------------------------------------------------
    stats_v = s2.groupby("method")["violations"].agg(["mean", "std", "count"]).copy()
    stats_a = s2.groupby("method")["shield_activations"].agg(["mean", "std", "count"]).copy()
    tmp = s2.copy()
    tmp["solve_ms_when_active"] = tmp["shield_solve_time_ms"] / tmp["shield_activations"].clip(lower=1)
    stats_t = tmp.groupby("method")["solve_ms_when_active"].agg(["mean", "std", "count"]).copy()

    methods = sorted(list(set(stats_v.index) | set(stats_a.index) | set(stats_t.index)), key=_method_sort_key)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    def _bar_nonneg(ax, stats_df: pd.DataFrame, methods: List[str], title: str, ylabel: str, symlog: bool = False) -> None:
        for i, m in enumerate(methods):
            if m not in stats_df.index:
                continue
            row = stats_df.loc[m]
            st = _method_style(m)
            y = float(row["mean"])
            ystd = float(row["std"]) if float(row["count"]) > 1 else 0.0
            yerr_low = min(ystd, max(0.0, y))
            yerr = np.array([[yerr_low], [ystd]])
            ax.bar(
                i,
                y,
                yerr=yerr,
                capsize=2,
                color=st["facecolor"],
                edgecolor=st["edgecolor"],
                hatch=st["hatch"],
                linewidth=1.0,
            )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.85")
        if symlog:
            ax.set_yscale("symlog", linthresh=1.0)

    _bar_nonneg(axes[0], stats_v, methods, "(a) Violations", r"$N_{\mathrm{viol}}$ / episode", symlog=True)
    _bar_nonneg(axes[1], stats_a, methods, "(b) Shield activations", r"$N_{\mathrm{shield}}$ / episode", symlog=True)
    _bar_nonneg(axes[2], stats_t, methods, "(c) QP solve time", "Solve time when active (ms)", symlog=False)

    plt.tight_layout()
    _save_figure(plots_dir, "figure2_violations_S2")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Scenario robustness (split panels)
    # ------------------------------------------------------------------
    scen_stats = (
        all_df.groupby(["scenario", "method"])["total_reward"]
        .agg([("mean", "mean"), ("std", "std"), ("count", "count")])
        .reset_index()
    )
    scen_mean = scen_stats.pivot(index="scenario", columns="method", values="mean")
    scen_std = scen_stats.pivot(index="scenario", columns="method", values="std")

    keep = [s for s in scenario_order if s in scen_mean.index]
    if keep:
        scen_mean = scen_mean.reindex(keep)
        scen_std = scen_std.reindex(keep)

    methods = sorted(list(scen_mean.columns), key=_method_sort_key)
    learning_methods, classical_methods = _split_methods(methods)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]})
    x = np.arange(len(scen_mean.index))

    def _line(ax, methods: List[str], title: str, zoom: bool = False) -> None:
        for m in methods:
            st = _method_style(m)
            y = scen_mean[m].values
            yerr = scen_std[m].values
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                label=m,
                color=st["linecolor"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=float(st["linewidth"]),
                markersize=6,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_ylabel(r"Return $R$")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.85")
        if zoom and len(methods) > 0:
            vals = []
            for m in methods:
                y = scen_mean[m].values
                yerr = scen_std[m].values
                vals.extend(list(y - yerr))
                vals.extend(list(y + yerr))
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            rng = max(1e-6, vmax - vmin)
            ax.set_ylim(vmin - 0.10 * rng, vmax + 0.10 * rng)

    _line(axes[0], learning_methods, "(a) Learning-based controllers (zoom)", zoom=True)
    _line(axes[1], classical_methods, "(b) Classical baselines", zoom=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_short_s(s) for s in scen_mean.index])
    axes[1].set_xlabel("Scenario")

    # Legend: place in the top panel to save space
    axes[0].legend(ncol=3, fontsize=9, loc="lower left")

    plt.tight_layout()
    _save_figure(plots_dir, "figure3_scenario_robustness")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Fairness (Jain index) on S2
    # ------------------------------------------------------------------
    if "fairness_jain" in s2.columns:
        stats_f = s2.groupby("method")["fairness_jain"].agg(["mean", "std", "count"]).copy()
        methods = sorted(list(stats_f.index), key=_method_sort_key)

        plt.figure(figsize=(11, 4))
        for i, m in enumerate(methods):
            row = stats_f.loc[m]
            st = _method_style(m)
            y = float(row["mean"])
            ystd = float(row["std"]) if float(row["count"]) > 1 else 0.0
            plt.bar(
                i,
                y,
                yerr=ystd,
                capsize=3,
                color=st["facecolor"],
                edgecolor=st["edgecolor"],
                hatch=st["hatch"],
                linewidth=1.0,
            )
        plt.xticks(range(len(methods)), methods, rotation=20, ha="right")
        plt.ylabel("Jain fairness index (± std)")
        plt.ylim(0.0, 1.05)
        plt.title("Scenario S2: Multi-agent cost fairness (higher is better)")
        plt.grid(axis="y", linestyle=":", linewidth=0.6, color="0.85")
        plt.tight_layout()
        _save_figure(plots_dir, "figure4_fairness_S2")
        plt.close()

    # ------------------------------------------------------------------
    # Figure 5: Scheduling adherence on S2 (optional)
    # ------------------------------------------------------------------
    if "demand_mismatch_seconds" in s2.columns:
        stats_m = s2.groupby("method")["demand_mismatch_seconds"].agg(["mean", "std", "count"]).copy()
        stats_s = s2.groupby("method")["mode_switches"].agg(["mean", "std", "count"]).copy()
        methods = sorted(list(stats_m.index), key=_method_sort_key)

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.0))

        def _bar_simple(ax, stats_df: pd.DataFrame, title: str, ylabel: str) -> None:
            for i, m in enumerate(methods):
                row = stats_df.loc[m]
                st = _method_style(m)
                y = float(row["mean"])
                ystd = float(row["std"]) if float(row["count"]) > 1 else 0.0
                yerr_low = min(ystd, max(0.0, y))
                yerr = np.array([[yerr_low], [ystd]])
                ax.bar(
                    i,
                    y,
                    yerr=yerr,
                    capsize=2,
                    color=st["facecolor"],
                    edgecolor=st["edgecolor"],
                    hatch=st["hatch"],
                    linewidth=1.0,
                )
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.85")

        _bar_simple(axes[0], stats_m, "(a) Demand mismatch", "Mismatch time (s) per episode")
        _bar_simple(axes[1], stats_s, "(b) Mode switches", "Mode switches per episode")

        plt.tight_layout()
        _save_figure(plots_dir, "figure5_scheduling_S2")
        plt.close(fig)


def write_executive_summary(all_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# AgentTwin Reproduction Summary\n")

    for scen in sorted(all_df["scenario"].unique()):
        df_s = all_df[all_df["scenario"] == scen]
        means = df_s.groupby("method")["total_reward"].mean().sort_values(ascending=False)
        lines.append(f"## {scen}: mean total reward (higher is better)\n")
        for m, v in means.items():
            lines.append(f"- {m}: {v:.2f}")
        lines.append("")

    out_path.write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AgentTwin and generate tables/figures")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to YAML config")
    parser.add_argument("--artifacts_root", type=str, default="artifacts", help="Root directory containing seed_* subfolders")
    parser.add_argument("--results_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

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

    # Optional reward shaping (incl. scheduling penalties)
    rw = env_cfg.get("reward", {})
    if isinstance(rw, dict):
        for k, v in rw.items():
            if hasattr(base_cfg.weights, str(k)):
                try:
                    setattr(base_cfg.weights, str(k), float(v))
                except Exception:
                    pass

    artifacts_root = Path(args.artifacts_root)
    results_dir = Path(args.results_dir)

    raw_dir = _ensure_dir(results_dir / "raw")
    tables_dir = _ensure_dir(results_dir / "tables")
    plots_dir = _ensure_dir(results_dir / "plots")

    all_rows: List[pd.DataFrame] = []

    for seed in seeds:
        seed_dir = artifacts_root / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"[WARN] Missing artifacts for seed {seed}: {seed_dir}")
            continue

        models = _load_models_for_seed(seed_dir)
        centralized: Optional[SAC] = models.get("centralized_sac")  # type: ignore
        group_models: Optional[Dict[str, SAC]] = models.get("multiagent")  # type: ignore
        scheduler: Optional[PPO] = models.get("scheduler_ppo")  # type: ignore

        group_policy = _build_group_control_policy(group_models) if group_models else None

        scenario_defs = get_scenario_set(
            name=scenario_set_name,
            seed=int(seed),
            episode_length_sec=int(base_cfg.episode_length_sec),
        )

        for scen in scenario_defs:
            # PID baseline (greedy demand scheduler + residual=0)
            df_pid = evaluate_method_on_scenario(
                method_name="PID (PI controller)",
                base_cfg=base_cfg,
                scenario=scen,
                scheduler_policy=_greedy_demand_scheduler,
                control_policy=lambda obs: np.zeros(12, dtype=np.float32),
                n_episodes=n_eval_episodes,
                seed_offset=seed * 10_000 + 100 * 0,
                train_seed=int(seed),
            )
            all_rows.append(df_pid)

            if centralized is not None:
                df_c = evaluate_method_on_scenario(
                    method_name="Centralized SAC",
                    base_cfg=base_cfg,
                    scenario=scen,
                    scheduler_policy=_greedy_demand_scheduler,
                    control_policy=lambda obs, m=centralized: np.asarray(m.predict(obs, deterministic=True)[0], dtype=np.float32),
                    n_episodes=n_eval_episodes,
                    seed_offset=seed * 10_000 + 100 * 1,
                    train_seed=int(seed),
                )
                all_rows.append(df_c)

            if group_policy is not None:
                df_ma = evaluate_method_on_scenario(
                    method_name="Multi-agent SAC",
                    base_cfg=base_cfg,
                    scenario=scen,
                    scheduler_policy=_greedy_demand_scheduler,
                    control_policy=group_policy,
                    n_episodes=n_eval_episodes,
                    seed_offset=seed * 10_000 + 100 * 2,
                    train_seed=int(seed),
                )
                all_rows.append(df_ma)

            if scheduler is not None and group_policy is not None:
                df_at = evaluate_method_on_scenario(
                    method_name="AgentTwin (Scheduler+MA)",
                    base_cfg=base_cfg,
                    scenario=scen,
                    scheduler_policy=lambda obs, s=scheduler: int(s.predict(obs, deterministic=True)[0]),
                    control_policy=group_policy,
                    n_episodes=n_eval_episodes,
                    seed_offset=seed * 10_000 + 100 * 3,
                    train_seed=int(seed),
                )
                all_rows.append(df_at)

                if bool(cfg_dict.get("evaluation", {}).get("include_ablation_no_shield", True)):
                    cfg_no = TEPConfig(**{k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__.keys()})
                    cfg_no.use_safety_shield = False
                    df_ns = evaluate_method_on_scenario(
                        method_name="Ablation: no shield",
                        base_cfg=cfg_no,
                        scenario=scen,
                        scheduler_policy=lambda obs, s=scheduler: int(s.predict(obs, deterministic=True)[0]),
                        control_policy=group_policy,
                        n_episodes=n_eval_episodes,
                        seed_offset=seed * 10_000 + 100 * 4,
                        train_seed=int(seed),
                    )
                    all_rows.append(df_ns)

    if not all_rows:
        raise RuntimeError("No results generated. Check artifacts_root and config seeds.")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(raw_dir / "all_episode_metrics.csv", index=False)

    make_tables(all_df, tables_dir)
    make_plots(all_df, plots_dir)
    write_executive_summary(all_df, results_dir / "executive_summary.md")

    print(f"Wrote results to: {results_dir}")


if __name__ == "__main__":
    main()