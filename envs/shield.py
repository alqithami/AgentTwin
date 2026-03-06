"""Safety shields for residual control.

The reproduction package uses a *residual* control structure:

    u_applied = u_PI(x) + u_residual

A safety shield filters u_residual to reduce the probability of hard constraint
violations (pressure, temperature, etc.).

Two implementations are provided:
- SafetyShieldHeuristic: inexpensive, conservative, rule-based fallback
- SafetyShieldQP: QP-based shield using a local sensitivity model and OSQP

This code is intended to be robust and runnable on typical CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import time
import warnings

import numpy as np

# Optional dependency: OSQP
try:
    import osqp
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    osqp = None
    sp = None

from third_party.tep.simulator import TEPSimulator, ControlMode
from third_party.tep.controllers import DecentralizedController


@dataclass
class QPShieldConfig:
    """Configuration for the QP-based shield."""

    alpha: float = 0.20
    lambda_reg: float = 1e-3
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    max_iter: int = 2000
    polish: bool = True
    warm_start: bool = True
    activation_tol: float = 1e-6


class SafetyShieldHeuristic:
    """Conservative, rule-based safety filter.

    If the plant is near key limits, the residual is set to zero.
    """

    def __init__(self, limits: "SafetyLimits"):
        self.limits = limits

    def filter(self, xmeas: np.ndarray, residual: np.ndarray, residual_max: float) -> Tuple[np.ndarray, Dict[str, float]]:
        res = np.clip(np.asarray(residual, dtype=np.float32).reshape(-1), -residual_max, residual_max)
        activated = False

        # When close to constraints, remove residual action.
        if (
            xmeas[6] > 0.97 * self.limits.reactor_pressure_max
            or xmeas[8] > 0.97 * self.limits.reactor_temp_max
            or xmeas[12] > 0.97 * self.limits.sep_pressure_max
            or xmeas[15] > 0.97 * self.limits.stripper_pressure_max
        ):
            res[...] = 0.0
            activated = True

        # Reactor level guard
        if (
            xmeas[7] < self.limits.reactor_level_min + 2.0
            or xmeas[7] > self.limits.reactor_level_max - 2.0
        ):
            res[...] = 0.0
            activated = True

        return res, {"activated": float(activated), "solve_time_ms": 0.0, "status": 0.0}


# Module-level cache: avoid recalibrating sensitivity matrix for every environment instance.
_SENSITIVITY_CACHE: Dict[tuple, np.ndarray] = {}

def _calibrate_sensitivity_matrix(
    *,
    meas_indices: Sequence[int],
    dt_sec: int = 1,
    mv_delta: float = 1.0,
    mode: int = 1,
    seed: int = 0,
    backend: str = "python",
) -> np.ndarray:
    """Finite-difference sensitivity of selected measurements w.r.t. MV residual.

    We approximate a local (steady-state) sensitivity:

        y_{t+dt} \approx y_t + B u_residual

    where B is estimated around the canonical initial steady state under the
    decentralized TE PI controller.

    Returns
    -------
    B : np.ndarray
        Shape (len(meas_indices), 12)
    """

    meas_indices = [int(i) for i in meas_indices]
    key = (str(backend), int(dt_sec), tuple(meas_indices))
    cached = _SENSITIVITY_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    n_y = len(meas_indices)
    n_u = 12

    B = np.zeros((n_y, n_u), dtype=np.float64)

    for j in range(n_u):
        # Base simulation
        sim_base = TEPSimulator(control_mode=ControlMode.OPEN_LOOP, backend=backend, random_seed=seed)
        sim_base.initialize()
        ctrl_base = DecentralizedController(mode=mode)
        ctrl_base.reset()

        # Perturbed simulation
        sim_pert = TEPSimulator(control_mode=ControlMode.OPEN_LOOP, backend=backend, random_seed=seed)
        sim_pert.initialize()
        ctrl_pert = DecentralizedController(mode=mode)
        ctrl_pert.reset()

        # Apply baseline for dt seconds, and baseline+residual for dt seconds
        for t in range(int(dt_sec)):
            yb = sim_base.process.get_xmeas()
            ub = sim_base.process.get_xmv()
            base_xmv = ctrl_base.calculate(yb, ub, t)

            yp = sim_pert.process.get_xmeas()
            up = sim_pert.process.get_xmv()
            pert_xmv = ctrl_pert.calculate(yp, up, t)

            # Apply base actions
            for i in range(n_u):
                sim_base.set_mv(i + 1, float(np.clip(base_xmv[i], 0.0, 100.0)))

            # Apply perturbed actions: add residual on MV j
            for i in range(n_u):
                add = mv_delta if i == j else 0.0
                sim_pert.set_mv(i + 1, float(np.clip(pert_xmv[i] + add, 0.0, 100.0)))

            sim_base.step(1)
            sim_pert.step(1)

        y_end_base = sim_base.process.get_xmeas()
        y_end_pert = sim_pert.process.get_xmeas()

        diff = (y_end_pert - y_end_base) / float(mv_delta)
        for k, idx in enumerate(meas_indices):
            B[k, j] = float(diff[idx])

    B = B.astype(np.float64)
    _SENSITIVITY_CACHE[key] = B
    return B.copy()



class SafetyShieldQP:
    """QP-based safety shield using OSQP.

    The shield solves:

        min_u 0.5||u - u_nom||^2 + 0.5*lambda*||u||^2
        s.t.  A u <= b(x)
              -u_max <= u <= u_max

    where A is derived from a local sensitivity model and b(x) depends on the
    current distance to safety limits.
    """

    def __init__(
        self,
        *,
        limits: "SafetyLimits",
        residual_max: float,
        cfg: QPShieldConfig,
        backend: str = "python",
        seed: int = 0,
        dt_sec: int = 1,
    ):
        if osqp is None or sp is None:
            raise ImportError("OSQP/scipy.sparse not available; install 'osqp' and 'scipy'.")

        self.limits = limits
        self.residual_max = float(residual_max)
        self.cfg = cfg

        # Sensitivity indices (0-based measurement indices)
        self._idx_pressure = 6
        self._idx_temp = 8
        self._idx_level = 7
        self._idx_sep_p = 12
        self._idx_strip_p = 15

        self._meas_indices = [self._idx_pressure, self._idx_temp, self._idx_level, self._idx_sep_p, self._idx_strip_p]

        # Calibrate (steady-state) sensitivity matrix once.
        self.B = _calibrate_sensitivity_matrix(
            meas_indices=self._meas_indices,
            dt_sec=dt_sec,
            mv_delta=1.0,
            mode=1,
            seed=seed,
            backend=backend,
        )  # shape (5,12)

        # Build linear constraint matrix A_lin (6 x 12)
        # 1) Reactor pressure upper
        # 2) Reactor temperature upper
        # 3) Separator pressure upper
        # 4) Stripper pressure upper
        # 5) Reactor level upper
        # 6) Reactor level lower  (implemented as -B_level u <= alpha*(level - min))
        b_p = self.B[self._meas_indices.index(self._idx_pressure), :]
        b_T = self.B[self._meas_indices.index(self._idx_temp), :]
        b_L = self.B[self._meas_indices.index(self._idx_level), :]
        b_Sp = self.B[self._meas_indices.index(self._idx_sep_p), :]
        b_Stp = self.B[self._meas_indices.index(self._idx_strip_p), :]

        self.A_lin = np.vstack([b_p, b_T, b_Sp, b_Stp, b_L, -b_L]).astype(np.float64)  # (6,12)

        # OSQP setup: P = (1+lambda) I
        P = sp.eye(12, format="csc") * float(1.0 + self.cfg.lambda_reg)

        # A_total = [A_lin; I]
        A = sp.vstack([sp.csc_matrix(self.A_lin), sp.eye(12, format="csc")], format="csc")

        # l/u will be updated online; start with loose bounds
        l = np.hstack([-np.inf * np.ones(6), -self.residual_max * np.ones(12)]).astype(np.float64)
        u = np.hstack([np.inf * np.ones(6), self.residual_max * np.ones(12)]).astype(np.float64)

        q0 = np.zeros(12, dtype=np.float64)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=P,
            q=q0,
            A=A,
            l=l,
            u=u,
            verbose=False,
            warm_start=bool(self.cfg.warm_start),
            polish=bool(self.cfg.polish),
            eps_abs=float(self.cfg.eps_abs),
            eps_rel=float(self.cfg.eps_rel),
            max_iter=int(self.cfg.max_iter),
        )

        # Cache indices for updating u-vector (upper bounds)
        self._u_base = u.copy()
        self._l_base = l.copy()

    def _compute_b(self, xmeas: np.ndarray) -> np.ndarray:
        lim = self.limits
        alpha = float(self.cfg.alpha)

        # Upper bounds
        b_pressure = alpha * float(lim.reactor_pressure_max - xmeas[self._idx_pressure])
        b_temp = alpha * float(lim.reactor_temp_max - xmeas[self._idx_temp])
        b_sep = alpha * float(lim.sep_pressure_max - xmeas[self._idx_sep_p])
        b_str = alpha * float(lim.stripper_pressure_max - xmeas[self._idx_strip_p])
        b_level_hi = alpha * float(lim.reactor_level_max - xmeas[self._idx_level])

        # Lower bound reformulated as -B_L u <= alpha*(level - level_min)
        b_level_lo = alpha * float(xmeas[self._idx_level] - lim.reactor_level_min)

        return np.array([b_pressure, b_temp, b_sep, b_str, b_level_hi, b_level_lo], dtype=np.float64)

    def filter(self, xmeas: np.ndarray, residual: np.ndarray, residual_max: float) -> Tuple[np.ndarray, Dict[str, float]]:
        # Keep residual_max from caller for backwards compatibility.
        res_max = float(residual_max)
        if abs(res_max - self.residual_max) > 1e-9:
            warnings.warn(
                "SafetyShieldQP was initialized with residual_max=%s but called with %s. Using the call-time value." % (self.residual_max, res_max)
            )
            self.residual_max = res_max

        u_nom = np.clip(np.asarray(residual, dtype=np.float64).reshape(12), -res_max, res_max)

        b = self._compute_b(xmeas)

        # Quick feasibility check for u_nom
        Au = self.A_lin.dot(u_nom)
        if np.all(Au <= b + 1e-9) and np.all(np.abs(u_nom) <= res_max + 1e-9):
            return u_nom.astype(np.float32), {"activated": 0.0, "solve_time_ms": 0.0, "status": 1.0}

        # Update objective and constraints
        q = -u_nom.astype(np.float64)

        u_vec = self._u_base.copy()
        u_vec[:6] = b
        u_vec[6:] = res_max

        l_vec = self._l_base.copy()
        l_vec[6:] = -res_max

        t0 = time.perf_counter()
        self._solver.update(q=q, l=l_vec, u=u_vec)
        res = self._solver.solve()
        t_ms = (time.perf_counter() - t0) * 1000.0

        status_val = 0.0
        if res.info is not None and hasattr(res.info, "status"):
            status = str(res.info.status).lower()
            if "solved" in status:
                status_val = 1.0

        if status_val < 0.5 or res.x is None:
            # Fail-safe: remove residual
            return np.zeros(12, dtype=np.float32), {"activated": 1.0, "solve_time_ms": float(t_ms), "status": status_val}

        u_filt = np.asarray(res.x, dtype=np.float64).reshape(12)
        activated = float(np.linalg.norm(u_filt - u_nom, ord=np.inf) > float(self.cfg.activation_tol))

        return u_filt.astype(np.float32), {"activated": activated, "solve_time_ms": float(t_ms), "status": status_val}


