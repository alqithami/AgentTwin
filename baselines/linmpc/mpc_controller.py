"""Linear MPC controller (residual) for the AgentTwin TEP environment.

We solve a *condensed* quadratic program at each control step:

  min_{u_0..u_{N-1}}  Σ_{k=1..N} ||s_k - s_ref||_Q^2 + Σ_{k=0..N-1} ||u_k||_R^2

  s_{k+1} = A s_k + B u_k + c

with input bounds: u_min <= u_k <= u_max (elementwise).

Implementation details
----------------------
- State s is the 53-dim normalized [xmeas_norm(41), xmv_norm(12)].
- The model is mode-dependent (one (A,B,c) triple per operating mode).
- We precompute condensed prediction matrices and the constant Hessian for OSQP.

This controller produces a **normalized residual action** in [-1, 1]^12 that
is compatible with the residual-control environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import osqp

from third_party.tep.constants import OPERATING_MODES

# Match the normalization used by envs.tep_env
_DEF_MAX_XMEAS = np.array([
    1.0, 10000.0, 10000.0, 20.0, 1.0, 1.0, 5000.0, 100.0, 200.0, 1.0,
    1.0, 100.0, 5000.0, 1.0, 100.0, 5000.0, 1.0, 200.0, 1.0, 1.0,
    1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0
], dtype=np.float32)

_DEF_MAX_XMV = np.array([100.0] * 12, dtype=np.float32)


@dataclass
class LinMPCConfig:
    """Configuration for the linMPC baseline."""

    horizon: int = 10

    # Q weights (diagonal) on tracking variables
    q_xmeas: float = 1.0
    q_xmv: float = 1e-2

    # R weight (diagonal) on residual magnitude
    r_u: float = 1e-1

    # Input bounds in normalized units
    u_bound: float = 1.0

    # OSQP settings
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3
    osqp_max_iter: int = 10_000


def _mode_from_obs(obs_control: np.ndarray) -> int:
    """Infer current operating mode from the control observation."""

    # obs_control: [xmeas_norm(41), xmv_norm(12), mode_onehot(6)]
    onehot = obs_control[41 + 12 : 41 + 12 + 6]
    return int(np.argmax(onehot)) + 1


def _build_ref_state(mode: int) -> np.ndarray:
    """Build the normalized reference state s_ref for a given mode."""

    mode_cfg = OPERATING_MODES[int(mode)]

    xmeas_ref = np.zeros(41, dtype=np.float32)
    for idx, val in mode_cfg.xmeas_setpoints.items():
        i = int(idx)
        xmeas_ref[i] = float(val) / float(_DEF_MAX_XMEAS[i])

    xmv_ref = (np.asarray(mode_cfg.xmv_setpoints, dtype=np.float32) / _DEF_MAX_XMV).astype(np.float32)

    return np.concatenate([xmeas_ref, xmv_ref]).astype(np.float32)


def _build_Q_diag(cfg: LinMPCConfig) -> np.ndarray:
    """Diagonal of Q for the 53-dim state."""

    q = np.zeros(53, dtype=np.float32)

    # Only penalize the standard TE setpoint variables (consistent across modes)
    tracked = list(OPERATING_MODES[1].xmeas_setpoints.keys())
    for idx in tracked:
        q[int(idx)] = float(cfg.q_xmeas)

    # Small weight on MV deviation (keeps action modest and near PI setpoint)
    q[41:] = float(cfg.q_xmv)
    return q


def _block_diag_diag(d: np.ndarray, N: int) -> sp.csc_matrix:
    """Return sparse block-diagonal matrix with diag(d) repeated N times."""

    d_rep = np.tile(np.asarray(d, dtype=np.float64), int(N))
    return sp.diags(d_rep, offsets=0, format="csc")


def _build_prediction_matrices(A: np.ndarray, B: np.ndarray, c: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build condensed prediction matrices.

    Returns
    -------
    S0 : (nN, n)  mapping from s0 to stacked states [s1..sN]
    Su : (nN, mN) mapping from stacked inputs U to stacked states
    Sc : (nN,)    constant offset
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)

    n = A.shape[0]
    m = B.shape[1]
    N = int(N)

    # Precompute powers A^k
    A_pows = [np.eye(n, dtype=np.float64)]
    for k in range(1, N + 1):
        A_pows.append(A_pows[-1] @ A)

    S0 = np.zeros((n * N, n), dtype=np.float64)
    Su = np.zeros((n * N, m * N), dtype=np.float64)
    Sc = np.zeros((n * N,), dtype=np.float64)

    # Cumulative sum for constant term: sum_{i=0}^{k-1} A^i c
    c_cum = np.zeros(n, dtype=np.float64)

    for k in range(1, N + 1):
        # s_k = A^k s0 + sum_{j=0}^{k-1} A^{k-1-j} B u_j + sum_{i=0}^{k-1} A^i c
        S0[(k - 1) * n : k * n, :] = A_pows[k]

        # input contributions
        for j in range(k):
            Akj = A_pows[k - 1 - j]
            Su[(k - 1) * n : k * n, j * m : (j + 1) * m] = Akj @ B

        # constant contribution
        if k == 1:
            c_cum = c.copy()
        else:
            c_cum = c_cum + (A_pows[k - 1] @ c)
        Sc[(k - 1) * n : k * n] = c_cum

    return S0, Su, Sc


class _ModeOSQP:
    """Per-mode cached condensed QP structures and OSQP solver."""

    def __init__(self, *, A: np.ndarray, B: np.ndarray, c: np.ndarray, cfg: LinMPCConfig, mode: int):
        self.cfg = cfg
        self.mode = int(mode)

        n = int(A.shape[0])
        m = int(B.shape[1])
        N = int(cfg.horizon)

        self.n = n
        self.m = m
        self.N = N

        # Precompute condensed dynamics
        self.S0, self.Su, self.Sc = _build_prediction_matrices(A, B, c, N)

        # Cost matrices
        Q_diag = _build_Q_diag(cfg).astype(np.float64)
        R_diag = (np.ones(m, dtype=np.float64) * float(cfg.r_u)).astype(np.float64)

        Qbar = _block_diag_diag(Q_diag, N)
        Rbar = _block_diag_diag(R_diag, N)

        Su_sp = sp.csc_matrix(self.Su)

        # Hessian: 2*(Su^T Q Su + R)
        P = (Su_sp.T @ Qbar @ Su_sp) + Rbar
        P = (P + P.T) * 0.5  # enforce symmetry
        self.P = (2.0 * P).tocsc()

        # Precompute q = M_S0 s0 + q_offset_mode
        # M = 2 Su^T Q
        self.M = (2.0 * (Su_sp.T @ Qbar)).tocsc()  # (mN, nN)
        self.MS0 = (self.M @ sp.csc_matrix(self.S0)).toarray()  # (mN, n)
        self.MSc = (self.M @ self.Sc.reshape(-1, 1)).reshape(-1)  # (mN,)

        # Reference stacked vector (nN,) for this mode
        s_ref = _build_ref_state(self.mode).astype(np.float64)
        Sref = np.tile(s_ref, N).astype(np.float64)
        self.q_offset = self.MSc - (self.M @ Sref.reshape(-1, 1)).reshape(-1)

        # OSQP problem: min 1/2 U^T P U + q^T U  s.t. l <= U <= u
        mN = m * N
        A_ineq = sp.eye(mN, format="csc")
        l = -float(cfg.u_bound) * np.ones(mN, dtype=np.float64)
        u = +float(cfg.u_bound) * np.ones(mN, dtype=np.float64)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P,
            q=np.zeros(mN, dtype=np.float64),
            A=A_ineq,
            l=l,
            u=u,
            warm_start=True,
            verbose=False,
            eps_abs=float(cfg.osqp_eps_abs),
            eps_rel=float(cfg.osqp_eps_rel),
            max_iter=int(cfg.osqp_max_iter),
        )

    def solve(self, s0: np.ndarray) -> np.ndarray:
        s0 = np.asarray(s0, dtype=np.float64).reshape(self.n)

        q = (self.MS0 @ s0) + self.q_offset
        self.prob.update(q=q)

        res = self.prob.solve()
        status = ""
        if res.info is not None and hasattr(res.info, "status"):
            status = str(res.info.status).lower()
        if ("solved" not in status) or (res.x is None):
            return np.zeros(self.m, dtype=np.float32)

        U = np.asarray(res.x, dtype=np.float64).reshape(self.m * self.N)
        u0 = U[: self.m]
        return np.clip(u0, -float(self.cfg.u_bound), float(self.cfg.u_bound)).astype(np.float32)


class LinearResidualMPC:
    """Mode-dependent linear MPC that outputs normalized residual actions."""

    def __init__(self, model_path: str | Path, cfg: Optional[LinMPCConfig] = None):
        self.model_path = Path(model_path)
        self.cfg = cfg or LinMPCConfig()

        with np.load(self.model_path, allow_pickle=True) as d:
            self.A_all = np.asarray(d["A"], dtype=np.float32)
            self.B_all = np.asarray(d["B"], dtype=np.float32)
            self.c_all = np.asarray(d["c"], dtype=np.float32)

        if self.A_all.shape != (6, 53, 53) or self.B_all.shape != (6, 53, 12) or self.c_all.shape != (6, 53):
            raise ValueError(
                f"Unexpected model shapes: A{self.A_all.shape}, B{self.B_all.shape}, c{self.c_all.shape}. "
                "Expected A(6,53,53), B(6,53,12), c(6,53)."
            )

        self._cache: Dict[int, _ModeOSQP] = {}

    def _get_mode_solver(self, mode: int) -> _ModeOSQP:
        mode = int(mode)
        if mode not in self._cache:
            A = self.A_all[mode - 1]
            B = self.B_all[mode - 1]
            c = self.c_all[mode - 1]
            self._cache[mode] = _ModeOSQP(A=A, B=B, c=c, cfg=self.cfg, mode=mode)
        return self._cache[mode]

    def __call__(self, obs_control: np.ndarray) -> np.ndarray:
        """Return residual action in normalized units [-1,1]^12."""

        obs_control = np.asarray(obs_control, dtype=np.float32).reshape(-1)
        if obs_control.size < 41 + 12 + 6:
            raise ValueError(f"obs_control has wrong size {obs_control.size}; expected >= 59")

        mode = _mode_from_obs(obs_control)
        solver = self._get_mode_solver(mode)

        s0 = obs_control[: 41 + 12]
        u0 = solver.solve(s0)
        return u0
