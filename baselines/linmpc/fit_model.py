"""Fit mode-dependent linear models for the linMPC baseline.

We fit, for each operating mode k:

  s_{t+1} = A_k s_t + B_k u_t + c_k

where:
- s_t is the 53-dim normalized state [xmeas_norm, xmv_norm]
- u_t is the 12-dim normalized residual action

The fit uses ridge regression on collected tuples saved by
`baselines.linmpc.collect_id_data`.

Outputs
-------
A single NPZ file containing arrays:
  A : (6, 53, 53)
  B : (6, 53, 12)
  c : (6, 53)
  meta : dict

Usage
-----
python -m baselines.linmpc.fit_model --data_dir artifacts/linmpc/data --out_path artifacts/linmpc/model_linmpc.npz

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _ridge_fit(X: np.ndarray, U: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Y ≈ A X + B U + c using ridge regression.

    Shapes
    ------
    X : (N, n)
    U : (N, m)
    Y : (N, n)

    Returns
    -------
    A : (n, n)
    B : (n, m)
    c : (n,)
    """

    if X.ndim != 2 or U.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, U, Y must be 2D arrays")
    if X.shape[0] != U.shape[0] or X.shape[0] != Y.shape[0]:
        raise ValueError("X, U, Y must have the same number of rows")

    N, n = X.shape
    m = U.shape[1]

    Phi = np.concatenate([X, U, np.ones((N, 1), dtype=X.dtype)], axis=1)  # (N, n+m+1)

    # Solve (Phi^T Phi + lam I) Theta = Phi^T Y
    XtX = Phi.T @ Phi
    reg = float(lam) * np.eye(XtX.shape[0], dtype=X.dtype)
    Theta = np.linalg.solve(XtX + reg, Phi.T @ Y)  # (n+m+1, n)

    A = Theta[:n, :].T
    B = Theta[n : n + m, :].T
    c = Theta[-1, :].T
    return A.astype(np.float32), B.astype(np.float32), c.astype(np.float32)


def _load_npz(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit mode-dependent linear models for linMPC")
    ap.add_argument("--data_dir", type=str, default="artifacts/linmpc/data", help="Directory with id_*.npz files")
    ap.add_argument("--out_path", type=str, default="artifacts/linmpc/model_linmpc.npz", help="Output NPZ path")
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Modes to fit")
    ap.add_argument("--lambda_reg", type=float, default=1e-4, help="Ridge regularization strength")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = [int(m) for m in args.modes]

    # Initialize arrays (we will fill for each mode index 1..6)
    A_all = np.zeros((6, 53, 53), dtype=np.float32)
    B_all = np.zeros((6, 53, 12), dtype=np.float32)
    c_all = np.zeros((6, 53), dtype=np.float32)

    fit_stats: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        files = sorted(data_dir.glob(f"id_mode{mode}_seed*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No identification data found for mode {mode} under {data_dir}. "
                "Run baselines.linmpc.collect_id_data first."
            )

        Xs: List[np.ndarray] = []
        Us: List[np.ndarray] = []
        Ys: List[np.ndarray] = []

        for f in files:
            d = _load_npz(f)
            Xs.append(np.asarray(d["X"], dtype=np.float32))
            Us.append(np.asarray(d["U"], dtype=np.float32))
            Ys.append(np.asarray(d["Y"], dtype=np.float32))

        X = np.vstack(Xs)
        U = np.vstack(Us)
        Y = np.vstack(Ys)

        A, B, c = _ridge_fit(X, U, Y, lam=float(args.lambda_reg))

        # Fit error (one-step prediction MSE)
        Yhat = (X @ A.T) + (U @ B.T) + c[None, :]
        mse = float(np.mean((Y - Yhat) ** 2))

        A_all[mode - 1] = A
        B_all[mode - 1] = B
        c_all[mode - 1] = c

        fit_stats[str(mode)] = {
            "n_samples": float(X.shape[0]),
            "mse_one_step": mse,
        }

        print(f"[linmpc] mode {mode}: fitted on {X.shape[0]} samples, one-step MSE={mse:.3e}")

    meta = {
        "modes": modes,
        "state_dim": 53,
        "input_dim": 12,
        "lambda_reg": float(args.lambda_reg),
        "fit_stats": fit_stats,
    }

    np.savez_compressed(out_path, A=A_all, B=B_all, c=c_all, meta=json.dumps(meta))
    print(f"[linmpc] wrote model: {out_path}")


if __name__ == "__main__":
    main()
