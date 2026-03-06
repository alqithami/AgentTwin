"""Linear data-driven MPC baseline.

This baseline:
1) Collects identification data from the *real* TEP simulator under the
   decentralized PI controller with injected residual excitation.
2) Fits a mode-dependent linear model (one model per operating mode).
3) Runs a linear MPC over the fitted model to compute residual actions.

The MPC is implemented as a condensed quadratic program solved with OSQP.
"""

from .mpc_controller import LinearResidualMPC
