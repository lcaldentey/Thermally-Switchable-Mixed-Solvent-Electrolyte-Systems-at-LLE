"""
uniquac.py
==========
Standalone UNIQUAC model for a binary solvent-water system (no ions).
Used independently to fit solvent-water interaction parameters before
the full LIQUAC fitting.

Species ordering: [solvent, water]  (2-component system)
"""

import numpy as np


class UNIQUAC:
    """UNIQUAC activity coefficient model for binary/ternary systems."""

    def uniquac_calc(
        self,
        x: np.ndarray,                       # shape (N, C)
        r: np.ndarray,                       # shape (C,)
        q: np.ndarray,                       # shape (C,)
        T: np.ndarray,                       # shape (N,) or scalar
        interaction_parameters: np.ndarray,  # shape depends on parameterisation
    ) -> np.ndarray:                         # shape (N, C)
        """Return ln(γ_UNIQUAC) = combinatorial + residual."""
        T = np.atleast_1d(T)
        if T.ndim == 0 or T.shape[0] != x.shape[0]:
            T = np.full(x.shape[0], float(T.flat[0]))
        sum_xr, sum_xq, _ = self._variables(x, r, q)
        ln_g_comb  = self._combinatorial(r, q, sum_xr, sum_xq)
        ln_g_resid = self._residual(x, q, sum_xq, T, interaction_parameters)
        return ln_g_comb + ln_g_resid

    def uniquac_detailed_calc(self, x, r, q, T, interaction_parameters):
        """Return (ln_gamma_comb, ln_gamma_resid) separately."""
        T = np.atleast_1d(T)
        if T.shape[0] != x.shape[0]:
            T = np.full(x.shape[0], float(T.flat[0]))
        sum_xr, sum_xq, _ = self._variables(x, r, q)
        ln_g_comb  = self._combinatorial(r, q, sum_xr, sum_xq)
        ln_g_resid = self._residual(x, q, sum_xq, T, interaction_parameters)
        return ln_g_comb, ln_g_resid

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _variables(x, r, q):
        sum_xr = x @ r          # (N,)
        sum_xq = x @ q          # (N,)
        phi    = x * q / sum_xq[:, np.newaxis]
        return sum_xr, sum_xq, phi

    @staticmethod
    def _combinatorial(r, q, sum_xr, sum_xq):
        return (1 - r / sum_xr[:, np.newaxis]
                + np.log(r / sum_xr[:, np.newaxis])
                - 5 * q * (1
                           - r * sum_xq[:, np.newaxis] / sum_xr[:, np.newaxis] / q
                           + np.log(r * sum_xq[:, np.newaxis]
                                    / sum_xr[:, np.newaxis] / q)))

    def _residual(self, x, q, sum_xq, T, interaction_parameters):
        N, C = x.shape
        ln_g = np.zeros((N, C))
        for h in range(N):
            tau = self.tau_calculation(interaction_parameters, T[h])
            for i in range(C):
                sum_phi_j_tau_ji = np.sum(x[h, :] * q * tau[:, i]) / sum_xq[h]
                sum_calc = 0.0
                for j in range(C):
                    denom = np.sum(q * x[h, :] * tau[:, j])
                    sum_calc += q[j] * x[h, j] * tau[i, j] / denom
                ln_g[h, i] = q[i] * (1.0 - np.log(sum_phi_j_tau_ji) - sum_calc)
        return ln_g

    @staticmethod
    def tau_calculation(interaction_parameters, T_scalar):
        """
        Build τ matrix from interaction_parameters.
        Supports 2, 4, 6, and 12 parameter formats (matching MATLAB original).
        """
        ip = np.asarray(interaction_parameters).ravel()
        n  = len(ip)

        if n == 2:
            tau = np.array([
                [1.0,                      np.exp(ip[0] / T_scalar)],
                [np.exp(ip[1] / T_scalar), 1.0],
            ])

        elif n == 4:
            tau_12 = np.exp(ip[0] + ip[2] / T_scalar)
            tau_21 = np.exp(ip[1] + ip[3] / T_scalar)
            tau = np.array([[1.0, tau_12], [tau_21, 1.0]])

        elif n == 6:
            tau_12 = np.exp(ip[0] / T_scalar)
            tau_21 = np.exp(ip[1] / T_scalar)
            tau_13 = np.exp(ip[2] / T_scalar)
            tau_31 = np.exp(ip[3] / T_scalar)
            tau_23 = np.exp(ip[4] / T_scalar)
            tau_32 = np.exp(ip[5] / T_scalar)
            tau = np.array([
                [1.0,    tau_12, tau_13],
                [tau_21, 1.0,    tau_23],
                [tau_31, tau_32, 1.0   ],
            ])

        elif n == 12:
            tau_12 = np.exp(ip[0] + ip[6]  / T_scalar)
            tau_21 = np.exp(ip[1] + ip[7]  / T_scalar)
            tau_13 = np.exp(ip[2] + ip[8]  / T_scalar)
            tau_31 = np.exp(ip[3] + ip[9]  / T_scalar)
            tau_23 = np.exp(ip[4] + ip[10] / T_scalar)
            tau_32 = np.exp(ip[5] + ip[11] / T_scalar)
            tau = np.array([
                [1.0,    tau_12, tau_13],
                [tau_21, 1.0,    tau_23],
                [tau_31, tau_32, 1.0   ],
            ])

        else:
            raise ValueError(
                f"interaction_parameters must have 2, 4, 6, or 12 elements; got {n}"
            )

        return tau