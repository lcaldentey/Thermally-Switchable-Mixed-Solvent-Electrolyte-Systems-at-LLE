"""
ShortRange.py
=============
Short-range (UNIQUAC) contribution to activity coefficients in the LIQUAC
model.  Ions are referenced to infinite dilution in pure water.

Species ordering: [solvent, water, cation, anion]
"""

import numpy as np


class ShortRange:
    """Computes ln(γ_SR) for all four species via a modified UNIQUAC equation."""

    def func_SR(
        self,
        x: np.ndarray,                  # shape (N,4)
        r: np.ndarray,                  # shape (4,)  van-der-Waals volumes
        q: np.ndarray,                  # shape (4,)  van-der-Waals surfaces
        T: np.ndarray,                  # shape (N,)  absolute temperature [K]
        interaction_parameters: np.ndarray,  # shape (4,) [a12, a21, b12, b21]
        MW: np.ndarray,                 # shape (4,)  molecular weights [kg/mol]
    ) -> np.ndarray:                    # shape (N,4)
        """Return ln(γ_SR) for all species including reference-state correction."""
        sum_xr, sum_xq = self._variables(x, r, q)
        ln_gamma_comb  = self._combinatorial(r, q, sum_xr, sum_xq)
        ln_gamma_resid = self._residual(x, q, sum_xq, T, interaction_parameters)
        ln_gamma = ln_gamma_comb + ln_gamma_resid

        # Convert molality reference state for ions
        MW_avg  = x[:, 0] * MW[0] + x[:, 1] * MW[1]
        molality = x[:, 2:4] / MW_avg[:, np.newaxis]
        ln_gamma[:, 2:4] -= np.log(
            MW[1] / MW_avg[:, np.newaxis]
            + MW[1] * np.sum(molality, axis=1, keepdims=True)
        )
        return ln_gamma

    def output_SR(self, x, r, q, T, interaction_parameters):
        """Return (ln_gamma_resid, ln_gamma_comb) separately for diagnostics."""
        sum_xr, sum_xq = self._variables(x, r, q)
        ln_gamma_comb  = self._combinatorial(r, q, sum_xr, sum_xq)
        ln_gamma_resid = self._residual(x, q, sum_xq, T, interaction_parameters)
        return ln_gamma_resid, ln_gamma_comb

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _variables(x, r, q):
        sum_xr = x @ r  # shape (N,)
        sum_xq = x @ q  # shape (N,)
        return sum_xr, sum_xq

    @staticmethod
    def _combinatorial(r, q, sum_xr, sum_xq):
        """Staverman-Guggenheim combinatorial term for all species."""
        # Broadcasting: r and q are (4,), sum_xr/sum_xq are (N,)
        ln_g = (1 - r / sum_xr[:, np.newaxis]
                + np.log(r / sum_xr[:, np.newaxis])
                - 5 * q * (1
                           - r * sum_xq[:, np.newaxis] / sum_xr[:, np.newaxis] / q
                           + np.log(r * sum_xq[:, np.newaxis]
                                    / sum_xr[:, np.newaxis] / q)))

        # Reference state for ions: infinite dilution in pure water (index 1)
        ln_g_inf_ions = (1 - r[2:4] / r[1]
                         + np.log(r[2:4] / r[1])
                         - 5 * q[2:4] * (1
                                          - r[2:4] * q[1] / (r[1] * q[2:4])
                                          + np.log(r[2:4] * q[1] / (r[1] * q[2:4]))))
        ln_g[:, 2:4] -= ln_g_inf_ions  # broadcast over rows
        return ln_g

    @staticmethod
    def _tau_calculation(interaction_parameters, T_scalar):
        """
        Build 4×4 τ matrix.  Only solvent(0)–water(1) interactions are non-zero;
        all ion interactions are set to τ = 1 (as per Kiepe et al.).
        τ_ij = exp(a_ij + b_ij / T)
        """
        a12, a21, b12, b21 = interaction_parameters
        tau_12 = np.exp(a12 + b12 / T_scalar)
        tau_21 = np.exp(a21 + b21 / T_scalar)
        tau = np.array([
            [1.0,    tau_12, 1.0, 1.0],
            [tau_21, 1.0,    1.0, 1.0],
            [1.0,    1.0,    1.0, 1.0],
            [1.0,    1.0,    1.0, 1.0],
        ])
        return tau

    def _residual(self, x, q, sum_xq, T, interaction_parameters):
        """UNIQUAC residual term for all species, row by row."""
        N, S = x.shape
        ln_gamma_resid = np.zeros((N, S))

        for h in range(N):
            tau = self._tau_calculation(interaction_parameters, T[h])
            for i in range(S):
                # Σ_j [ x_j * q_j * τ_ji ] / Σ_k [ x_k * q_k ]
                sum_phi_j_tau_ji = (np.sum(x[h, :] * q * tau[:, i])
                                    / sum_xq[h])
                sum_calc = 0.0
                for j in range(S):
                    denom = np.sum(q * x[h, :] * tau[:, j])
                    sum_calc += q[j] * x[h, j] * tau[i, j] / denom

                ln_gamma_resid[h, i] = q[i] * (1.0
                                                 - np.log(sum_phi_j_tau_ji)
                                                 - sum_calc)

        # Reference state for ions: infinite dilution in pure water
        # Use T[0] as representative (consistent with MATLAB behaviour)
        tau0 = self._tau_calculation(interaction_parameters, T[0])
        ln_resid_inf_cat = q[2] * (1.0 - np.log(tau0[1, 2]) - tau0[2, 1])
        ln_resid_inf_an  = q[3] * (1.0 - np.log(tau0[1, 3]) - tau0[3, 1])
        ln_gamma_resid[:, 2] -= ln_resid_inf_cat
        ln_gamma_resid[:, 3] -= ln_resid_inf_an

        return ln_gamma_resid