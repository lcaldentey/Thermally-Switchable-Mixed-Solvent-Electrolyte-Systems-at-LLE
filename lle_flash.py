"""
lle_flash.py
============
Python port of MATLAB LLEFlash.m (Kiepe / Yip Lab).

Provides:
  - LLEFlash.flash_uniquac   : successive-substitution K-value flash for binary UNIQUAC
  - LLEFlash.gibbs_uniquac   : Eubanks area method for binary UNIQUAC (used in uniquac_gibbsfit.py)
  - LLEFlash.flash_liquac    : successive-substitution flash for the full 4-component LIQUAC system
                               (included for completeness; not used in the main workflow)

Species ordering (UNIQUAC binary): [solvent, water]
Species ordering (LIQUAC):         [solvent, water, cation, anion]
"""

import numpy as np
from scipy.optimize import brentq

from uniquac import UNIQUAC


# ── sentinel value returned when a solver fails ───────────────────────────────
_SENTINEL = 1e10


class LLEFlash:
    """Liquid–liquid equilibrium flash calculations."""

    # ── UNIQUAC Gibbs-area method (used by uniquac_gibbsfit.py) ──────────────

    def gibbs_uniquac(
        self,
        r: np.ndarray,                       # shape (2,)
        q: np.ndarray,                       # shape (2,)
        T: float,
        interaction_parameters: np.ndarray,  # shape (4,) [a12, a21, b12, b21]
    ):
        """
        Find binary LLE compositions via the Eubanks area method.
        Equivalent to MATLAB LLEFlash.Gibbs_UNIQUAC.

        Returns
        -------
        xE, xR : np.ndarray, shape (2,)  [x_solvent, x_water]
            Sentinel [100, 100] is returned for each phase that cannot be found.
        """
        # Fine grid (0.0001 steps) then coarser (0.001 steps)
        x1 = np.concatenate([np.arange(1, 1000) / 10000,
                              np.arange(100, 1000) / 1000])
        x2 = 1.0 - x1
        node_lengths = np.concatenate([np.full(999, 0.0001),
                                        np.full(900, 0.001)])

        T_vec = np.full(x1.size, float(T))
        obj_u = UNIQUAC()
        ln_gamma = obj_u.uniquac_calc(
            np.column_stack([x1, x2]), r, q, T_vec, interaction_parameters
        )
        dg = (x1 * np.log(x1) + x2 * np.log(x2)
              + x1 * ln_gamma[:, 0] + x2 * ln_gamma[:, 1])

        max_area = 0.0
        max_j    = 1
        xE = np.array([_SENTINEL, _SENTINEL])
        xR = np.array([_SENTINEL, _SENTINEL])

        # First pass: sweep right boundary to find organic-phase composition xE
        for j in range(1, x1.size):
            trap  = abs(dg[0] + dg[j]) * abs(x1[0] - x1[j]) / 2.0
            curve = abs(np.sum(dg[0:j + 1] * node_lengths[0:j + 1]))
            diff  = trap - curve
            if diff > max_area:
                max_area = diff
                xE = np.array([x1[j], 1.0 - x1[j]])
                max_j = j

        if xE[0] == _SENTINEL:
            return xE, xR

        # Second pass: sweep left boundary to find aqueous-phase composition xR
        for k in range(1, max_j):
            trap  = abs(dg[k] + dg[max_j]) * abs(x1[k] - xE[0]) / 2.0
            curve = abs(np.sum(dg[k:max_j + 1] * node_lengths[k:max_j + 1]))
            diff  = trap - curve
            if diff > max_area:
                max_area = diff
                xR = np.array([x1[k], 1.0 - x1[k]])

        if xR[0] == _SENTINEL:
            xR = np.array([_SENTINEL, _SENTINEL])

        return xE, xR

    # ── successive-substitution flash (UNIQUAC binary) ────────────────────────

    def flash_uniquac(
        self,
        z: np.ndarray,                       # shape (2,)  feed mole fractions
        r: np.ndarray,                       # shape (2,)
        q: np.ndarray,                       # shape (2,)
        T: float,
        interaction_parameters: np.ndarray,
    ):
        """
        Successive-substitution K-value flash for a binary UNIQUAC system.
        Equivalent to MATLAB LLEFlash.Flash_UNIQUAC.

        Returns
        -------
        xE, xR : np.ndarray, shape (2,)
            Sentinel arrays (filled with 1e10) if convergence fails.
        """
        errorK, tol, count, EOF, xR, xE = self._set_params(z)
        obj_u = UNIQUAC()

        T_vec = np.array([T])
        gamma_R = np.exp(obj_u.uniquac_calc(xR[np.newaxis], r, q, T_vec, interaction_parameters)).ravel()
        gamma_E = np.exp(obj_u.uniquac_calc(xE[np.newaxis], r, q, T_vec, interaction_parameters)).ravel()
        K = gamma_R / gamma_E

        while errorK > tol:
            if np.all(np.isreal(K)) and np.all(K > 1e-10):
                # Solve Rachford-Rice for EOF
                try:
                    EOF = brentq(lambda e: np.sum(z * (1 - K) / (1 + e * (K - 1))),
                                 1e-6, 1 - 1e-6, xtol=1e-10)
                except ValueError:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break

                xR_new = z / (1 + EOF * (K - 1))
                xE_new = xR_new * K
                xR = xR_new / xR_new.sum()
                xE = xE_new / xE_new.sum()

                gamma_R = np.exp(obj_u.uniquac_calc(xR[np.newaxis], r, q, T_vec, interaction_parameters)).ravel()
                gamma_E = np.exp(obj_u.uniquac_calc(xE[np.newaxis], r, q, T_vec, interaction_parameters)).ravel()
                K_new = gamma_R / gamma_E

                errorK = np.sum(np.abs(K_new - K) / K)
                K = K_new

                count += 1
                if count > 1000:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break

                # Single-phase check
                if abs(K[0] - 1.0) < tol:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break
            else:
                xE = np.full_like(z, _SENTINEL)
                xR = np.full_like(z, _SENTINEL)
                break

        return xE, xR

    # ── successive-substitution flash (full LIQUAC, 4-component) ─────────────

    def flash_liquac(
        self,
        z: np.ndarray,          # shape (4,)
        r: np.ndarray,          # shape (4,)
        q: np.ndarray,          # shape (4,)
        T: float,
        interaction_params: np.ndarray,   # shape (14,)  full ip vector
        MW: np.ndarray,         # shape (4,)
        valency: np.ndarray,    # shape (4,)
        rho_solvent: float,
        dielec_solvent: float,
        salt: str,
        objMR,
        objSR,
        objLR,
    ):
        """
        Successive-substitution flash for the full 4-component LIQUAC system.
        Equivalent to MATLAB LLEFlash.Flash_LIQUAC.

        Note: per MATLAB comments, this solver often struggles near LLE;
        the Gibbs minimisation route (GibbsMinimization) is preferred.

        Returns
        -------
        xE, xR : np.ndarray, shape (4,)
        """
        from gibbs_minimization import _mean_activity_coeff

        errorK, tol, count, EOF, xR, xE = self._set_params(z)

        b_ij      = interaction_params[0:4]
        c_ij      = interaction_params[4:8]
        b_jcja    = interaction_params[8]
        c_jcja    = interaction_params[9]
        ip_UNIQUAC = interaction_params[10:14]

        def _ln_gamma(x_row):
            x2 = x_row.reshape(1, -1)
            T_v   = np.array([T])
            rho_v = np.array([rho_solvent])
            die_v = np.array([dielec_solvent])
            lng = (objLR.func_LR(MW, valency, x2, T_v, rho_v, die_v, salt)
                   + objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x2, MW, valency, salt)
                   + objSR.func_SR(x2, r, q, T_v, ip_UNIQUAC, MW))
            return _mean_activity_coeff(lng, valency).ravel()

        gamma_E = np.exp(_ln_gamma(xE))
        gamma_R = np.exp(_ln_gamma(xR))
        K = gamma_R / gamma_E

        while errorK > tol:
            if np.all(np.isreal(K)) and np.all(K > 1e-10):
                try:
                    EOF = brentq(lambda e: np.sum(z * (1 - K) / (1 + e * (K - 1))),
                                 1e-6, 1 - 1e-6, xtol=1e-10)
                except ValueError:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break

                xR_new = z / (1 + EOF * (K - 1))
                xE_new = xR_new * K
                xR = xR_new / xR_new.sum()
                xE = xE_new / xE_new.sum()

                gamma_E = np.exp(_ln_gamma(xE))
                gamma_R = np.exp(_ln_gamma(xR))
                K_new = gamma_R / gamma_E

                errorK = np.sum(np.abs(K_new - K) / K)
                K = K_new

                count += 1
                if count > 1000:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break

                if abs(K[0] - 1.0) < tol:
                    xE = np.full_like(z, _SENTINEL)
                    xR = np.full_like(z, _SENTINEL)
                    break
            else:
                xE = np.full_like(z, _SENTINEL)
                xR = np.full_like(z, _SENTINEL)
                break

        return xE, xR

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _set_params(z: np.ndarray):
        """
        Initialise solver state.  Matches MATLAB setParams().

        Returns
        -------
        errorK, tol, count, EOF_guess, xR, xE
        """
        errorK    = 1.0
        tol       = 1e-5
        count     = 0
        EOF_guess = 0.5

        # Avoid exact zeros (→ NaN in log/gamma evaluations)
        xR = np.full(len(z), 1e-50)
        xE = np.full(len(z), 1e-50)
        xR[0], xR[1] = 0.02, 0.98
        xE[0], xE[1] = 0.98, 0.02

        return errorK, tol, count, EOF_guess, xR, xE