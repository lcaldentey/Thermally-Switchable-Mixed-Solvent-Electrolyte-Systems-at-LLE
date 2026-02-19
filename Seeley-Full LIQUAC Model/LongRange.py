"""
LongRange.py
============
Long-range (Debye-Hückel) contribution to activity coefficients in the
LIQUAC model (Kiepe et al.).

Species ordering convention throughout this codebase:
    index 0 : organic solvent  (e.g. DIPA)
    index 1 : water
    index 2 : cation           (e.g. Na+)
    index 3 : anion            (e.g. Cl-)
"""

import numpy as np


class LongRange:
    """Computes ln(γ_LR) for all four species via a modified Debye-Hückel
    equation.  All inputs are NumPy arrays (rows = data points).
    """

    def func_LR(
        self,
        MW: np.ndarray,          # shape (4,)  molecular weights [kg/mol]
        valency: np.ndarray,      # shape (4,)  ionic charges (0 for solvents)
        x: np.ndarray,            # shape (N,4) mole fractions
        T: np.ndarray,            # shape (N,)  absolute temperature [K]
        rho_solvent: np.ndarray,  # shape (N,)  solvent density [kg/m³]
        dielec_solvent: np.ndarray,  # shape (N,) solvent dielectric constant
        salt: str,
    ) -> np.ndarray:              # shape (N,4) ln activity coefficients
        """Return the long-range ln(γ) contributions for all species."""

        # ── molality & ionic strength ────────────────────────────────────────
        kg_solvent = x[:, 0] * MW[0] + x[:, 1] * MW[1]
        molality_cation = x[:, 2] / kg_solvent
        molality_anion  = x[:, 3] / kg_solvent
        ionic_strength  = 0.5 * (molality_cation * valency[2]**2
                                  + molality_anion  * valency[3]**2)

        # salt-free volume fractions
        x_saltfree = np.column_stack([
            x[:, 0] / (x[:, 0] + x[:, 1]),
            x[:, 1] / (x[:, 0] + x[:, 1]),
        ])

        # ── densities  [kg/m³] ───────────────────────────────────────────────
        # Water density correlation (T in K)
        rho_H2O = -0.0043 * T**2 + 2.3147 * T + 687.31

        # Volume fraction of each solvent (used for density mixing)
        denom_phi = (x_saltfree[:, 0] * MW[0] / rho_solvent
                     + x_saltfree[:, 1] * MW[1] / rho_H2O)
        phi_prime = np.column_stack([
            x_saltfree[:, 0] * MW[0] / rho_solvent / denom_phi,
            x_saltfree[:, 1] * MW[1] / rho_H2O    / denom_phi,
        ])
        rho_solution = phi_prime[:, 0] * rho_solvent + phi_prime[:, 1] * rho_H2O

        # ── dielectric constants ─────────────────────────────────────────────
        T_C = T - 273.15   # Celsius
        dielec_H2O = (87.74
                      - 0.4008  * T_C
                      + 9.398e-4 * T_C**2
                      - 1.410e-6 * T_C**3)

        # Oster mixing rule for solution dielectric constant
        dielec_solution = (dielec_H2O
                           + ((dielec_solvent - 1) * (dielec_solvent * 2 + 1)
                              / (2 * dielec_solvent) - (dielec_H2O - 1))
                           * phi_prime[:, 0])

        # ── Debye-Hückel parameters ──────────────────────────────────────────
        A = (1.327757e5 * np.sqrt(rho_solution)
             / (dielec_solution * T)**1.5)
        b = (6.359696 * np.sqrt(rho_solution)
             / np.sqrt(dielec_solution * T))

        # ── activity coefficients ────────────────────────────────────────────
        sqrt_I = np.sqrt(ionic_strength)
        kappa = (1 + b * sqrt_I
                 - 1.0 / (1 + b * sqrt_I)
                 - 2 * np.log(1 + b * sqrt_I))

        ln_gamma_solv_LR = np.column_stack([
            2 * MW[0] * A * rho_solution / b**3 / rho_solvent * kappa,
            2 * MW[1] * A * rho_solution / b**3 / rho_H2O    * kappa,
        ])

        ln_gamma_ions_LR = np.column_stack([
            -valency[2]**2 * A * sqrt_I / (1 + b * sqrt_I),
            -valency[3]**2 * A * sqrt_I / (1 + b * sqrt_I),
        ])

        return np.hstack([ln_gamma_solv_LR, ln_gamma_ions_LR])