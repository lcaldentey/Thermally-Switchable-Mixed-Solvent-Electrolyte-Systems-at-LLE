"""
MediumRange.py
==============
Medium-range contribution to activity coefficients in the LIQUAC model
(Kiepe et al., 2006).

Species ordering: [solvent, water, cation, anion]
"""

import numpy as np


# Mapping from salt name to the a_4 constant used in the B_ion term.
_A4_MAP = {
    "LiCl":   0.1451,
    "LiBr":   0.0695,
    "LiI":    0.1797,
    "LiOH":   0.0894,
    "LiNO3":  0.1820,
    "Li2SO4": 0.2936,
    "LiClO3": 0.1325,
    "CaCl2":  0.2170,
}
_A4_DEFAULT = 0.1250


class MediumRange:
    """Computes ln(γ_MR) for all four species."""

    # ── public API ────────────────────────────────────────────────────────────

    def func_MR(
        self,
        b_ij: np.ndarray,        # shape (4,)
        c_ij: np.ndarray,        # shape (4,)
        b_jcja: float,
        c_jcja: float,
        x: np.ndarray,           # shape (N,4)
        MW: np.ndarray,          # shape (4,)
        valency: np.ndarray,     # shape (4,)
        salt: str,
    ) -> np.ndarray:             # shape (N,4)
        """Medium-range ln(γ), no explicit temperature dependence."""
        m_ion, ionic_strength, x_saltfree, mean_MW = self._variables(x, MW, valency)
        B_solv, dB_solv, B_ion, dB_ion = self._interaction_terms(
            b_ij, c_ij, b_jcja, c_jcja, ionic_strength, salt
        )
        ln_gamma_solv = self._solvent_gamma(
            m_ion, x_saltfree, ionic_strength, MW, mean_MW,
            B_solv, dB_solv, B_ion, dB_ion
        )
        ln_gamma_ions = self._ions_gamma(
            mean_MW, m_ion, x_saltfree,
            B_solv, dB_solv, B_ion, dB_ion, MW, b_ij, c_ij, valency
        )
        return np.hstack([ln_gamma_solv, ln_gamma_ions])

    def func_MR_T(
        self,
        b_ij: np.ndarray,
        c_ij: np.ndarray,
        b_jcja: float,
        c_jcja: float,
        x: np.ndarray,
        MW: np.ndarray,
        valency: np.ndarray,
        salt: str,
        T: np.ndarray,           # shape (N,)
    ) -> np.ndarray:
        """Medium-range ln(γ) with temperature dependence in B terms."""
        m_ion, ionic_strength, x_saltfree, mean_MW = self._variables(x, MW, valency)
        B_solv, dB_solv, B_ion, dB_ion = self._interaction_terms_T(
            b_ij, c_ij, b_jcja, c_jcja, ionic_strength, T, salt
        )
        ln_gamma_solv = self._solvent_gamma(
            m_ion, x_saltfree, ionic_strength, MW, mean_MW,
            B_solv, dB_solv, B_ion, dB_ion
        )
        ln_gamma_ions = self._ions_gamma(
            mean_MW, m_ion, x_saltfree,
            B_solv, dB_solv, B_ion, dB_ion, MW, b_ij, c_ij, valency
        )
        return np.hstack([ln_gamma_solv, ln_gamma_ions])

    # ── private helpers ───────────────────────────────────────────────────────

    def _variables(self, x, MW, valency):
        kg_solvent   = x[:, 0] * MW[0] + x[:, 1] * MW[1]
        m_ion        = x[:, 2:4] / kg_solvent[:, np.newaxis]
        ionic_strength = 0.5 * np.sum(m_ion * valency[2:4]**2, axis=1)
        x_saltfree   = np.column_stack([
            x[:, 0] / (x[:, 0] + x[:, 1]),
            x[:, 1] / (x[:, 0] + x[:, 1]),
        ])
        mean_MW = x_saltfree[:, 0] * MW[0] + x_saltfree[:, 1] * MW[1]
        return m_ion, ionic_strength, x_saltfree, mean_MW

    @staticmethod
    def _a_constants(salt: str):
        a1 = -1.2
        a3 = -1.0
        a4 = _A4_MAP.get(salt, _A4_DEFAULT)
        a2 = 2 * a4
        return a1, a2, a3, a4

    def _interaction_terms(self, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, salt):
        a1, a2, a3, a4 = self._a_constants(salt)
        sqrt_I = np.sqrt(ionic_strength)
        exp_solv = np.exp(a1 * sqrt_I + a2 * ionic_strength)
        exp_ion  = np.exp(a3 * sqrt_I + a4 * ionic_strength)

        # B_solv shape: (N,4)  [solv-cat, solv-an, water-cat, water-an]
        B_solv  = b_ij + c_ij * exp_solv[:, np.newaxis]
        dB_solv = (a1 / 2.0 / sqrt_I + a2)[:, np.newaxis] * c_ij * exp_solv[:, np.newaxis]

        # B_ion shape: (N,)
        B_ion  = b_jcja + c_jcja * exp_ion
        dB_ion = (a3 / 2.0 / sqrt_I + a4) * c_jcja * exp_ion
        return B_solv, dB_solv, B_ion, dB_ion

    def _interaction_terms_T(self, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, T, salt):
        a1, a2, a3, a4 = self._a_constants(salt)
        sqrt_I = np.sqrt(ionic_strength)
        exp_solv = np.exp(a1 * sqrt_I + a2 * ionic_strength)
        exp_ion  = np.exp(a3 * sqrt_I + a4 * ionic_strength)

        B_solv  = b_ij / T[:, np.newaxis] + c_ij / T[:, np.newaxis] * exp_solv[:, np.newaxis]
        dB_solv = (a1 / 2.0 / sqrt_I + a2)[:, np.newaxis] * c_ij / T[:, np.newaxis] * exp_solv[:, np.newaxis]

        B_ion  = b_jcja / T + c_jcja / T * exp_ion
        dB_ion = (a3 / 2.0 / sqrt_I + a4) * c_jcja / T * exp_ion
        return B_solv, dB_solv, B_ion, dB_ion

    def _solvent_gamma(self, m_ion, x_saltfree, ionic_strength, MW, mean_MW,
                       B_solv, dB_solv, B_ion, dB_ion):
        # Shared intermediate: sum over ions of x_sf * m * (B + I*dB)
        shared = (
            np.sum(x_saltfree[:, 0:1] * m_ion * (B_solv[:, 0:2] + ionic_strength[:, np.newaxis] * dB_solv[:, 0:2]), axis=1)
          + np.sum(x_saltfree[:, 1:2] * m_ion * (B_solv[:, 2:4] + ionic_strength[:, np.newaxis] * dB_solv[:, 2:4]), axis=1)
        )

        Solv1_T1 = np.sum(m_ion * B_solv[:, 0:2], axis=1)
        Solv1_T2 = MW[0] / mean_MW * shared

        Solv2_T1 = np.sum(m_ion * B_solv[:, 2:4], axis=1)
        Solv2_T2 = MW[1] / mean_MW * shared

        Solv_T3  = m_ion[:, 0] * m_ion[:, 1] * (B_ion + ionic_strength * dB_ion)

        ln_g1 = Solv1_T1 - Solv1_T2 - MW[0] * Solv_T3
        ln_g2 = Solv2_T1 - Solv2_T2 - MW[1] * Solv_T3
        return np.column_stack([ln_g1, ln_g2])

    def _ions_gamma(self, mean_MW, m_ion, x_saltfree,
                    B_solv, dB_solv, B_ion, dB_ion,
                    MW, b_ij, c_ij, valency):
        # Term 1: weighted sum of B_solv over salt-free fractions
        Cat_T1 = (1.0 / mean_MW
                  * (x_saltfree[:, 0] * B_solv[:, 0] + x_saltfree[:, 1] * B_solv[:, 2]))
        An_T1  = (1.0 / mean_MW
                  * (x_saltfree[:, 0] * B_solv[:, 1] + x_saltfree[:, 1] * B_solv[:, 3]))

        # Term 2: valency²/(2*mean_MW) × sum of x_sf * m * dB_solv
        # v_sq shape: (2,), mean_MW shape: (N,), shared_dB shape: (N,)
        v_sq = valency[2:4]**2
        shared_dB = (
            np.sum(x_saltfree[:, 0:1] * m_ion * dB_solv[:, 0:2], axis=1)
          + np.sum(x_saltfree[:, 1:2] * m_ion * dB_solv[:, 2:4], axis=1)
        )
        # Result shape: (N, 2)
        Ions_T2 = (v_sq[np.newaxis, :] / (2.0 * mean_MW[:, np.newaxis])) * shared_dB[:, np.newaxis]

        # Term 3: counter-ion molality × B_ion
        Ions_T3 = np.column_stack([
            m_ion[:, 1] * B_ion,
            m_ion[:, 0] * B_ion,
        ])

        # Term 4: 0.5 * z² * m_cat * m_an * dB_ion
        Ions_T4 = (0.5 * v_sq[np.newaxis, :]
                   * (m_ion[:, 0] * m_ion[:, 1] * dB_ion)[:, np.newaxis])

        # Reference states (infinite dilution in pure water)
        Ions_Ref_Cat = (1.0 / MW[1]) * (b_ij[2] + c_ij[2])
        Ions_Ref_An  = (1.0 / MW[1]) * (b_ij[3] + c_ij[3])

        ln_g_cat = Cat_T1 + Ions_T2[:, 0] + Ions_T3[:, 0] + Ions_T4[:, 0] - Ions_Ref_Cat
        ln_g_an  = An_T1  + Ions_T2[:, 1] + Ions_T3[:, 1] + Ions_T4[:, 1] - Ions_Ref_An
        return np.column_stack([ln_g_cat, ln_g_an])