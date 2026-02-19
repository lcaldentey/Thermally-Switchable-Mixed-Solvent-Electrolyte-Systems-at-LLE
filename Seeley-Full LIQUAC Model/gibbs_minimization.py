"""
gibbs_minimization.py
=====================
Eubanks area method for finding LLE tie lines via Gibbs free energy
minimisation.  Equivalent to MATLAB GibbsMinimization.m.

Species ordering: [solvent, water, cation, anion]
"""

import numpy as np
from scipy.signal import argrelmin

from LongRange   import LongRange
from MediumRange import MediumRange
from ShortRange  import ShortRange


def _mean_activity_coeff(ln_gamma: np.ndarray, valency: np.ndarray) -> np.ndarray:
    """
    Replace columns 2 and 3 with the mean ionic activity coefficient,
    weighted by |z_+| and |z_-|.  Matches the MATLAB inline calculation:
        ln_γ_±  = (|z+|*ln_γ+ + |z-|*ln_γ-) / (|z+| + |z-|)
    Modifies ln_gamma in-place and returns it.
    """
    vp = abs(valency[2])
    vm = abs(valency[3])
    mean = (vp * ln_gamma[:, 2] + vm * ln_gamma[:, 3]) / (vp + vm)
    ln_gamma[:, 2] = mean
    ln_gamma[:, 3] = mean
    return ln_gamma


def _dg_mix(x_tot: np.ndarray, ln_gamma: np.ndarray) -> np.ndarray:
    """
    Gibbs free energy of mixing along a tie-line scan.
    dg = Σ_i  x_i * (ln x_i + ln γ_i)

    Rows where any x_i ≈ 0 (scan boundary) produce log(0) = -inf.
    These are filtered by the np.isreal() check in the calling code,
    so the numpy warning is suppressed here.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x_tot[:, 0] * (np.log(x_tot[:, 0]) + ln_gamma[:, 0])
              + x_tot[:, 1] * (np.log(x_tot[:, 1]) + ln_gamma[:, 1])
              + x_tot[:, 2] * (np.log(x_tot[:, 2]) + ln_gamma[:, 2])
              + x_tot[:, 3] * (np.log(x_tot[:, 3]) + ln_gamma[:, 3]))


def _build_scan(z_ternary, D, sigfigs):
    """
    Build the 1-D composition scan along tie-line direction D.
    Returns (x_NaCl, node_lengths, x_tot) or None if the range is degenerate.

    Branch condition matches MATLAB exactly: D > -1 uses the positive branch.
    D = 0 is handled safely: z[1]/0 = ±inf, which is then caught by the
    isfinite guard below and clamped to step (matching MATLAB's Inf → clamp).
    """
    step = 10**(-sigfigs)

    # MATLAB uses 'if D > -1' (not 'if D > 0') so D=0 goes into this branch.
    # Division by zero at D=0 produces ±inf in both Python and MATLAB;
    # the isfinite/<=0 guards below clamp it identically to MATLAB's behaviour.
    with np.errstate(divide='ignore', invalid='ignore'):
        if D > -1:
            x_NaCl_min = -(z_ternary[1] / D - z_ternary[2])
            x_NaCl_max = (D * z_ternary[2] + 1 - z_ternary[1]) / (1 + D)
        else:
            x_NaCl_min = (D * z_ternary[2] + 1 - z_ternary[1]) / (1 + D)
            x_NaCl_max = z_ternary[2] - z_ternary[1] / D

    if not np.isfinite(x_NaCl_min) or x_NaCl_min <= 0:
        x_NaCl_min = step
    if not np.isfinite(x_NaCl_max) or x_NaCl_max > 1:
        x_NaCl_max = 1.0

    x_NaCl_min = round(x_NaCl_min, sigfigs)
    x_NaCl_max = round(x_NaCl_max, sigfigs)

    x_NaCl = np.arange(x_NaCl_min, x_NaCl_max, step)
    if x_NaCl.size < 2:
        return None

    node_lengths = np.full(x_NaCl.size, step)
    y_H2O  = -D * (x_NaCl_max - x_NaCl) + (1 - x_NaCl_max)
    z_solv = 1 - x_NaCl - y_H2O

    x_tot = np.column_stack([z_solv, y_H2O, x_NaCl, x_NaCl])
    row_sums = x_tot.sum(axis=1, keepdims=True)
    x_tot /= row_sums
    return x_NaCl, node_lengths, x_tot


def _compute_ln_gamma(x_tot, T_scalar, rho_solvent_scalar, dielec_solvent_scalar,
                      MW, valency, r, q, b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC,
                      salt, objLR, objMR, objSR):
    """Vectorised LIQUAC ln(γ) calculation for an (N,4) composition array."""
    N = x_tot.shape[0]
    T_vec     = np.full(N, T_scalar)
    rho_v     = np.full(N, rho_solvent_scalar)
    dielec_v  = np.full(N, dielec_solvent_scalar)

    ln_g = (objLR.func_LR(MW, valency, x_tot, T_vec, rho_v, dielec_v, salt)
           + objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x_tot, MW, valency, salt)
           + objSR.func_SR(x_tot, r, q, T_vec, ip_UNIQUAC, MW))
    return _mean_activity_coeff(ln_g, valency)


def _eubanks_area(dg, x_NaCl, node_lengths, D_scale):
    """
    Run the Eubanks area search on a pre-computed dg_mix curve.
    Returns (max_area, idx_E, idx_R) where indices index into x_NaCl.
    Returns (0, -1, -1) if fewer than two local minima are found.
    """
    n = len(dg)

    # ---- initial guess at two minima ----------------------------------------
    # Method 1: local minima of (dg - baseline)
    m_base = (dg[-1] - dg[0]) / (x_NaCl[-1] - x_NaCl[0])
    dg_line = dg[0] + m_base * (x_NaCl - x_NaCl[0])
    deltas  = dg - dg_line
    mins_delta = argrelmin(deltas, order=1)[0]

    if len(mins_delta) < 2:
        mins_delta = argrelmin(dg, order=1)[0]

    if len(mins_delta) < 2:
        return 0.0, -1, -1

    guesses = mins_delta[:2]

    def area(i, k):
        trap  = abs(dg[i] + dg[k]) * abs(x_NaCl[i] - x_NaCl[k]) / 2.0
        curve = abs(np.sum(dg[i:k+1] * node_lengths[i:k+1]))
        return D_scale * (trap - curve)

    # Initial best from the two guesses
    max_area = area(guesses[0], guesses[1])
    idx_E = int(guesses[0])
    idx_R = int(guesses[1])

    # Sweep left boundary (E) leftward from guesses[0]
    for j in range(guesses[0]):
        diff = area(j, idx_R)
        if diff > max_area:
            max_area = diff
            idx_E = j
        elif diff < max_area:
            break

    # Sweep left boundary (E) rightward toward guesses[1]
    for j in range(guesses[0], guesses[1]):
        diff = area(j, idx_R)
        if diff > max_area:
            max_area = diff
            idx_E = j
        elif diff < max_area:
            break

    # Sweep right boundary (R) rightward from guesses[1]
    for k in range(guesses[1] + 1, n):
        diff = area(idx_E, k)
        if diff > max_area:
            max_area = diff
            idx_R = k
        elif diff < max_area:
            break

    # Sweep right boundary (R) leftward back toward idx_E
    for k in range(idx_E + 1, guesses[1]):
        diff = area(idx_E, k)
        if diff > max_area:
            max_area = diff
            idx_R = k
        elif diff < max_area:
            break

    return max_area, idx_E, idx_R


class GibbsMinimization:
    """Phase equilibrium via the Eubanks area maximisation method."""

    # ── public methods ────────────────────────────────────────────────────────

    def gibbs_liquac_with_D(
        self,
        z_exp_ternary: np.ndarray,    # shape (3,)  salt-free + salt ternary fracs
        r: np.ndarray,
        q: np.ndarray,
        T: float,
        interaction_params: np.ndarray,
        MW: np.ndarray,
        valency: np.ndarray,
        rho_solvent: float,
        dielec_solvent: float,
        salt: str,
        objMR: MediumRange,
        objSR: ShortRange,
        objLR: LongRange,
        D: float,
    ):
        """
        Given a known tie-line slope D, find equilibrium compositions xE and xR.
        Equivalent to MATLAB Gibbs_LIQUAC_withD.
        """
        b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC = self._unpack_ip(interaction_params)
        sigfigs = 5

        scan = _build_scan(z_exp_ternary, D, sigfigs)
        if scan is None:
            return np.full(4, 100.0), np.full(4, 100.0)
        x_NaCl, node_lengths, x_tot = scan

        ln_gamma = _compute_ln_gamma(
            x_tot, T, rho_solvent, dielec_solvent,
            MW, valency, r, q, b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC,
            salt, objLR, objMR, objSR,
        )
        dg = _dg_mix(x_tot, ln_gamma)

        if not np.all(np.isreal(dg)):
            return np.full(4, 100.0), np.full(4, 100.0)

        D_scale = np.sqrt(1 + D**2)
        max_area, idx_E, idx_R = _eubanks_area(dg, x_NaCl, node_lengths, D_scale)

        if idx_E < 0:
            return np.full(4, 100.0), np.full(4, 100.0)

        return x_tot[idx_E, :].copy(), x_tot[idx_R, :].copy()

    def gibbs_liquac_eubanks(
        self,
        z_exp_ternary: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        T: float,
        interaction_params: np.ndarray,
        MW: np.ndarray,
        valency: np.ndarray,
        rho_solvent: float,
        dielec_solvent: float,
        salt: str,
        objMR: MediumRange,
        objSR: ShortRange,
        objLR: LongRange,
        coarsegrain: bool = False,
    ):
        """
        Search over tie-line slopes D to find the equilibrium compositions
        xE, xR, and the best slope best_D.
        Equivalent to MATLAB Gibbs_LIQUAC_Eubanks.
        """
        b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC = self._unpack_ip(interaction_params)
        sigfigs = 4 if coarsegrain else 5

        xE = np.full(4, 100.0)
        xR = np.full(4, 100.0)
        best_D   = 0.0
        max_area = 0.0
        D        = 0.0

        while D < 100:
            scan = _build_scan(z_exp_ternary, D, sigfigs)
            if scan is None:
                D += 1
                continue
            x_NaCl, node_lengths, x_tot = scan

            ln_gamma = _compute_ln_gamma(
                x_tot, T, rho_solvent, dielec_solvent,
                MW, valency, r, q, b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC,
                salt, objLR, objMR, objSR,
            )
            dg = _dg_mix(x_tot, ln_gamma)

            if not np.all(np.isreal(dg)):
                D += 2.5
                continue

            D_scale = np.sqrt(1 + D**2)
            area_inner, idx_E, idx_R = _eubanks_area(
                dg, x_NaCl, node_lengths, D_scale
            )

            if idx_E < 0:
                D += 1
                continue

            if area_inner > max_area:
                max_area = area_inner
                xE       = x_tot[idx_E, :].copy()
                xR       = x_tot[idx_R, :].copy()
                best_D   = D
                D += 1          # small step once we have a good candidate
            elif area_inner > 0 and area_inner < max_area:
                break           # area started decreasing — we've passed the peak
            else:
                D += 1

        return xE, xR, best_D

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_ip(ip):
        """Unpack flat interaction_params vector into named components."""
        b_ij      = ip[0:4]
        c_ij      = ip[4:8]
        b_jcja    = ip[8]
        c_jcja    = ip[9]
        ip_UNIQUAC = ip[10:14]
        return b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC