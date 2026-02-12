"""
uniquac_gibbsfit.py
===================
Standalone fitting of UNIQUAC solvent-water binary interaction parameters.
Run this BEFORE liquac_fit.py — the fitted parameters are then held fixed
during the full LIQUAC optimisation.

Equivalent to MATLAB UNIQUAC_Gibbsfit.m.

WORKFLOW
--------˜
1. Edit the USER SETTINGS block.
2. Run:
       python uniquac_gibbsfit.py
3. Copy the printed IP_UNIQUAC values into liquac_fit.py → IP_UNIQUAC.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, dual_annealing

from liquac_inputs import UNIQUACInputs
from uniquac       import UNIQUAC
from lle_flash     import LLEFlash


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SOLVENT  = "DIPA"
DATA_DIR = "/Users/lucascaldentey/Desktop/Yip Lab"

# Starting guess  [a12, a21, b12, b21]
#   where τ_ij = exp(a_ij + b_ij / T)
IP_GUESS = np.array([-2.13881699866569, -2.13155946322135,
                      541.616375907126,  587.067085535208])

BOUND_LO = np.array([-10.0, -10.0, -10000.0, -10000.0])
BOUND_HI = np.array([ 10.0,  10.0,  10000.0,  10000.0])

# ═══════════════════════════════════════════════════════════════════════════════


def _gibbs_uniquac(r, q, T_scalar, interaction_parameters):
    """
    Find binary LLE compositions via the Eubanks area method on ΔG_mix.
    Equivalent to LLEFlash.Gibbs_UNIQUAC in the MATLAB code.

    Returns (xE, xR) each as shape (2,) arrays [x_solvent, x_water].
    """
    # Fine + coarse grid matching the MATLAB implementation
    x1 = np.concatenate([np.arange(1, 1000) / 10000,
                          np.arange(100, 1000) / 1000])
    x2 = 1.0 - x1
    node_lengths = np.concatenate([np.full(999, 0.0001),
                                    np.full(900, 0.001)])

    T_vec = np.full(x1.size, T_scalar)
    obj_u = UNIQUAC()
    ln_gamma = obj_u.uniquac_calc(
        np.column_stack([x1, x2]), r, q, T_vec, interaction_parameters
    )
    dg = (x1 * np.log(x1) + x2 * np.log(x2)
        + x1 * ln_gamma[:, 0] + x2 * ln_gamma[:, 1])

    max_area = 0.0
    max_j    = 1
    xE = np.array([100.0, 100.0])
    xR = np.array([100.0, 100.0])

    # Find the point furthest from the chord (first pass, right boundary)
    for j in range(1, x1.size):
        trap  = abs(dg[0] + dg[j]) * abs(x1[0] - x1[j]) / 2.0
        curve = abs(np.sum(dg[0:j+1] * node_lengths[0:j+1]))
        diff  = trap - curve
        if diff > max_area:
            max_area = diff
            xE = np.array([x1[j], 1.0 - x1[j]])
            max_j = j

    if xE[0] == 100.0:
        return xE, xR

    # Refine the left boundary
    for k in range(1, max_j):
        trap  = abs(dg[k] + dg[max_j]) * abs(x1[k] - xE[0]) / 2.0
        curve = abs(np.sum(dg[k:max_j+1] * node_lengths[k:max_j+1]))
        diff  = trap - curve
        if diff > max_area:
            max_area = diff
            xR = np.array([x1[k], 1.0 - x1[k]])

    return xE, xR


def objective(interaction_parameters, r, q, T_exp, xE_exp, xR_exp, z_exp):
    """
    RMS objective combining relative composition errors and isoactivity.
    Matches the MATLAB convergence() function.
    """
    N = xE_exp.shape[0]
    xE_pred = np.zeros_like(xE_exp)
    xR_pred = np.zeros_like(xR_exp)

    for i in range(N):
        xE_pred[i], xR_pred[i] = _gibbs_uniquac(
            r, q, T_exp[i], interaction_parameters
        )

    # Relative RMS (same as MATLAB)
    denom = np.minimum(xE_exp, xE_pred)
    rms   = (np.sum(((xE_exp - xE_pred) / denom)**2)
           + np.sum(((xR_exp - xR_pred) / denom)**2))

    print(f"  RMS = {rms:.6f}   params = {np.round(interaction_parameters, 6)}")
    return rms


def run_fit(method="nelder-mead"):
    t0 = time.time()

    # ── load data ──────────────────────────────────────────────────────────────
    loader = UNIQUACInputs(DATA_DIR)
    r, q, T_exp, xE_exp, xR_exp = loader.species_data(SOLVENT)

    # UNIQUAC uses only the first two columns [solvent, water]
    r_bin  = r[:2]
    q_bin  = q[:2]
    xE_bin = xE_exp[:, :2]
    xR_bin = xR_exp[:, :2]
    z_bin  = 0.5 * (xE_bin + xR_bin)

    print(f"\n{'='*60}")
    print(f"UNIQUAC fit:  {SOLVENT}   ({xE_bin.shape[0]} data points)")
    print(f"Method:       {method}")
    print(f"{'='*60}\n")

    bounds = list(zip(BOUND_LO, BOUND_HI))
    scalar_obj = lambda p: objective(p, r_bin, q_bin, T_exp, xE_bin, xR_bin, z_bin)

    if method == "nelder-mead":
        result = minimize(scalar_obj, IP_GUESS, method="Nelder-Mead",
                          options={"maxiter": 10000, "xatol": 1e-8,
                                   "fatol": 1e-8, "disp": True})
    elif method == "dual-annealing":
        result = dual_annealing(scalar_obj, bounds, x0=IP_GUESS,
                                maxiter=1000, seed=42)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    IP_fitted = result.x
    elapsed   = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Optimisation complete  ({elapsed:.1f} s)")
    print(f"Final RMS    : {result.fun:.8f}")
    print(f"Fitted params: {IP_fitted}")
    print(f"\nCopy these values into liquac_fit.py → IP_UNIQUAC:")
    print(f"  IP_UNIQUAC = np.array({list(np.round(IP_fitted, 10))})")
    print(f"{'='*60}\n")

    # Save
    out_dir = Path(DATA_DIR) / SOLVENT
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({
        "parameter": ["a12", "a21", "b12", "b21"],
        "value":     IP_fitted,
    }).to_csv(out_dir / "uniquac_params.csv", index=False)

    return IP_fitted


if __name__ == "__main__":
    # Uncomment the method you want:
    run_fit(method="nelder-mead")
    # run_fit(method="dual-annealing")