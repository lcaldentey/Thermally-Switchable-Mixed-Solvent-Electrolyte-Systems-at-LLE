"""
liquac_fit.py
=============
Main driver for fitting LIQUAC medium-range interaction parameters
[b_ij, c_ij] to experimental LLE tie-line data.

Equivalent to MATLAB LIQUAC.m.

WORKFLOW
--------
1. Edit the USER SETTINGS block below (salt, solvent, data_dir, ip_guess).
2. Run:
       python liquac_fit.py
3. The script prints RMS at each iteration and saves results to:
       <data_dir>/<salt>-<solvent>/results_liquac.csv
       <data_dir>/<salt>-<solvent>/D_values.csv   (for use in d_resolution.py)

OPTIMISATION
------------
Two modes are available (uncomment as needed):
  - dual_annealing  : global search, slower but more robust
  - Nelder-Mead     : local refinement from a good starting guess

The objective function is:
    RMS = 1000 * [ Σ(xE_pred - xE_exp)² + Σ(xR_pred - xR_exp)² ]
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, dual_annealing

from liquac_inputs      import LIQUACInputs
from LongRange         import LongRange
from MediumRange       import MediumRange
from ShortRange        import ShortRange
from gibbs_minimization import GibbsMinimization


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS — edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

SALT    = "NaCl"
SOLVENT = "DIPA"

# Root directory containing the <salt>-<solvent> data folder
# e.g. "/Users/lucascaldentey/Desktop"
DATA_DIR = "/Users/lucascaldentey/Desktop/Yip Lab"

# UNIQUAC parameters for the solvent-water binary (held fixed during LIQUAC fit)
# Determine these first using uniquac_gibbsfit.py
IP_UNIQUAC = np.array([-2.138817, -2.13155946, 541.61637591,  587.06708554])

# Initial guess for the four LIQUAC interaction parameters being fitted:
#   [b_solvent-cation, b_solvent-anion, c_solvent-cation, c_solvent-anion]
IP_GUESS = np.array([-1.49245819676001,  1.99466864109731,
                      0.52614759793941,   0.70575476122774])

# Search bounds for optimisation (applied to all four parameters)
BOUND_LO = -10.0
BOUND_HI =  10.0

# Mesh resolution for the Gibbs scan
# coarsegrain=True  → 4 sig-figs (faster, less accurate)
# coarsegrain=False → 5 sig-figs (slower, more accurate)
COARSEGRAIN = False

# Maximum parallel workers (set to 1 to disable parallelism)
N_WORKERS = 4

# ═══════════════════════════════════════════════════════════════════════════════


def _build_ip_full(interaction_params, ip_Gmehling, ip_UNIQUAC):
    """
    Combine the four fitted solvent-ion parameters with the fixed Gmehling
    water-ion parameters and UNIQUAC parameters into the single flat vector
    expected by GibbsMinimization.

    MATLAB equivalent:
        b_ij   = [interaction_params(1:2), ip_Gmehling(1:2)]
        c_ij   = [interaction_params(3:4), ip_Gmehling(3:4)]
        b_jcja = ip_Gmehling(5)
        c_jcja = ip_Gmehling(6)
        ip     = [b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC]
    """
    b_ij   = np.concatenate([interaction_params[0:2], ip_Gmehling[0:2]])
    c_ij   = np.concatenate([interaction_params[2:4], ip_Gmehling[2:4]])
    b_jcja = ip_Gmehling[4]
    c_jcja = ip_Gmehling[5]
    return np.concatenate([b_ij, c_ij, [b_jcja, c_jcja], ip_UNIQUAC])


def _solve_one(args):
    """
    Worker function: solve a single data point.
    Returns (i, xE_row, xR_row, D_val).
    Defined at module level so it can be pickled for multiprocessing.
    """
    (i, z_ternary, T, rho_s, diel_s, ip_full,
     r, q, MW, valency, salt, coarsegrain) = args

    objLR = LongRange()
    objMR = MediumRange()
    objSR = ShortRange()
    objG  = GibbsMinimization()

    xE, xR, D = objG.gibbs_liquac_eubanks(
        z_ternary, r, q, T, ip_full, MW, valency,
        rho_s, diel_s, salt, objMR, objSR, objLR, coarsegrain
    )
    return i, xE, xR, D


def objective(interaction_params, data, ip_Gmehling, ip_UNIQUAC, r, q, MW,
              valency, salt, coarsegrain, n_workers):
    """
    Objective function minimised by the optimiser.

    Returns RMS_AllAbsolute = 1000 × [Σ(xE-xE_exp)² + Σ(xR-xR_exp)²]
    """
    ip_full = _build_ip_full(interaction_params, ip_Gmehling, ip_UNIQUAC)

    z_ternary_all = data["z_ternary"]
    T_all         = data["T_exp"]
    rho_all       = data["rho_solvent"]
    diel_all      = data["dielec_solvent"]
    xE_exp        = data["xE_exp"]
    xR_exp        = data["xR_exp"]
    N             = z_ternary_all.shape[0]

    xE_pred = np.zeros_like(xE_exp)
    xR_pred = np.zeros_like(xR_exp)
    D_vals  = np.zeros(N)

    work_args = [
        (i, z_ternary_all[i], T_all[i], rho_all[i], diel_all[i],
         ip_full, r, q, MW, valency, salt, coarsegrain)
        for i in range(N)
    ]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_one, a): a[0] for a in work_args}
            for fut in as_completed(futures):
                i, xE, xR, D = fut.result()
                xE_pred[i] = xE
                xR_pred[i] = xR
                D_vals[i]  = D
    else:
        for args in work_args:
            i, xE, xR, D = _solve_one(args)
            xE_pred[i] = xE
            xR_pred[i] = xR
            D_vals[i]  = D

    rms = 1000.0 * (np.sum((xE_exp - xE_pred)**2)
                  + np.sum((xR_exp - xR_pred)**2))
    print(f"  RMS = {rms:.6f}   params = {np.round(interaction_params, 6)}")
    return rms, xE_pred, xR_pred, D_vals


def _objective_scalar(interaction_params, data, ip_Gmehling, ip_UNIQUAC,
                       r, q, MW, valency, salt, coarsegrain, n_workers):
    """Scalar wrapper for scipy optimisers (returns only the RMS value)."""
    rms, *_ = objective(interaction_params, data, ip_Gmehling, ip_UNIQUAC,
                        r, q, MW, valency, salt, coarsegrain, n_workers)
    return rms


def run_fit(method="nelder-mead"):
    """
    Load data, run the optimisation, save results.

    Parameters
    ----------
    method : "nelder-mead"  → local Nelder-Mead search  (fast, needs good guess)
             "dual-annealing" → global search             (slow, more robust)
    """
    t0 = time.time()

    # ── load data ──────────────────────────────────────────────────────────────
    loader = LIQUACInputs(DATA_DIR)
    r, q, MW, valency = loader.species_data(SALT, SOLVENT)
    (z_exp, xE_exp, xR_exp, T_exp,
     rho_solvent, dielec_solvent,
     Selec_exp, ip_Gmehling) = loader.experimental_data(SALT, SOLVENT)

    z_ternary = loader.to_ternary(z_exp)

    data = dict(
        z_ternary    = z_ternary,
        T_exp        = T_exp,
        rho_solvent  = rho_solvent,
        dielec_solvent = dielec_solvent,
        xE_exp       = xE_exp,
        xR_exp       = xR_exp,
    )

    print(f"\n{'='*60}")
    print(f"LIQUAC fit:  {SALT} / {SOLVENT}   ({z_exp.shape[0]} data points)")
    print(f"Method:      {method}")
    print(f"Workers:     {N_WORKERS}")
    print(f"{'='*60}\n")

    # ── optimise ───────────────────────────────────────────────────────────────
    bounds = [(BOUND_LO, BOUND_HI)] * 4

    scalar_obj = lambda p: _objective_scalar(
        p, data, ip_Gmehling, IP_UNIQUAC, r, q, MW, valency,
        SALT, COARSEGRAIN, N_WORKERS
    )

    if method == "nelder-mead":
        result = minimize(scalar_obj, IP_GUESS, method="Nelder-Mead",
                          options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6,
                                   "disp": True})
        IP_fitted = result.x
        rms_final = result.fun

    elif method == "dual-annealing":
        result = dual_annealing(scalar_obj, bounds, x0=IP_GUESS,
                                maxiter=1000, seed=42)
        IP_fitted = result.x
        rms_final = result.fun

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nelder-mead' or 'dual-annealing'.")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Optimisation complete  ({elapsed:.1f} s)")
    print(f"Final RMS    : {rms_final:.6f}")
    print(f"Fitted params: {IP_fitted}")
    print(f"{'='*60}\n")

    # ── evaluate once at fitted params to get xE, xR, D ───────────────────────
    _, xE_pred, xR_pred, D_vals = objective(
        IP_fitted, data, ip_Gmehling, IP_UNIQUAC,
        r, q, MW, valency, SALT, COARSEGRAIN, N_WORKERS
    )

    # ── save results ───────────────────────────────────────────────────────────
    out_dir = Path(DATA_DIR) / f"{SALT}-{SOLVENT}"

    # D values (for use in d_resolution.py)
    pd.DataFrame(D_vals).to_csv(out_dir / "D_values.csv", index=False, header=False)

    # Predicted vs experimental compositions
    cols = ["solvent", "water", "cation", "anion"]
    df_xE = pd.DataFrame(xE_pred, columns=[f"xE_{c}" for c in cols])
    df_xR = pd.DataFrame(xR_pred, columns=[f"xR_{c}" for c in cols])
    df_xE_exp = pd.DataFrame(xE_exp, columns=[f"xE_exp_{c}" for c in cols])
    df_xR_exp = pd.DataFrame(xR_exp, columns=[f"xR_exp_{c}" for c in cols])
    results = pd.concat([df_xE_exp, df_xR_exp, df_xE, df_xR], axis=1)
    results["RMS_contribution"] = (
        ((xE_exp - xE_pred)**2).sum(axis=1)
      + ((xR_exp - xR_pred)**2).sum(axis=1)
    ) * 1000
    results.to_csv(out_dir / "results_liquac.csv", index=False)

    # Fitted parameters
    param_names = ["b_solv_cation", "b_solv_anion", "c_solv_cation", "c_solv_anion"]
    pd.DataFrame({"parameter": param_names, "value": IP_fitted}).to_csv(
        out_dir / "fitted_params.csv", index=False
    )

    print(f"Results saved to: {out_dir}")
    return IP_fitted, rms_final, xE_pred, xR_pred, D_vals


if __name__ == "__main__":
    # ── choose method here ────────────────────────────────────────────────────
    # "nelder-mead"    → fast local refinement (start from a good IP_GUESS)
    # "dual-annealing" → slow global search    (use when IP_GUESS is uncertain)
    run_fit(method="nelder-mead")