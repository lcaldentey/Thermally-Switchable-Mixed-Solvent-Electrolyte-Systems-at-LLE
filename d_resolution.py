"""
d_resolution.py
===============
Tie-line refinement script — equivalent to MATLAB DResolution.m.

PURPOSE
-------
After running liquac_fit.py you have coarse estimates of the tie-line
slopes (D_values.csv).  This script takes those as starting guesses and
uses scipy.optimize.minimize (Nelder-Mead) to find the D value that
maximises the Eubanks area for each data point individually, giving
higher-precision xE and xR.

WORKFLOW
--------
1. Run liquac_fit.py first to produce D_values.csv.
2. Edit the USER SETTINGS block below.
3. Run:
       python d_resolution.py
4. Results are saved to:
       <data_dir>/<salt>-<solvent>/results_Dresolution.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from liquac_inputs       import LIQUACInputs
from LongRange          import LongRange
from MediumRange        import MediumRange
from ShortRange         import ShortRange
from gibbs_minimization  import GibbsMinimization, _build_scan, _compute_ln_gamma, _dg_mix, _eubanks_area


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SALT    = "NaCl"
SOLVENT = "DIPA"
DATA_DIR = "/Users/lucascaldentey/Desktop/Yip Lab"

# UNIQUAC parameters (same as used during fitting — hold fixed)
IP_UNIQUAC = np.array([-2.13881699866569, -2.13155946322135,
                        541.616375907126,  587.067085535208])

# Fitted LIQUAC interaction parameters from liquac_fit.py
INTERACTION_PARAMS = np.array([-1.49245819676001,  1.99466864109731,
                                 0.52614759793941,   0.70575476122774])

# ═══════════════════════════════════════════════════════════════════════════════


def _find_D_objective(D_scalar, z_ternary, T, rho_s, diel_s,
                      b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC,
                      r, q, MW, valency, salt,
                      objLR, objMR, objSR):
    """
    Returns the *negative* Eubanks area for a given D (so minimising this
    maximises the area).  Returns 1e10 if the scan is degenerate.

    Equivalent to the MATLAB `findD` nested function in DResolution.m.
    """
    D = float(D_scalar)
    sigfigs = 5

    scan = _build_scan(z_ternary, D, sigfigs)
    if scan is None:
        return 1e10

    x_NaCl, node_lengths, x_tot = scan

    ln_gamma = _compute_ln_gamma(
        x_tot, T, rho_s, diel_s,
        MW, valency, r, q,
        b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC,
        salt, objLR, objMR, objSR,
    )
    dg = _dg_mix(x_tot, ln_gamma)

    if not np.all(np.isreal(dg)):
        return 1e10

    D_scale = np.sqrt(1 + D**2)
    max_area, idx_E, idx_R = _eubanks_area(dg, x_NaCl, node_lengths, D_scale)

    if idx_E < 0:
        return 1e10

    return -max_area   # negate: scipy minimises, we want to maximise area


def run_resolution():
    # ── load data ──────────────────────────────────────────────────────────────
    loader = LIQUACInputs(DATA_DIR)
    r, q, MW, valency = loader.species_data(SALT, SOLVENT)
    (z_exp, xE_exp, xR_exp, T_exp,
     rho_solvent, dielec_solvent,
     _, ip_Gmehling) = loader.experimental_data(SALT, SOLVENT)

    z_ternary = loader.to_ternary(z_exp)
    N = z_exp.shape[0]

    # ── load D starting guesses ───────────────────────────────────────────────
    d_path = Path(DATA_DIR) / f"{SALT}-{SOLVENT}" / "D_values.csv"
    if not d_path.exists():
        raise FileNotFoundError(
            f"D_values.csv not found at {d_path}.\n"
            "Run liquac_fit.py first to generate it."
        )
    D_guess = pd.read_csv(d_path, header=None).values.ravel()

    # ── assemble full interaction parameter vector ─────────────────────────────
    b_ij   = np.concatenate([INTERACTION_PARAMS[0:2], ip_Gmehling[0:2]])
    c_ij   = np.concatenate([INTERACTION_PARAMS[2:4], ip_Gmehling[2:4]])
    b_jcja = ip_Gmehling[4]
    c_jcja = ip_Gmehling[5]
    ip_full = np.concatenate([b_ij, c_ij, [b_jcja, c_jcja], IP_UNIQUAC])

    # ── initialise objects ────────────────────────────────────────────────────
    objLR = LongRange()
    objMR = MediumRange()
    objSR = ShortRange()
    objG  = GibbsMinimization()

    # ── refine each data point ────────────────────────────────────────────────
    D_refined = np.zeros(N)
    xE_pred   = np.zeros((N, 4))
    xR_pred   = np.zeros((N, 4))

    print(f"\n{'='*60}")
    print(f"D-resolution:  {SALT} / {SOLVENT}   ({N} data points)")
    print(f"{'='*60}\n")

    for i in range(N):
        z_t   = z_ternary[i]
        T_i   = T_exp[i]
        rho_i = rho_solvent[i]
        die_i = dielec_solvent[i]

        obj_fn = lambda D_arr: _find_D_objective(
            D_arr[0], z_t, T_i, rho_i, die_i,
            b_ij, c_ij, b_jcja, c_jcja, IP_UNIQUAC,
            r, q, MW, valency, SALT, objLR, objMR, objSR
        )

        res = minimize(
            obj_fn,
            x0  = [D_guess[i]],
            method = "Nelder-Mead",
            options = {"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
        )

        D_refined[i] = res.x[0]

        xE_pred[i], xR_pred[i] = objG.gibbs_liquac_with_D(
            z_t, r, q, T_i, ip_full, MW, valency,
            rho_i, die_i, SALT, objMR, objSR, objLR, D_refined[i]
        )

        print(f"  [{i+1:3d}/{N}]  D_guess={D_guess[i]:.4f}  "
              f"D_refined={D_refined[i]:.6f}  "
              f"f={res.fun:.6e}  converged={res.success}")

    # ── ternary normalised compositions ──────────────────────────────────────
    xE_ternary = xE_pred[:, 0:3] / xE_pred[:, 0:3].sum(axis=1, keepdims=True)
    xR_ternary = xR_pred[:, 0:3] / xR_pred[:, 0:3].sum(axis=1, keepdims=True)

    # ── save results ─────────────────────────────────────────────────────────
    out_dir = Path(DATA_DIR) / f"{SALT}-{SOLVENT}"
    cols_4   = ["solvent", "water", "cation", "anion"]
    cols_3   = ["solvent", "water", "salt"]

    df = pd.DataFrame({
        **{f"xE_{c}":         xE_pred[:, j]    for j, c in enumerate(cols_4)},
        **{f"xR_{c}":         xR_pred[:, j]    for j, c in enumerate(cols_4)},
        **{f"xE_tern_{c}":    xE_ternary[:, j] for j, c in enumerate(cols_3)},
        **{f"xR_tern_{c}":    xR_ternary[:, j] for j, c in enumerate(cols_3)},
        "D_guess":   D_guess,
        "D_refined": D_refined,
    })
    out_path = out_dir / "results_Dresolution.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return D_refined, xE_pred, xR_pred


if __name__ == "__main__":
    run_resolution()