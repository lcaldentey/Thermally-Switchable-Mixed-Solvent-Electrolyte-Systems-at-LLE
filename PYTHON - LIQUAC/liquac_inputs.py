"""
liquac_inputs.py
================
Data loader — equivalent to LIQUACinputs.m and UNIQUACinputs.m.

All input Excel files are expected in the folder:
    <data_dir>/<salt>-<solvent>/   e.g.  ~/Desktop/Yip Lab/NaCl-DIPA/

Each file is a single-sheet xlsx.  The first row may optionally contain
column labels (e.g. "DIPA", "H2O", "Na+", "Cl-") — these are detected
automatically and skipped.  Pure-number files also work without any changes.

File list
---------
r.xlsx        van-der-Waals relative volumes  [solvent, water, cation, anion]
q.xlsx        van-der-Waals relative surfaces [solvent, water, cation, anion]
MW.xlsx       molecular weights in kg/mol     [solvent, water, cation, anion]
valency.xlsx  ionic charges (0 for solvents)  [solvent, water, cation, anion]
z_exp.xlsx    feed mole fractions             [solvent, water, cation, anion]  (N rows)
xE_exp.xlsx   organic-phase mole fractions    [solvent, water, cation, anion]  (N rows)
xR_exp.xlsx   aqueous-phase mole fractions    [solvent, water, cation, anion]  (N rows)
T_exp.xlsx    temperature in Kelvin           (N rows, 1 column)

For the UNIQUAC-only fit (UNIQUACinputs equivalent) the data directory is:
    <data_dir>/<solvent>/          e.g.  ~/Desktop/Yip Lab/DIPA/
with the same file set minus MW.xlsx and valency.xlsx.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ── Gmehling group water-ion interaction parameters for common salts ──────────
# Format: [b_H2O-cation, b_H2O-anion, c_H2O-cation, c_H2O-anion, b_jcja, c_jcja]
GMEHLING_PARAMS = {
    "NaCl": np.array([ 0.00331, -0.00128, -0.00143, -0.00020,  0.17219, -0.26495]),
    "LiCl": np.array([ 0.00319, -0.00128, -0.00099, -0.00020,  0.37690, -0.36090]),
    "KCl":  np.array([ 0.02580, -0.00128, -0.00088, -0.00020,  0.09387, -0.19630]),
    "KBr":  np.array([ 0.02580, -0.00247, -0.00088, -0.00008,  0.11020, -0.15500]),
    "NaBr": np.array([ 0.00331, -0.00247,  0.00143, -0.00008,  0.21660, -0.22130]),
}


# ── Excel reading helpers ─────────────────────────────────────────────────────

def _read_excel_numeric(path: Path) -> pd.DataFrame:
    """
    Read an xlsx file and return a DataFrame containing only numeric rows.

    If the first row contains any string values (column labels such as
    "DIPA", "H2O", "Na+", "Cl-", "T", etc.) that row is silently skipped.
    All other rows are returned as float64.  This means you never need to
    manually remove header rows from your Excel files.
    """
    df = pd.read_excel(path, header=None)

    # Drop any rows where every cell is a string (header / label rows)
    def _row_is_header(row):
        return row.apply(lambda v: isinstance(v, str)).any()

    mask = df.apply(_row_is_header, axis=1)
    df = df[~mask].reset_index(drop=True)

    return df.astype(float)


def _read(path: Path) -> np.ndarray:
    """Read a single Excel file to a flat 1-D numpy array (skips header if present)."""
    return _read_excel_numeric(path).values.ravel()


# ── Main data loader ──────────────────────────────────────────────────────────

class LIQUACInputs:
    """Load all data needed to run the LIQUAC model."""

    def __init__(self, data_dir: str | Path):
        """
        Parameters
        ----------
        data_dir : str or Path
            Root directory that contains the <salt>-<solvent> sub-folder.
            e.g. "/Users/you/Desktop/Yip Lab"
        """
        self.data_dir = Path(data_dir)

    def _folder(self, salt: str, solvent: str) -> Path:
        folder = self.data_dir / f"{salt}-{solvent}"
        if not folder.exists():
            raise FileNotFoundError(
                f"Data folder not found: {folder}\n"
                f"Expected structure: {folder}/<r.xlsx, q.xlsx, ...>"
            )
        return folder

    # ── species physical data ─────────────────────────────────────────────────

    def species_data(self, salt: str, solvent: str):
        """
        Returns
        -------
        r, q, MW, valency : np.ndarray, shape (4,)
            [solvent, water, cation, anion]
        """
        folder = self._folder(salt, solvent)
        r       = _read(folder / "r.xlsx")
        q       = _read(folder / "q.xlsx")
        MW      = _read(folder / "MW.xlsx")
        valency = _read(folder / "valency.xlsx")
        return r, q, MW, valency

    # ── experimental data ─────────────────────────────────────────────────────

    def experimental_data(self, salt: str, solvent: str):
        """
        Returns
        -------
        z_exp          : np.ndarray (N, 4)  feed compositions
        xE_exp         : np.ndarray (N, 4)  organic-phase compositions
        xR_exp         : np.ndarray (N, 4)  aqueous-phase compositions
        T_exp          : np.ndarray (N,)    temperatures [K]
        rho_solvent    : np.ndarray (N,)    solvent densities [kg/m³]
        dielec_solvent : np.ndarray (N,)    solvent dielectric constants
        Selec_exp      : np.ndarray (N, 2)  selectivities
        ip_Gmehling    : np.ndarray (6,)    Gmehling water-ion interaction params
        """
        folder = self._folder(salt, solvent)

        z_exp  = _read_excel_numeric(folder / "z_exp.xlsx").values
        xE_exp = _read_excel_numeric(folder / "xE_exp.xlsx").values
        xR_exp = _read_excel_numeric(folder / "xR_exp.xlsx").values
        T_exp  = _read(folder / "T_exp.xlsx")   # shape (N,)

        # ── solvent physical properties as a function of T ────────────────────
        rho_solvent, dielec_solvent = self._solvent_properties(solvent, T_exp)

        # ── Gmehling parameters ───────────────────────────────────────────────
        if salt not in GMEHLING_PARAMS:
            raise ValueError(
                f"Gmehling parameters for '{salt}' not available. "
                f"Available salts: {list(GMEHLING_PARAMS.keys())}"
            )
        ip_Gmehling = GMEHLING_PARAMS[salt]

        # ── selectivity ───────────────────────────────────────────────────────
        # Defined as x_H2O,org / x_ion,org / 1000
        Selec_exp = np.column_stack([
            xE_exp[:, 1] / xE_exp[:, 2] / 1000,
            xE_exp[:, 1] / xE_exp[:, 3] / 1000,
        ])

        return (z_exp, xE_exp, xR_exp, T_exp,
                rho_solvent, dielec_solvent, Selec_exp, ip_Gmehling)

    # ── solvent property correlations ─────────────────────────────────────────

    @staticmethod
    def _solvent_properties(solvent: str, T_exp: np.ndarray):
        """
        Return (rho_solvent, dielec_solvent) as arrays matching T_exp.
        Add new solvents here as correlations become available.
        """
        if "DIPA" in solvent:
            # Density [kg/m³]:      https://doi.org/10.1016/j.jct.2018.12.012
            rho_solvent    = -0.9675 * T_exp + 1003.6
            # Dielectric constant:  experimentally determined by Eliza Dach
            dielec_solvent = -7.32e-3 * (T_exp - 273.15) + 3.24
        else:
            raise ValueError(
                f"Solvent property correlations for '{solvent}' are not yet "
                "implemented.  Please add them to LIQUACInputs._solvent_properties()."
            )
        return rho_solvent, dielec_solvent

    # ── ternary feed compositions (convenience) ───────────────────────────────

    @staticmethod
    def to_ternary(z_exp: np.ndarray) -> np.ndarray:
        """
        Normalise feed compositions to ternary (salt-free + salt),
        matching the MATLAB z_exp_ternary calculation.
        Returns array of shape (N, 3): [x_solvent, x_water, x_salt].
        """
        total = z_exp[:, 0:3].sum(axis=1, keepdims=True)
        ternary = z_exp[:, 0:3] / total
        return np.round(ternary, 5)


# ── UNIQUAC-only data loader (binary solvent-water, no salt) ──────────────────

class UNIQUACInputs:
    """Load data for the UNIQUAC-only (binary solvent-water) fit."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def species_data(self, solvent: str):
        """
        Returns
        -------
        r, q           : np.ndarray (2,)  [solvent, water]
        T_exp          : np.ndarray (N,)
        xE_exp         : np.ndarray (N, 2)
        xR_exp         : np.ndarray (N, 2)
        """
        folder = self.data_dir / solvent
        if not folder.exists():
            raise FileNotFoundError(f"Data folder not found: {folder}")

        r      = _read(folder / "r.xlsx")
        q      = _read(folder / "q.xlsx")
        T_exp  = _read(folder / "T_exp.xlsx")
        xE_exp = _read_excel_numeric(folder / "xE_exp.xlsx").values
        xR_exp = _read_excel_numeric(folder / "xR_exp.xlsx").values

        return r, q, T_exp, xE_exp, xR_exp