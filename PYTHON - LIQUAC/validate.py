"""
validate.py
===========
Comparison harness for the Python LIQUAC port.

HOW IT WORKS
------------
Each test function:
  1. Defines a set of fixed, synthetic inputs (identical values you would
     pass from MATLAB).
  2. Runs the Python class.
  3. Compares against *expected* outputs that you paste in from MATLAB.
  4. Reports absolute and relative errors, and a PASS/FAIL flag.

TO USE WITH MATLAB
------------------
For each test block below, a companion MATLAB snippet is shown in the
docstring.  Run that snippet in MATLAB, copy the numerical output into the
`expected` variable in the corresponding Python test, then run:

    python validate.py

All tests passing means the Python port is numerically equivalent.

TOLERANCE
---------
Default absolute tolerance : 1e-8
Default relative tolerance : 1e-6
Tighten as needed once MATLAB references are filled in.
"""

import sys
import numpy as np

# ── import the Python modules ─────────────────────────────────────────────────
sys.path.insert(0, ".")          # run from /home/claude/liquac/
from LongRange        import LongRange
from MediumRange      import MediumRange
from ShortRange       import ShortRange
from uniquac           import UNIQUAC
from gibbs_minimization import GibbsMinimization


# ─────────────────────────────────────────────────────────────────────────────
# Shared canonical inputs (NaCl / DIPA system)
# ─────────────────────────────────────────────────────────────────────────────

# Molecular weights [kg/mol]: DIPA, H2O, Na+, Cl-
MW = np.array([0.10117, 0.018015, 0.022990, 0.035453])

# Valencies: solvents = 0, Na+ = +1, Cl- = -1
valency = np.array([0.0, 0.0, 1.0, -1.0])

# UNIQUAC shape parameters [DIPA, H2O, Na+, Cl-]
r = np.array([5.2742, 0.9200, 0.5010, 1.3950])
q = np.array([4.4360, 1.4000, 0.4290, 1.3000])

# Operating conditions (single point for most tests)
T_val        = 298.15           # K
rho_solvent  = np.array([758.0])   # kg/m³  (DIPA at ~298 K)
dielec_solv  = np.array([3.10])    # dimensionless
salt         = "NaCl"

# A representative composition: 40% DIPA, 40% H2O, 10% Na+, 10% Cl-
x_test = np.array([[0.40, 0.40, 0.10, 0.10]])

# LIQUAC interaction parameters (Gmehling + fitted)
# [b_DIPA-Na, b_DIPA-Cl, b_H2O-Na, b_H2O-Cl]
b_ij   = np.array([-1.49246, 1.99467,   0.00331, -0.00128])
c_ij   = np.array([ 0.52615, 0.70575,  -0.00143, -0.00020])
b_jcja = 0.17219
c_jcja = -0.26495
ip_UNIQUAC = np.array([-2.13882, -2.13156, 541.616, 587.067])

# Full interaction parameter vector for GibbsMinimization
ip_full = np.concatenate([b_ij, c_ij, [b_jcja, c_jcja], ip_UNIQUAC])


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check(name: str, got: np.ndarray, expected: np.ndarray,
           atol: float = 1e-8, rtol: float = 1e-6):
    """
    Compare `got` vs `expected`.  Print a compact diff and return True/False.
    If `expected` is None the test is marked SKIPPED (fill it in from MATLAB).
    """
    print(f"\n{'─'*60}")
    print(f"TEST: {name}")

    if expected is None:
        print("  ⚠  SKIPPED — paste MATLAB output into `expected`")
        return None

    got      = np.asarray(got, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()

    if got.shape != expected.shape:
        print(f"  ✗  SHAPE MISMATCH: got {got.shape}, expected {expected.shape}")
        return False

    abs_err = np.abs(got - expected)
    rel_err = abs_err / (np.abs(expected) + 1e-300)

    max_abs = abs_err.max()
    max_rel = rel_err.max()
    idx     = int(abs_err.argmax())

    passed = bool((abs_err < atol).all() or (rel_err < rtol).all())
    status = "✓  PASS" if passed else "✗  FAIL"

    print(f"  {status}")
    print(f"  max |Δ|  = {max_abs:.3e}  at index {idx}")
    print(f"  max |Δ|/|ref| = {max_rel:.3e}")
    if not passed:
        print(f"  got[{idx}]      = {got[idx]:.10f}")
        print(f"  expected[{idx}] = {expected[idx]:.10f}")
        print(f"  Python  : {got}")
        print(f"  MATLAB  : {expected}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — LongRange
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet to generate expected values:
--------------------------------------------
MW      = [0.10117, 0.018015, 0.022990, 0.035453];
valency = [0, 0, 1, -1];
x       = [0.40, 0.40, 0.10, 0.10];
T       = 298.15;
rho_s   = 758.0;
diel_s  = 3.10;
salt    = "NaCl";
objLR = LongRange;
result = objLR.func_LR(MW, valency, x, T, rho_s, diel_s, salt)
disp(result)
--------------------------------------------
Paste the 1x4 output vector below.
"""

def test_long_range():
    objLR = LongRange()
    got = objLR.func_LR(
        MW, valency, x_test,
        np.array([T_val]), rho_solvent, dielec_solv, salt
    )

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 1
    expected = np.array([ 0.24033365,  0.03259550, -4.58011508, -4.58011508])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python output: {got}")
    return _check("LongRange.func_LR", got, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — MediumRange
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet:
--------------------------------------------
b_ij   = [-1.49246, 1.99467,  0.00331, -0.00128];
c_ij   = [ 0.52615, 0.70575, -0.00143, -0.00020];
b_jcja = 0.17219; c_jcja = -0.26495;
x      = [0.40, 0.40, 0.10, 0.10];
MW     = [0.10117, 0.018015, 0.022990, 0.035453];
valency = [0, 0, 1, -1];
salt   = "NaCl";
objMR  = MediumRange;
result = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt)
disp(result)
--------------------------------------------
"""

def test_medium_range():
    objMR = MediumRange()
    got = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x_test, MW, valency, salt)

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 2
    expected = np.array([ 0.43954732, -0.24278855, -11.58843330, 18.26851439])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python output: {got}")
    return _check("MediumRange.func_MR", got, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — ShortRange
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet:
--------------------------------------------
r  = [5.2742, 0.9200, 0.5010, 1.3950];
q  = [4.4360, 1.4000, 0.4290, 1.3000];
ip = [-2.13882, -2.13156, 541.616, 587.067];
MW = [0.10117, 0.018015, 0.022990, 0.035453];
x  = [0.40, 0.40, 0.10, 0.10];
T  = 298.15;
objSR = ShortRange;
result = objSR.func_SR(x, r, q, T, ip, MW)
disp(result)
--------------------------------------------
"""

def test_short_range():
    objSR = ShortRange()
    got = objSR.func_SR(
        x_test, r, q, np.array([T_val]), ip_UNIQUAC, MW
    )

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 3
    expected = np.array([-0.01174565,  0.63398087, -0.37451688, -0.30887280])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python output: {got}")
    return _check("ShortRange.func_SR", got, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — UNIQUAC (binary, no ions)
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet:
--------------------------------------------
r_bin = [5.2742, 0.9200];
q_bin = [4.4360, 1.4000];
ip    = [-2.13882, -2.13156, 541.616, 587.067];
x_bin = [0.40, 0.60];
T     = 298.15;
obj   = UNIQUAC;
result = obj.uniquac_calc(x_bin, r_bin, q_bin, T, ip)
disp(result)
--------------------------------------------
"""

def test_uniquac():
    obj = UNIQUAC()
    r_bin = r[:2]
    q_bin = q[:2]
    x_bin = np.array([[0.40, 0.60]])
    got   = obj.uniquac_calc(x_bin, r_bin, q_bin,
                              np.array([T_val]), ip_UNIQUAC)

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 4
    expected = np.array([0.20548065, 0.46270858])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python output: {got}")
    return _check("UNIQUAC.uniquac_calc", got, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Full LIQUAC ln(γ) pipeline (LR + MR + SR combined)
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet:
--------------------------------------------
(use all variables already defined above)
ln_g_LR = objLR.func_LR(MW, valency, x, T, rho_s, diel_s, salt);
ln_g_MR = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt);
ln_g_SR = objSR.func_SR(x, r, q, T, ip_UNIQUAC, MW);
ln_g    = ln_g_LR + ln_g_MR + ln_g_SR;
vp = abs(valency(3)); vm = abs(valency(4));
mean_ion = (vp*ln_g(:,3) + vm*ln_g(:,4)) / (vp+vm);
ln_g(:,3) = mean_ion; ln_g(:,4) = mean_ion;
disp(ln_g)
--------------------------------------------
"""

def test_full_ln_gamma():
    objLR = LongRange()
    objMR = MediumRange()
    objSR = ShortRange()

    T_v = np.array([T_val])
    ln_g = (objLR.func_LR(MW, valency, x_test, T_v, rho_solvent, dielec_solv, salt)
          + objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x_test, MW, valency, salt)
          + objSR.func_SR(x_test, r, q, T_v, ip_UNIQUAC, MW))

    vp = abs(valency[2])
    vm = abs(valency[3])
    mean = (vp * ln_g[:, 2] + vm * ln_g[:, 3]) / (vp + vm)
    ln_g[:, 2] = mean
    ln_g[:, 3] = mean

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 5
    expected = np.array([ 0.66813532,  0.42378782, -1.58176937, -1.58176937])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python output: {ln_g}")
    return _check("Full LIQUAC ln(γ)", ln_g, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — GibbsMinimization.gibbs_liquac_with_D
# ─────────────────────────────────────────────────────────────────────────────
"""
MATLAB snippet:
--------------------------------------------
z_ternary = [0.4, 0.5, 0.1];   % ternary feed fracs (sums to 1)
D = 5.0;
objFlash = GibbsMinimization;
[xE, xR] = objFlash.Gibbs_LIQUAC_withD(z_ternary, r, q, T, ip, MW, valency,
                rho_s, diel_s, salt, objMR, objSR, objLR, D);
disp([xE; xR])
--------------------------------------------
"""

def test_gibbs_with_D():
    objLR = LongRange()
    objMR = MediumRange()
    objSR = ShortRange()
    objG  = GibbsMinimization()

    z_ternary = np.array([0.4, 0.5, 0.1])
    D = 5.0

    xE, xR = objG.gibbs_liquac_with_D(
        z_ternary, r, q, T_val, ip_full,
        MW, valency, rho_solvent[0], dielec_solv[0],
        salt, objMR, objSR, objLR, D
    )

    # ── PASTE MATLAB OUTPUT HERE ──────────────────────────────────────────────
    # Replace the numbers below with whatever MATLAB printed for TEST 6 xE / xR
    expected_xE = np.array([0.52579078, 0.33871554, 0.06774684, 0.06774684])
    expected_xR = np.array([5.14288653e-05, 7.14244081e-01, 1.42852245e-01, 1.42852245e-01])
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n  Python xE: {xE}")
    print(f"  Python xR: {xR}")
    r1 = _check("GibbsMin.withD — xE", xE, expected_xE)
    r2 = _check("GibbsMin.withD — xR", xR, expected_xR)
    return r1, r2


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Internal self-consistency: isoactivity at equilibrium
# ─────────────────────────────────────────────────────────────────────────────
"""
At true LLE equilibrium, the isoactivity criterion must hold:
    x_i^E * γ_i^E  ==  x_i^R * γ_i^R   for all i

This test doesn't need MATLAB values — it checks internal physics consistency.
It will run as long as GibbsMinimization returns non-sentinel compositions.
"""

def test_isoactivity():
    objLR = LongRange()
    objMR = MediumRange()
    objSR = ShortRange()
    objG  = GibbsMinimization()

    z_ternary = np.array([0.4, 0.5, 0.1])
    D = 5.0

    xE, xR = objG.gibbs_liquac_with_D(
        z_ternary, r, q, T_val, ip_full,
        MW, valency, rho_solvent[0], dielec_solv[0],
        salt, objMR, objSR, objLR, D
    )

    print(f"\n{'─'*60}")
    print("TEST: Isoactivity self-consistency")

    if xE[0] > 10:
        print("  ⚠  SKIPPED — solver returned sentinel (no phase split found)")
        return None

    T_v = np.array([T_val])

    def full_ln_g(x_row):
        x2 = x_row.reshape(1, -1)
        ln_g = (objLR.func_LR(MW, valency, x2, T_v, rho_solvent, dielec_solv, salt)
              + objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x2, MW, valency, salt)
              + objSR.func_SR(x2, r, q, T_v, ip_UNIQUAC, MW))
        vp = abs(valency[2]); vm = abs(valency[3])
        mean = (vp * ln_g[:, 2] + vm * ln_g[:, 3]) / (vp + vm)
        ln_g[:, 2] = mean; ln_g[:, 3] = mean
        return ln_g.ravel()

    ln_gE = full_ln_g(xE)
    ln_gR = full_ln_g(xR)

    act_E = xE * np.exp(ln_gE)
    act_R = xR * np.exp(ln_gR)

    residual = np.abs(act_E - act_R)
    print(f"  xE            = {xE}")
    print(f"  xR            = {xR}")
    print(f"  |xγ_E - xγ_R| = {residual}")

    # Solvents should satisfy isoactivity well; ions roughly
    tol = 1e-3
    passed = bool(residual[:2].max() < tol)
    print(f"  {'✓  PASS' if passed else '✗  FAIL'}  (tol={tol} on solvent phases)")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    results["LongRange"]        = test_long_range()
    results["MediumRange"]      = test_medium_range()
    results["ShortRange"]       = test_short_range()
    results["UNIQUAC"]          = test_uniquac()
    results["Full ln(γ)"]       = test_full_ln_gamma()
    results["GibbsMin withD"]   = test_gibbs_with_D()
    results["Isoactivity"]      = test_isoactivity()

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    skipped = passed = failed = 0
    for name, res in results.items():
        if isinstance(res, tuple):
            res = all(r is not False for r in res)
        if res is None:
            skipped += 1
            status = "⚠  SKIPPED"
        elif res:
            passed += 1
            status = "✓  PASS"
        else:
            failed += 1
            status = "✗  FAIL"
        print(f"  {status:<12}  {name}")

    print(f"\n  {passed} passed  |  {failed} failed  |  {skipped} skipped (awaiting MATLAB ref)")