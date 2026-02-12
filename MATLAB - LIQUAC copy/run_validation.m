% run_validation.m
% ================
% Run this script from inside your MATLAB - LIQUAC copy folder.
% It prints the reference values for all 6 Python validation tests.
% Copy-paste the output numbers into validate.py as instructed.
%
% HOW TO RUN:
%   1. Open MATLAB
%   2. In the MATLAB file browser, navigate to your "MATLAB - LIQUAC copy" folder
%   3. Either:
%        a) Open this file and press the green Run button, OR
%        b) Type this in the MATLAB Command Window:
%              cd '/Users/lucascaldentey/Desktop/Yip Lab/MATLAB - LIQUAC copy'
%              run_validation
%
% WHAT TO DO WITH THE OUTPUT:
%   Each test prints a line like:
%       TEST1_LR = [0.2403  0.0326  -4.5801  -4.5801]
%   Find the matching line in validate.py and replace "expected = None" with
%   the printed numbers.  Full instructions are printed at the end.


% ── Shared inputs (identical to validate.py) ─────────────────────────────────
MW      = [0.10117, 0.018015, 0.022990, 0.035453];
valency = [0, 0, 1, -1];
r       = [5.2742, 0.9200, 0.5010, 1.3950];
q       = [4.4360, 1.4000, 0.4290, 1.3000];
T       = 298.15;
rho_s   = 758.0;
diel_s  = 3.10;
salt    = "NaCl";
x       = [0.40, 0.40, 0.10, 0.10];

b_ij    = [-1.49246,  1.99467,  0.00331, -0.00128];
c_ij    = [ 0.52615,  0.70575, -0.00143, -0.00020];
b_jcja  = 0.17219;
c_jcja  = -0.26495;
ip_UNIQUAC = [-2.13882, -2.13156, 541.616, 587.067];
ip      = [b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC];

% ── Instantiate objects ───────────────────────────────────────────────────────
objLR    = LongRange;
objMR    = MediumRange;
objSR    = ShortRange;
objUQ    = UNIQUAC;
objGibbs = GibbsMinimization;

fprintf('\n');
fprintf('=============================================================\n');
fprintf('LIQUAC VALIDATION — MATLAB REFERENCE VALUES\n');
fprintf('=============================================================\n\n');


% ── TEST 1: LongRange ─────────────────────────────────────────────────────────
fprintf('--- TEST 1: LongRange.func_LR ---\n');
result1 = objLR.func_LR(MW, valency, x, T, rho_s, diel_s, salt);
fprintf('MATLAB output:\n');
disp(result1);
fprintf('Python output (for comparison):\n');
fprintf('  [ 0.24033365  0.03259550 -4.58011508 -4.58011508 ]\n\n');


% ── TEST 2: MediumRange ──────────────────────────────────────────────────────
fprintf('--- TEST 2: MediumRange.func_MR ---\n');
result2 = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt);
fprintf('MATLAB output:\n');
disp(result2);
fprintf('Python output (for comparison):\n');
fprintf('  [ 0.43954732 -0.24278855 -11.58843330 18.26851439 ]\n\n');


% ── TEST 3: ShortRange ───────────────────────────────────────────────────────
fprintf('--- TEST 3: ShortRange.func_SR ---\n');
result3 = objSR.func_SR(x, r, q, T, ip_UNIQUAC, MW);
fprintf('MATLAB output:\n');
disp(result3);
fprintf('Python output (for comparison):\n');
fprintf('  [ -0.01174565  0.63398087 -0.37451688 -0.30887280 ]\n\n');


% ── TEST 4: UNIQUAC binary ───────────────────────────────────────────────────
fprintf('--- TEST 4: UNIQUAC.uniquac_calc (binary, no ions) ---\n');
r_bin = r(1:2);
q_bin = q(1:2);
x_bin = [0.40, 0.60];
result4 = objUQ.uniquac_calc(x_bin, r_bin, q_bin, T, ip_UNIQUAC);
fprintf('MATLAB output:\n');
disp(result4);
fprintf('Python output (for comparison):\n');
fprintf('  [ 0.20548065  0.46270858 ]\n\n');


% ── TEST 5: Full LIQUAC ln(gamma) ────────────────────────────────────────────
fprintf('--- TEST 5: Full LIQUAC ln(gamma) — LR + MR + SR combined ---\n');
ln_g_LR = objLR.func_LR(MW, valency, x, T, rho_s, diel_s, salt);
ln_g_MR = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt);
ln_g_SR = objSR.func_SR(x, r, q, T, ip_UNIQUAC, MW);
ln_g    = ln_g_LR + ln_g_MR + ln_g_SR;
vp = abs(valency(3));
vm = abs(valency(4));
mean_ion = (vp * ln_g(:,3) + vm * ln_g(:,4)) / (vp + vm);
ln_g(:,3) = mean_ion;
ln_g(:,4) = mean_ion;
fprintf('MATLAB output:\n');
disp(ln_g);
fprintf('Python output (for comparison):\n');
fprintf('  [ 0.66813532  0.42378782 -1.58176937 -1.58176937 ]\n\n');


% ── TEST 6: GibbsMinimization with D ─────────────────────────────────────────
fprintf('--- TEST 6: GibbsMinimization.Gibbs_LIQUAC_withD (D=5) ---\n');
z_ternary = [0.4, 0.5, 0.1];
D = 5.0;
[xE, xR] = objGibbs.Gibbs_LIQUAC_withD(z_ternary, r, q, T, ip, MW, valency, ...
               rho_s, diel_s, salt, objMR, objSR, objLR, D);
fprintf('MATLAB xE:\n');
disp(xE);
fprintf('MATLAB xR:\n');
disp(xR);
fprintf('Python xE (for comparison):\n');
fprintf('  [ 0.52579078  0.33871554  0.06774684  0.06774684 ]\n');
fprintf('Python xR (for comparison):\n');
fprintf('  [ 5.14289e-05  0.71424408  0.14285224  0.14285224 ]\n\n');


% ── Summary / instructions ────────────────────────────────────────────────────
fprintf('=============================================================\n');
fprintf('DONE. Now open validate.py in your Python folder.\n');
fprintf('\n');
fprintf('For each test, find the line that says:\n');
fprintf('    expected = None\n');
fprintf('and replace it with the MATLAB output numbers above.\n');
fprintf('\n');
fprintf('Example — if TEST 1 MATLAB output was [0.2403 0.0326 -4.5801 -4.5801]:\n');
fprintf('    expected = np.array([0.2403, 0.0326, -4.5801, -4.5801])\n');
fprintf('\n');
fprintf('There are 5 "expected = None" lines to fill in:\n');
fprintf('  test_long_range()    -> replace expected = None\n');
fprintf('  test_medium_range()  -> replace expected = None\n');
fprintf('  test_short_range()   -> replace expected = None\n');
fprintf('  test_uniquac()       -> replace expected = None\n');
fprintf('  test_full_ln_gamma() -> replace expected = None\n');
fprintf('  test_gibbs_with_D()  -> replace expected_xE = None AND expected_xR = None\n');
fprintf('\n');
fprintf('Then in your Python terminal:\n');
fprintf('    python validate.py\n');
fprintf('All tests should show PASS.\n');
fprintf('=============================================================\n');