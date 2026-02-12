% The purpose of this script is to give you an exact tie line, xE, and xR.
% The script can be very slow and should only be used after you have guesses at
% D from LIQUAC.m


% GLOBAL VARIABLES
% You need to instantiate certain variables as global so that they remain
% constant and can be used in the function below.
global r q MW valency z_exp_total xE_exp xR_exp T_exp_total ...
    rho_solvent_total dielec_solvent_total z_exp_ternary_total D_values ...    
    ip_UNIQUAC objLR objMR objSR objFlash salt ...
    ip_Gmehling b_ij c_ij b_jcja c_jcja ...
    z_exp_ternary T_exp rho_solvent dielec_solvent



% INPUT VALUES -- USER-SUPPLIED
% All relevant data needs to be saved in the following file format:
% "/Users/username/Desktop/salt-solvent/parameters.xlsx"
% See LIQUACinputs.m for additional details
salt = "NaCl";
solvent = "DIPA";
username = "kinnarishah";



% INPUT VALUES -- GRABBED FROM USER-SUPPLIED VIA CSV FILES
% See LIQUACinputs.m for additional details on each variable
obj = LIQUACinputs;
[r, q, MW, valency] = obj.speciesData(salt, solvent, username);
[z_exp_total, xE_exp, xR_exp, T_exp_total, rho_solvent_total, dielec_solvent_total, ...
    Selec_exp, ip_Gmehling] = obj.experimentalData(salt, solvent, username);
z_exp_ternary_total = [z_exp_total(:,1) ./ sum(z_exp_total(:,1:3),2), z_exp_total(:,2) ./ sum(z_exp_total(:,1:3),2), z_exp_total(:,3) ./ sum(z_exp_total(:,1:3),2)];
z_exp_ternary_total = round(z_exp_ternary_total,5);



% INPUT VALUES -- USER SUPPLIED
% This script runs on the basis that you have a good guess for the tie line
% slopes based on your interaction paramters. You must save the D_values 
% from LIQUAC.m and supply them here to make this script work.
D_values = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/D_values.csv"));
D_values = D_values{:,:};



% UNIQUAC PARAMETERS FOR SOLVENT-WATER
% DIPA-H2O values are currently supplied; use UNIQUAC_Gibbsfits.m if you
% want to determine the UNIQUAC parameters for a different solvent-water
% pair. Note that it is most robust to solve for these separately and hold
% them constant during LIQUAC fitting
ip_UNIQUAC = [-2.13881699866569,	-2.13155946322135,	541.616375907126,	587.067085535208];



% OBJECT INITIALIZATION
% Initializes objects for each component of LIQUAC: Long, Medium, and Short
% Range. Also initializes GibbsMinimization.m which solves for equilibrium
% mole fractions and tie line slopes
objLR = LongRange;
objMR = MediumRange;
objSR = ShortRange; 
objFlash = GibbsMinimization;



% LIQUAC INTERACTION PARAMETERS TO BE FITTED: [b_ij, c_ij]
% Look to Gmehling group papers for good starting guesses
interaction_params = [-1.49245819676001,	1.99466864109731,	0.526147597939412,	0.705754761227743];



% Holds interaction parameters constant. The purpose of this script is to
% find the equilibrium mole fractions and tie line slopes, not to iterate
% on the interaction parameters.
b_ij = [interaction_params(1:2), ip_Gmehling(1:2)];
c_ij = [interaction_params(3:4), ip_Gmehling(3:4)];
b_jcja = ip_Gmehling(5);
c_jcja = ip_Gmehling(6);
ip = [b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC];


% Initializes xE, xR, and tie line slopes (D_refined) all of which will be
% solved for
xE = zeros(size(z_exp_total,1), size(z_exp_total,2));
xR = zeros(size(z_exp_total,1), size(z_exp_total,2));
D_refined = zeros(size(z_exp_total,1),1);


% Solves for xE, xR, D_refined
for i = 1:size(z_exp_total,1)

    % z denotes feed mole fractions
    z_exp_ternary = z_exp_ternary_total(i,:);
    T_exp = T_exp_total(i,:);
    rho_solvent = rho_solvent_total(i,:);
    dielec_solvent = dielec_solvent_total(i,:);

    [D_refined(i), f_val2, exitflag2, output2] = fminsearch(@findD, D_values(i));
    [xE(i,:), xR(i,:)] = objFlash.Gibbs_LIQUAC_withD(z_exp_ternary, r, q, T_exp, ip, MW, valency, rho_solvent, dielec_solvent, salt, objMR, objSR, objLR, D_refined(i));
    
end



z_exp_ternary = z_exp_ternary_total;
T_exp = T_exp_total;
rho_solvent = rho_solvent_total;
dielec_solvent = dielec_solvent_total;

xE_ternary = xE(:,1:3) ./ sum(xE(:,1:3),2);
xR_ternary = xR(:,1:3) ./ sum(xR(:,1:3),2);



% Same function as is in GibbsMinimization.m; see that script for fully
% commented details
function max_area = findD(D)

    global r q MW valency T_exp rho_solvent dielec_solvent ...
        ip_UNIQUAC objLR objMR objSR salt z_exp_ternary ...
        b_ij c_ij b_jcja c_jcja

    sigfigs = 5;

    if D > 0
        x_NaCl_min = -(z_exp_ternary(2)/D - z_exp_ternary(3));
        x_NaCl_max = (D * z_exp_ternary(3) + 1 - z_exp_ternary(2)) / (1 + D);
    else
        x_NaCl_min = (D * z_exp_ternary(3) + 1 - z_exp_ternary(2)) / (1 + D);
        x_NaCl_max = z_exp_ternary(3) - z_exp_ternary(2) / D;
    end    
    
    if (x_NaCl_min <= 0)
        x_NaCl_min = 10^-sigfigs;
    end
    if (x_NaCl_max > 1)
        x_NaCl_max = 1;
    end
    x_NaCl_min = round(x_NaCl_min, sigfigs);
    x_NaCl_max = round(x_NaCl_max, sigfigs);
    
    x_NaCl = transpose(x_NaCl_min:(10^-sigfigs):(x_NaCl_max - (10^-sigfigs)));
    node_lengths = zeros(size(x_NaCl,1),1) + 10^-sigfigs;
    index = 1:size(x_NaCl,1);

    y_H2O = -D * (x_NaCl_max - x_NaCl) + (1 - x_NaCl_max);
    z_solv = 1 - x_NaCl - y_H2O;
    x_tot = [z_solv, y_H2O, x_NaCl, x_NaCl];
    x_tot = x_tot ./ sum(x_tot,2);

    temperature = zeros(size(x_NaCl,1),1) + T_exp;
    rho_solv = zeros(size(x_NaCl,1),1) + rho_solvent;
    dielec_solv = zeros(size(x_NaCl,1),1) + dielec_solvent;

    ln_gamma_LR = objLR.func_LR(MW, valency, x_tot, temperature, rho_solv, dielec_solv, salt);
    ln_gamma_MR = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x_tot, MW, valency, salt);
    ln_gamma_SR = objSR.func_SR(x_tot, r, q, temperature, ip_UNIQUAC, MW);
    ln_gamma = ln_gamma_LR + ln_gamma_MR + ln_gamma_SR; 
    ln_gamma(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,4), ...
                    abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,4)];
                
    
    dg_mix = x_tot(:,1) .* log(x_tot(:,1)) + x_tot(:,2) .* log(x_tot(:,2)) + ...
        x_tot(:,3) .* log(x_tot(:,3)) + x_tot(:,1) .* ln_gamma(:,1) + ...
        x_tot(:,2) .* ln_gamma(:,2) + x_tot(:,3) .* ln_gamma(:,3) + ...
        x_tot(:,4) .* log(x_tot(:,4)) + x_tot(:,4) .* ln_gamma(:,4);


    if isreal(dg_mix)
            
        m1 = (dg_mix(size(x_NaCl,1)) - dg_mix(1)) / (x_NaCl(size(x_NaCl,1)) - x_NaCl(1));
        b1 = dg_mix(1) - m1 * x_NaCl(1);
        dg_line = m1 * x_NaCl + b1;
        deltas = dg_mix - dg_line;
        indices = index(islocalmin(deltas));

        if (size(indices, 2) ~= 2)
            indices = index(islocalmin(dg_mix));
        end

        if size(indices,2) == 2

            guesses = indices(1:2);

            trap_area = abs(dg_mix(guesses(1)) + dg_mix(guesses(2))) * abs(x_NaCl(guesses(1)) - x_NaCl(guesses(2))) / 2;
            curve_area = abs(sum(dg_mix(guesses(1):guesses(2)) .* node_lengths((guesses(1):guesses(2)))));
            max_area = sqrt(1 + D^2) * (trap_area - curve_area);
    
                                
            for j = 1:(guesses(1) - 1)
    
                trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                difference = sqrt(1 + D^2) * (trap_area - curve_area);
    
                if difference > max_area
                    max_area = difference;
                    indices(1) = j;
                elseif difference < max_area
                    break
                end
            end
    
            for j = guesses(1):(guesses(2) - 1)
    
                trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                difference = sqrt(1 + D^2) * (trap_area - curve_area);
    
                if difference > max_area
                    max_area = difference;
                    indices(1) = j;
                elseif difference < max_area
                    break
                end
            end
            

            for k = (guesses(2) + 1):size(x_NaCl,1)
                
                trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                difference = sqrt(1 + D^2) * (trap_area - curve_area);
                
                if difference > max_area
                    max_area = difference;
                    indices(2) = k;
                elseif difference < max_area
                    break
                end
            end

            
            for k = (indices(1) + 1):(guesses(2) - 1)
                trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                difference = sqrt(1 + D^2) * (trap_area - curve_area);
                
                if difference > max_area
                    max_area = difference;
                    indices(2) = k;
                elseif difference < max_area
                    break
                end
            end
            max_area = -max_area;
        else
            max_area = 10^10;
        end
    else
        max_area = 10^10;
    end

end