% GLOBAL VARIABLES
% You need to instantiate certain variables as global so they can be used
% and won't change as the algorithmic solver is running
global solvent r q T_exp xE_exp xR_exp obj_UNIQUAC obj_Flash z_exp


% Instantiates objects to be used
obj_inputs = UNIQUACinputs;
obj_UNIQUAC = UNIQUAC;
obj_Flash = LLEFlash;

% INPUT VALUES -- USER SUPPLIED
% Note: the purpose of this is to pull relevant data needed to run the
% UNIQUAC model from user-supplied files in the format:
% "/Users/username/Desktop/solvent/parameters.xlsx"
% See UNQUACinputs.m for additional details
solvent = "DIPA";


% Pulls relevant data using the input solvent above
[r, q, T_exp, xE_exp, xR_exp] = obj_inputs.speciesData(solvent);
z_exp = 0.5 * (xE_exp + xR_exp);


% Interaction parameters defined
ip_final = [-2.16204963864155,	-2.11041103797793,	534.584874928107,	590.112865901132];
ip_final_w_fmin = [-2.13881699866569,	-2.13155946322135,	541.616375907126,	587.067085535208];
ip_reproduced = [-1.52162345228748,	-2.36628685616026,	311.719039351369,	685.750998449254];
ip_reproduced_w_fmin = [-1.52162345228748,	-2.36628685616026,	311.719039351369,	685.750998449254];


% Establish the initial guess for the interaction parameters (ip_guess),
% the lower bound and upper bound limits on the interaction parameters (lb,
% ub)
ip_guess = ip_final_w_fmin;
lb = [-10, -10, -10000, -10000];
ub = lb * -1;


% ALGORITHMIC PARAMETERS
% Set interaction parameters to certain values (in this case ip_guess)
% Fitting algorithms: simulatedannealbnd for global maxima search,
% fminsearch for fine-tuning
interaction_parameters = ip_guess;
[IP, RMS, exitflag, output] = simulannealbnd(@convergence, ip_guess, lb, ub);
%[IP, f_val, exitflag, output] = fminsearch(@convergence, interaction_parameters);

function RMS = convergence(interaction_parameters)
   
    global r q T_exp xE_exp xR_exp obj_UNIQUAC obj_Flash z_exp

    xE = zeros(size(xE_exp,1),size(xE_exp,2));
    xR = zeros(size(xR_exp,1),size(xR_exp,2));

    for i = 1:size(xE_exp,1)
        % Resolve xE and xR using the Flash algorithm
        %[xE(i,:), xR(i,:)] = obj_Flash.Flash_UNIQUAC(z_exp(i,:), q, T_exp(i,:), interaction_parameters);
        
        % Resolve xE and xR using the Eubanks algorithm (better for LLE)
        [xE(i,:), xR(i,:)] = obj_Flash.Gibbs_UNIQUAC(r, q, T_exp(i,:), interaction_parameters);

    end

    RMS = sum(sum(((xE_exp - xE)./min(xE_exp, xE)).^2)) + sum(sum(((xR_exp - xR)./min(xE_exp, xE)).^2))  

    
    ln_gamma_E = obj_UNIQUAC.uniquac_calc(xE, r, q, T_exp, interaction_parameters);
    %[ln_gamma_E_comb, ln_gamma_E_resid] = obj_UNIQUAC.uniquac_detailed_calc(xE, r, q, T_exp, interaction_parameters);
    ln_gamma_R = obj_UNIQUAC.uniquac_calc(xR, r, q, T_exp, interaction_parameters);
    %[ln_gamma_R_comb, ln_gamma_R_resid] = obj_UNIQUAC.uniquac_detailed_calc(xR, r, q, T_exp, interaction_parameters);
    isoactivity = sum(sum((xE .* exp(ln_gamma_E) - xR .* exp(ln_gamma_R)).^2));
    

end