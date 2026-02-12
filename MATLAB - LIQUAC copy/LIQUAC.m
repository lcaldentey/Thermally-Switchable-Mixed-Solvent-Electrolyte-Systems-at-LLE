% GLOBAL VARIABLES
% You need to instantiate certain variables as global so that they remain
% constant and can be used in the function below.
global r q MW valency z_exp xE_exp xR_exp T_exp rho_solvent dielec_solvent ...
    ip_UNIQUAC objLR objMR objSR objGibbs salt z_exp_ternary ...
    ip_Gmehling coarsegrain



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
[z_exp, xE_exp, xR_exp, T_exp, rho_solvent, dielec_solvent, ...
    Selec_exp, ip_Gmehling] = obj.experimentalData(salt, solvent, username);
z_exp_ternary = [z_exp(:,1) ./ sum(z_exp(:,1:3),2), z_exp(:,2) ./ sum(z_exp(:,1:3),2), z_exp(:,3) ./ sum(z_exp(:,1:3),2)];
z_exp_ternary = round(z_exp_ternary,5);



% UNIQUAC PARAMETERS FOR SOLVENT-WATER
% DIPA-H2O values are currently supplied; use UNIQUAC_Gibbsfits.m if you
% want to determine the UNIQUAC parameters for a different solvent-water
% pair. Note that it is most robust to solve for these separately and hold
% them constant during LIQUAC fitting
ip_UNIQUAC = [-2.13881699866569,	-2.13155946322135,	541.616375907126,	587.067085535208];



% OBJECT INITIALIZATION
% Initializes objects for each component of LIQUAC: Long, Medium, and Short
% Range, as well as the newly proposed model for medium range.
objLR = LongRange;
objMR = MediumRange;
objSR = ShortRange; 
objGibbs = GibbsMinimization;



% LIQUAC INTERACTION PARAMETERS TO BE FITTED: [b_ij, c_ij]
% Look to Gmehling group papers for good starting guesses
interaction_params = [-1.49245819676001,	1.99466864109731,	0.526147597939412,	0.705754761227743];



% COARSEGRAIN SETTING
% Sets the size of the search mesh in the Gibbs_LIQUAC_Eubanks function
% in GibbsMinimization.m
% coarsegrain == 1 means the mesh is large. This will make the script run
% faster but you get less resolution on the equilibrium mole fractions
% (which can sometimes be a problem given how small mole fractions of salt
% in the organic phase can be.
% corasegrain == 0 makes the mesh small but takes much longer to run
coarsegrain = 0;



% ALGORITHMIC PARAMETERS
% Upper and lower search bounds for the optimization algorithms
% Fitting algorithms: simulatedannealbnd for global maxima search,
% fminsearch for fine-tuning
% Returns the fitted interaction parameters (IP) based on the initial guess
% of interaction_params
lowerbound = ones(1,size(interaction_params,2)) * -10;
upperbound = lowerbound * -1;
%[IP, f_val, exitflag, output] = simulannealbnd(@fun, interaction_params, lowerbound, upperbound);
%[IP, f_val, exitflag, output] = fminsearch(@fun, interaction_params);



% ALGORITHMIC FITTING FUNCTION
%function RMS = fun(interaction_params)

    global r q MW valency z_exp xE_exp xR_exp T_exp rho_solvent dielec_solvent ...
        ip_UNIQUAC objLR objMR objSR objGibbs salt z_exp_ternary ...
        ip_Gmehling coarsegrain


    % Resets the full set of interaction parameters within the function for
    % use in the for loop. See Gmehling papers for explanation of these
    % variables (or my paper)
    b_ij = [interaction_params(1:2), ip_Gmehling(1:2)];
    c_ij = [interaction_params(3:4), ip_Gmehling(3:4)];
    b_jcja = ip_Gmehling(5);
    c_jcja = ip_Gmehling(6);
    ip = [b_ij, c_ij, b_jcja, c_jcja, ip_UNIQUAC];


    % Initializes xE and xR, both of which will be returned by the for loop
    % below
    xE = zeros(size(z_exp,1),size(z_exp,2));
    xR = zeros(size(z_exp,1),size(z_exp,2));
    
    % Initializes D_values, which will be returned by the for loop
    % D_values are the slopes of the tie lines connecting xE and xR
    D_values = zeros(size(z_exp,1), 1);


    % Parallelize your for loops where possible
    parfor i = 1:size(z_exp,1)

        % If D_values are unknown, use this function to solve for xE, xR,
        % and D_values
        [xE(i,:), xR(i,:), D_values(i,:)] = objGibbs.Gibbs_LIQUAC_Eubanks(z_exp_ternary(i,:), r, q, T_exp(i), ip, MW, valency, rho_solvent(i), dielec_solvent(i), salt, objMR, objSR, objLR, coarsegrain);
        
        
        % If D_values are known, use this function to solve for xE and xR
        % Do not use experimental D_values unless you are trying to test
        % something specific. If you simply try to fit the system with your
        % known experimental D values, you will get trivial results often.
        [xE(i,:), xR(i,:)] = objGibbs.Gibbs_LIQUAC_withD(z_exp_ternary(i,:), r, q, T_exp(i), ip, MW, valency, rho_solvent(i), dielec_solvent(i), salt, objMR, objSR, objLR, 5);
        
    end


    % OBJECTIVE FUNCTION
    % To be chosen by the user. This is the value that the global/local
    % search algorithm is trying to minimize
    % RMS stands for Root Mean Square
    RMS_AllAbsolute = (sum(sum((xE_exp - xE).^2)) + sum(sum((xR_exp - xR).^2))) * 1000;
    %RMS_AllRelative = sum(sum(((xE_exp - xE)./min(xE_exp,xE)).^2)) + sum(sum(((xR_exp - xR)./min(xR_exp,xR)).^2));
    RMS = RMS_AllAbsolute
    
    
   % SAVE D_Values
   % Use this line to save you D_values to the same folder as the rest of
   % the parameters, if you choose to do so.
   % You can use these D_values in DResolution.m to get further refinement
   % on xE, xR, and D (tie line slopes)
   % csvwrite(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/D_values.csv"), D_values)

%end