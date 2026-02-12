% The purpose of this script is to pull all the relevant data to run the
% LIQUAC model. All of these files must be supplied by the user and must be
% saved in the format:
% "/Users/username/Desktop/salt-solvent/parameters.xlsx"
% If you wish to change the format, please do so manually

classdef LIQUACinputs

    methods

        % Pulls species relevant data
        % All input files must be in the format: [solvent water cation anion]
        function [r, q, MW, valency] = speciesData(obj, salt, solvent, username)
            
            % van der Waals relative volume
            r = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/r.xlsx"));
            r = r{:,:};

            % van der Waals relative surface area
            q = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/q.xlsx"));
            q = q{:,:};

            % Molecular weights in kg/mol
            MW = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/MW.xlsx"));
            MW = MW{:,:};

            valency = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/valency.xlsx"));
            valency = valency{:,:};
        end

        function [z_exp, xE_exp, xR_exp, T_exp, rho_solvent, dielec_solvent, Selec_exp, ip_Gmehling] = experimentalData(obj, salt, solvent, username)

            % Molar feed composition [solvent, water, cation, anion]
            z_exp = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/z_exp.xlsx"));
            z_exp = z_exp{:,:};
            
            % Experimentally-determined mole fractions of organic phase
            % [solvent, water, cation, anion]
            xE_exp = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/xE_exp.xlsx"));
            xE_exp = xE_exp{:,:};
            
            % Experimentally-determined mole fractions of aqueous phase
            % [solvent, water, cation, anion]
            xR_exp = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/xR_exp.xlsx"));
            xR_exp = xR_exp{:,:};
            
            % Absolute temperature (Kelvin), one cell per experiment
            T_exp = readtable(strcat("/Users/",username,"/Desktop/",salt,"-",solvent,"/T_exp.xlsx"));
            T_exp = T_exp{:,:};
            
            % Supplies solvent density and dielectric constant
            % Add in new solvent values as appropriate
            if contains(solvent, "DIPA")
                rho_solvent = -0.9675 * T_exp + 1003.6; % https://doi.org/10.1016/j.jct.2018.12.012
                dielec_solvent = -7.32 * 10^(-3) * (T_exp - 273.15) + 3.24; % experimentally determined by Eliza Dach
            else
                fprintf(['Solvent density and dielectric data not' ...
                    'supplied, please see LIQUACinputs.m']);
            end


            % [b_ij, c_ij] parameters for water and different salts, as
            % determined by Gmehling group
            if salt == "NaCl"
                ip_Gmehling = [0.00331, -0.00128, -0.00143, -0.00020, 0.17219, -0.26495];
            elseif salt == "LiCl"
                ip_Gmehling = [0.00319, -0.00128, -0.00099, -0.00020, 0.37690, -0.36090];
            elseif salt == "KCl"
                ip_Gmehling = [0.0258, -0.00128, -0.00088, -0.00020, 0.09387, -0.19630];
            elseif salt == "KBr"
                ip_Gmehling = [0.0258, -0.00247, -0.00088, -0.00008, 0.11020, -0.15500];
            elseif salt == "NaBr"
                ip_Gmehling = [0.00331, -0.00247, 0.00143, -0.00008, 0.21660, -0.22130];
            else
                fprintf(['Gmehling parameters for this salt are not' ...
                    'supplied, please see LIQUACinputs.m']);
            end

            % Experimentally-determined selectivity, defined as: 
            % x_H2O,org / x_ion,org / 1000
            Selec_exp = [xE_exp(:,2) ./ xE_exp(:,3) / 1000, xE_exp(:,2) ./ xE_exp(:,4) / 1000];

        end

    end
end