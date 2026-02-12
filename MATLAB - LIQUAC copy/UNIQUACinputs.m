classdef UNIQUACinputs

    methods

        function [r, q, T_exp, xE_exp, xR_exp] = speciesData(obj, solvent)
            
            % van der Waals relative volume [solvent, water, cation, anion]
            r = readtable(strcat("/Users/kinnarishah/Desktop/",solvent,"/r.xlsx"));
            r = r{:,:};

            % van der Waals relative surface area [solvent, water, cation, anion]
            q = readtable(strcat("/Users/kinnarishah/Desktop/",solvent,"/q.xlsx"));
            q = q{:,:};

            % Experimentally-determined mole fractions of organic phase
            % [solvent, water, cation, anion]
            xE_exp = readtable(strcat("/Users/kinnarishah/Desktop/",solvent,"/xE_exp.xlsx"));
            xE_exp = xE_exp{:,:};
            
            % Experimentally-determined mole fractions of aqueous phase
            % [solvent, water, cation, anion]
            xR_exp = readtable(strcat("/Users/kinnarishah/Desktop/",solvent,"/xR_exp.xlsx"));
            xR_exp = xR_exp{:,:};
            
            % Absolute temperature (Kelvin), one cell per experiment
            T_exp = readtable(strcat("/Users/kinnarishah/Desktop/",solvent,"/T_exp.xlsx"));
            T_exp = T_exp{:,:};

        end

    end
end