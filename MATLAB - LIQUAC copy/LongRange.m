classdef LongRange

    methods

        % Returns long range contribution to gamma for all components based
        % on LIQUAC model
        function ln_gamma_LR = func_LR(obj, MW, valency, x, T, rho_solvent, dielec_solvent, salt)

            kg_solvent = x(:, 1) * MW(1) + x(:, 2) * MW(2);
            molality_cation = x(:, 3) ./ kg_solvent;
            molality_anion = x(:, 4) ./ kg_solvent;
            ionic_strength = 0.5 * (molality_cation * valency(3)^2 + molality_anion * valency(4)^2);
            x_saltfree = [x(:,1) ./ (x(:,1) + x(:,2)), x(:,2) ./ (x(:,1) + x(:,2))];
            
            % Densities in kg m^-3
            rho_H2O = -0.0043 * T.^2 + 2.3147 * T + 687.31;
            
            phi_prime = [x_saltfree(:,1) * MW(1) ./ rho_solvent ./ (x_saltfree(:,1) * MW(1) ./ rho_solvent + x_saltfree(:,2) * MW(2) ./ rho_H2O), x_saltfree(:,2) * MW(2) ./ rho_H2O ./ (x_saltfree(:,1) * MW(1) ./ rho_solvent + x_saltfree(:,2) * MW(2) ./ rho_H2O)];
            rho_solution = phi_prime(:,1) .* rho_solvent(:, 1) + phi_prime(:, 2) .* rho_H2O(:,1);
            dielec_H2O = 87.74 - 0.4008 * (T - 273.15) + 9.398 * 10^(-4) * (T - 273.15).^2 - 1.410 * 10^(-6) * (T - 273.15).^3;
            
            % Several different models for estimating the dielectric
            % constant of the solution exist. The one that is not commented
            % out is the chosen one, based on Oster mixing rule
            dielec_solution = dielec_H2O + ((dielec_solvent - 1) .* (dielec_solvent * 2 + 1) ./ (2 * dielec_solvent) - (dielec_H2O - 1)) .* phi_prime(:,1);
            %dielec_solution = dielec_H2O + ((dielec_solvent - 1) .* (dielec_solvent * 2 + 1) ./ (2 * dielec_solvent) - (dielec_H2O - 1)) .* phi_prime(1);
            %dielec_solution = dielec_H2O + phi_prime(:,1) ./ (1./(dielec_solvent - dielec_H2O) + phi_prime(:,2)./3./dielec_H2O);
            %dielec_solution_ions = dielec_solvent .* phi_prime(:,1) + dielec_H2O .* phi_prime(:,2);

            % Debye-Huckel parameters A and b
            A = 1.327757 * 10^5 * sqrt(rho_solution) ./ (dielec_solution .* T).^1.5;
            b = 6.359696 * sqrt(rho_solution) ./ sqrt(dielec_solution .* T);
            
            % Calculation of long range activity coefficients via LIQUAC
            kappa = 1 + b .* sqrt(ionic_strength) - 1 ./ (1 + b .* sqrt(ionic_strength)) - 2 * log(1 + b .* sqrt(ionic_strength)); 
            ln_gamma_solvents_LR = [2 * MW(1) .* A .* rho_solution ./ b.^3 ./ rho_solvent .* kappa, 2 * MW(2) .* A .* rho_solution ./ b.^3 ./ rho_H2O .* kappa];
            ln_gamma_ions_LR = [-valency(3).^2 .* A .* sqrt(ionic_strength) ./ (1 + b .* sqrt(ionic_strength)), -valency(4).^2 .* A .* sqrt(ionic_strength) ./ (1 + b .* sqrt(ionic_strength))];
            
            ln_gamma_LR = [ln_gamma_solvents_LR, ln_gamma_ions_LR];
        end

        
    end
end