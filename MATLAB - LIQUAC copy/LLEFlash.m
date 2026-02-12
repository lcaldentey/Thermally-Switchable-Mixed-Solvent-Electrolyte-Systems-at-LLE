
classdef LLEFlash

    methods

        % Calculate LLE splitting based on UNIQUAC modeling
        % Components 1 and 2 must be the solvent and water, respectively
        function [xE, xR] = Flash_UNIQUAC(obj, z, r, q, T, interaction_parameters)
            
            % Initializes key parameters for the flash function
            [errorK, tol, count, EOF_guess, xR, xE] = setParams(obj, z);

            % Establishes initial K value based on initialized xE and xR
            % values
            obj1 = UNIQUAC;
            gammaR = exp(obj1.uniquac_calc(xR, r, q, T, interaction_parameters));
            gammaE = exp(obj1.uniquac_calc(xE, r, q, T, interaction_parameters));    
            K = gammaR./gammaE;
            
            % Main algorithm loop
            while errorK > tol
                
                % % If the interaction parameters lead to nonsensical (e.g.,
                % non-real, extremely small, or infinite) results, breaks out
                % of loop as the inner while loop will not be able to resolve
                if (prod(K == real(K), 'all') && prod((K > 1e-10),'all'))
                    
                     % Solves for EOF (extract over feed ratio)
                    obj2 = @(EOF)sum(z.*(1-K)./(1+EOF*(K-1)));
                    EOFnew = fzero(obj2,EOF_guess);
                    EOF = EOFnew;
    
                    % Calculates new molar compositions
                    xR = z./(1+EOF*(K-1));
                    xE = xR.*K;
                    xR = xR/sum(xR);
                    xE = xE/sum(xE);
    
                    % Re-calculates new K based on new xE, xR
                    gammaR = exp(obj1.uniquac_calc(xR, r, q, T, interaction_parameters));
                    gammaE = exp(obj1.uniquac_calc(xE, r, q, T, interaction_parameters));
                    Knew = gammaR./gammaE;
    
                    % Calculates errorK to determine if loop complete
                    errorK = sum(abs(Knew - K)./(K));
                    K = Knew;
        
                    % Break out of loop if it takes too long
                    count = count + 1;
                    if count > 1000
                        %fprintf('Could not converge \n');
                        xE = rand(1) * 10^10;
                        xR = rand(1) * 10^10;
                        errorK = 10^-10;
                    end
                    
                    % K values will become 1 if the flash results in a one-phase product
                    if ((K(1) < 1+tol) && (K(1) > 1-tol))
                        %fprintf('Feed results in one phase \n'); 
                        xE = rand(1) * 10^10;
                        xR = rand(1) * 10^10;
                        errorK = 10^-10;
                    end

                else 
                    %fprintf('Non real solution \n')
                    xE = rand(1) * 10^10;
                    xR = rand(1) * 10^10;
                    errorK = 10^-10;

                end

            end
        end

        % Flash solver typically results in errors for systems at LLE. The
        % function is included here, but not used
        function [xE, xR] = Flash_LIQUAC(obj, z, r, q, T, interaction_params, MW, valency, rho_solvent, dielec_solvent, salt, objMR, objSR, objLR)
            
            % Initializes key parameters for the flash function
            [errorK, tol, count, EOF_guess, xR, xE] = setParams(obj, z);
            

            % Establishes initial K value based on initialized xE and xR
            % values
            b_ij = interaction_params(1:4);
            c_ij = interaction_params(5:8);
            b_jcja = interaction_params(9);
            c_jcja = interaction_params(10);
            ip_UNIQUAC = interaction_params(11:14);

            ln_gamma_LR_E = objLR.func_LR(MW, valency, xE, T, rho_solvent, dielec_solvent, salt);
            ln_gamma_MR_E = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, xE, MW, valency, salt);
            ln_gamma_SR_E = objSR.func_SR(xE, r, q, T, ip_UNIQUAC, MW);
            ln_gamma_E = ln_gamma_LR_E + ln_gamma_MR_E + ln_gamma_SR_E; 
            ln_gamma_E(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,4), ...
                    abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,4)];
            gammaE = exp(ln_gamma_E);
        
            ln_gamma_LR_R = objLR.func_LR(MW, valency, xR, T, rho_solvent, dielec_solvent, salt);
            ln_gamma_MR_R = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, xR, MW, valency, salt);
            ln_gamma_SR_R = objSR.func_SR(xR, r, q, T, ip_UNIQUAC, MW);
            ln_gamma_R = ln_gamma_LR_R + ln_gamma_MR_R + ln_gamma_SR_R;
            ln_gamma_R(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,4), ...
                    abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,4)];
            gammaR = exp(ln_gamma_R);
            
            K = gammaR./gammaE;
            
            % Main algorithm loop
            while errorK > tol
                
                % % If the interaction parameters lead to nonsensical (e.g.,
                % non-real, extremely small, or infinite) results, breaks out
                % of loop as the inner while loop will not be able to resolve
                if (prod(K == real(K), 'all') && prod((K > 1e-10),'all'))
                    
                     % Solves for EOF (extract over feed ratio)
                    obj2 = @(EOF)sum(z.*(1-K)./(1+EOF*(K-1)));
                    EOFnew = fzero(obj2,EOF_guess);
                    EOF = EOFnew;
    
                    % Calculates new molar compositions
                    xR = z./(1+EOF*(K-1));
                    xE = xR.*K;
                    xR = xR/sum(xR);
                    xE = xE/sum(xE);
    
                    % Re-calculates new K based on new xE, xR
                    ln_gamma_LR_E = objLR.func_LR(MW, valency, xE, T, rho_solvent, dielec_solvent, salt);
                    ln_gamma_MR_E = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, xE, MW, valency, salt);
                    ln_gamma_SR_E = objSR.func_SR(xE, r, q, T, ip_UNIQUAC, MW);
                    ln_gamma_E = ln_gamma_LR_E + ln_gamma_MR_E + ln_gamma_SR_E; 
                    ln_gamma_E(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,4), ...
                            abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_E(:,4)];
                    gammaE = exp(ln_gamma_E);
                
                    ln_gamma_LR_R = objLR.func_LR(MW, valency, xR, T, rho_solvent, dielec_solvent, salt);
                    ln_gamma_MR_R = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, xR, MW, valency, salt);
                    ln_gamma_SR_R = objSR.func_SR(xR, r, q, T, ip_UNIQUAC, MW);
                    ln_gamma_R = ln_gamma_LR_R + ln_gamma_MR_R + ln_gamma_SR_R;
                    ln_gamma_R(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,4), ...
                            abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma_R(:,4)];
                    gammaR = exp(ln_gamma_R);
                    
                    Knew = gammaR./gammaE;
    
                    % Calculates errorK to determine if loop complete
                    errorK = sum(abs(Knew - K)./(K));
                    K = Knew;
        
                    % Break out of loop if it takes too long
                    count = count + 1;
                    if count > 1000
                        %fprintf('Could not converge \n');
                        xE = rand(1) * 10^10;
                        xR = rand(1) * 10^10;
                        errorK = 10^-10;
                    end
                    
                    % K values will become 1 if the flash results in a one-phase product
                    if ((K(1) < 1+tol) && (K(1) > 1-tol))
                        %fprintf('Feed results in one phase \n'); 
                        xE = rand(1) * 10^10;
                        xR = rand(1) * 10^10;
                        errorK = 10^-10;
                    end

                else 
                    %fprintf('Non real solution \n')
                    xE = rand(1) * 10^10;
                    xR = rand(1) * 10^10;
                    errorK = 10^-10;

                end

            end
        end


        function [xE, xR] = Gibbs_UNIQUAC(obj, r, q, T, interaction_parameters)
            
            x1 = transpose([(1:999)/10000,(100:999)/1000]);
            x2 = 1 - x1;
            node_lengths = [zeros(999,1) + 0.0001; zeros(900,1) + 0.001];
            T = zeros(size(x1,1)) + T;
            
            obj_UNIQUAC = UNIQUAC;
            ln_gamma = obj_UNIQUAC.uniquac_calc([x1, x2], r, q, T, interaction_parameters);        
            dg_mix = x1 .* log(x1) + x2 .* log(x2) + x1 .* ln_gamma(:,1) + x2 .* ln_gamma(:,2);
    
            max_area = 0;
            max_j = 2;

            xE = [0, 0];
            xR = [0, 0];
    
            for j = 2:size(x1, 1)
                trap_area = abs(dg_mix(1) + dg_mix(j)) * abs(x1(1) - x1(j)) / 2;
                curve_area = abs(sum(dg_mix(1:j) .* node_lengths(1:j)));
                difference = trap_area - curve_area;

                if difference > max_area
                    max_area = difference;
                    xE = [x1(j), 1 - x1(j)];
                    max_j = j;
                end
            end
                
            if (sum(xE) == 0)
                xE = [100, 100];
            end
            
            for k = 2:max_j
                trap_area = abs(dg_mix(k) + dg_mix(max_j)) * abs(x1(k) - xE(1)) / 2;
                curve_area = abs(sum(dg_mix(k:max_j) .* node_lengths(k:max_j)));
                difference = trap_area - curve_area;
                
                if difference > max_area
                    max_area = difference;
                    xR = [x1(k), 1 - x1(k)];
                end
            end
    
            if (sum(xR) == 0)
                xR = [100, 100];
            end

            plot(x1, dg_mix)

        end


        % Initializes key parameters for determining mole fractions
        function [errorK, tol, count, EOF_guess, xR, xE] = setParams(obj, z)             
            
            errorK = 1; % change in K value divided by K
            tol = 0.00001; % acceptable value of errorK
            count = 0; % iteration number
            EOF_guess = 0.5; % initial guess for E/F (extract over feed)

            % Initializes xR and xE, note that zeros in the composition
            % vectors will lead to NaN solutions
            xR = zeros(1,length(z)) + 1e-50;
            xE = xR;
            xR(1:2) = [0.02 0.98];
            xE(1:2) = [0.98 0.02];
        end
        

    end
end