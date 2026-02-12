classdef ShortRange % based on LIQUAC model
    
    methods
            
        % Short range activity coefficients, as specified by Kiepe et al.
        % LIQUAC model
        function ln_gamma = func_SR(obj, x, r, q, T, interaction_parameters, MW)

            [sum_xr, sum_xq] = variables(obj, x, r, q);
            ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq);
            ln_gamma_resid = residual(obj, x, q, sum_xq, T, interaction_parameters);
            ln_gamma = ln_gamma_comb + ln_gamma_resid;   

            
            MW_avg = x(:,1) * MW(1) + x(:,2) * MW(2);
            molality = [x(:,3), x(:,4)] ./ MW_avg;
            ln_gamma(:,3:4) = ln_gamma(:,3:4) - log(MW(2) ./ MW_avg + MW(2) * sum(molality,2));
            
        end
        
        % Output function that allows the user to see the breakdown between
        % residual and combinatorial contributions to the short range
        % activity coefficients for all components
        function [ln_gamma_resid, ln_gamma_comb] = output_SR(obj, x, r, q, T, interaction_parameters)
            [sum_xr, sum_xq] = variables(obj, x, r, q);
            ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq);
            ln_gamma_resid = residual(obj, x, q, sum_xq, T, interaction_parameters);

        end
        
        % Initializes key variables to UNIQUAC euqation
        function [sum_xr, sum_xq] = variables(obj, x,r,q)

            sum_xr = x * transpose(r);
            sum_xq = x * transpose(q);

        end
    
        % Calculates combinatorial portion of UNIQUAC model for all
        % components
        function ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq)

            ln_gamma_comb = 1 - r ./ sum_xr + log(r ./ sum_xr) - ...
                5 * q .* (1 - r .* sum_xq ./ sum_xr ./ q + ...
                log(r .* sum_xq ./ sum_xr ./ q));
        
            ln_gamma_comb_inf = 1 - r(3:4) ./ r(2) + log(r(3:4) / r(2)) ...
                - 5 * q(3:4) .* (1 - r(3:4) .* q(2) ./ (r(2) .* q(3:4)) ...
                + log(r(3:4) .* q(2) ./ (r(2) .* q(3:4))));

            ln_gamma_comb(:,3:4) = ln_gamma_comb(:,3:4) - ln_gamma_comb_inf;


        end
    
        % Calculates residual portion of UNIQUAC model for all
        % components
        function ln_gamma_resid = residual(obj, x, q, sum_xq, T, interaction_parameters)            
            
            ln_gamma_resid = zeros(size(x,1),size(x,2));
            for h = 1:size(x,1)

                tau = tau_calculation(obj, interaction_parameters, T(h));

                for i = 1:size(x,2)
            
                    sum_phi_j_tau_ji = sum(x(h,:) .* q .* transpose(tau(:,i))) / sum_xq(h);
                    sum_calc = zeros(1, size(x,2));
                    for j = 1:size(x,2)
                       
                        denominator = 0;
                        for k = 1:size(x,2)
                            denominator = denominator + q(k) * x(h,k) * tau(k, j);
                        end
                    
                        sum_calc(i) = sum_calc(i) + q(j) * x(h,j) * tau(i, j) / denominator;
            
                    end
                    
                    ln_gamma_resid(h, i) = q(i) * (1 - log(sum_phi_j_tau_ji) - sum_calc(i));
            
                end
            
            end
            ln_gamma_resid_inf_cation = q(3) * (1 - log(tau(2,3)) - tau(3,2));
            ln_gamma_resid_inf_anion = q(4) * (1 - log(tau(2,4)) - tau(4,2));

            ln_gamma_resid(:,3:4) = ln_gamma_resid(:,3:4) - [ln_gamma_resid_inf_cation, ln_gamma_resid_inf_anion];
 

        end

    
        % Calculation of interaction energies, tau_ij
        % tau_ij = a_ij + b_ij / T
        % Absolute temperature is used
        % a_ij and b_ij of solvent-ion and ion-ion pairs are set to
        % zero (as specified by Kiepe et al. (2006), resulting in tau_ij
        % these parameters to be equivalent to one.
        function tau = tau_calculation(obj, interaction_parameters, T)
            tau_12 = exp(interaction_parameters(1) + interaction_parameters(3) / T);
            tau_21 = exp(interaction_parameters(2) + interaction_parameters(4) / T);
            tau = [1, tau_12, 1, 1; tau_21, 1, 1, 1; 1, 1, 1, 1; 1, 1, 1, 1];
        end  

    end
    
end