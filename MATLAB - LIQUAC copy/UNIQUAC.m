classdef UNIQUAC
    
    methods
        
        function ln_gamma = uniquac_calc(obj, x, r, q, T, interaction_parameters)
            [sum_xr, sum_xq, phi] = variables(obj, x, r, q);
            ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq);
            ln_gamma_resid = residual(obj, x, q, sum_xq, T, interaction_parameters);
            ln_gamma = ln_gamma_comb + ln_gamma_resid;      
        end

        function [ln_gamma_comb, ln_gamma_resid] = uniquac_detailed_calc(obj, x, r, q, T, interaction_parameters)
            [sum_xr, sum_xq, phi] = variables(obj, x, r, q);
            ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq);
            ln_gamma_resid = residual(obj, x, q, sum_xq, T, interaction_parameters);
        end

        function [sum_xr, sum_xq, phi] = variables(obj, x,r,q)
            sum_xr = x * transpose(r);
            sum_xq = x * transpose(q);
            phi = x .* q ./ sum_xq;
        end
    
        function ln_gamma_comb = combinatorial(obj, r, q, sum_xr, sum_xq)

            ln_gamma_comb = 1 - r ./ sum_xr + log(r ./ sum_xr) - ...
                5 * q .* (1 - r .* sum_xq ./ sum_xr ./ q + ...
                log(r .* sum_xq ./ sum_xr ./ q));
        
        end
    
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
        end
    
        % Calculation of interaction parameters, tau_ij, using both a_ij and
        % b_ij
        function tau = tau_calculation(obj, interaction_parameters, T)
            if size(interaction_parameters,2) == 2
               tau = [1, exp(interaction_parameters(1)) / T; exp(interaction_parameters(2)) / T, 1];
            end
        
            if size(interaction_parameters,2) == 4
                tau_12 = exp(interaction_parameters(1) + interaction_parameters(3) / T);
                tau_21 = exp(interaction_parameters(2) + interaction_parameters(4) / T);
                tau = [1, tau_12; tau_21, 1];
            end
            
            if size(interaction_parameters,2) == 6
                tau_12 = exp(interaction_parameters(1) / T);
                tau_21 = exp(interaction_parameters(2) / T);
                tau_13 = exp(interaction_parameters(3) / T);
                tau_31 = exp(interaction_parameters(4) / T);
                tau_23 = exp(interaction_parameters(5) / T);
                tau_32 = exp(interaction_parameters(6) / T);        
                tau = [1, tau_12, tau_13; tau_21, 1, tau_23; tau_31, tau_32, 1];
            end
            
            if size(interaction_parameters,2) == 12
                tau_12 = exp(interaction_parameters(1) + interaction_parameters(7) ./ T);
                tau_21 = exp(interaction_parameters(2) + interaction_parameters(8) ./ T);
                tau_13 = exp(interaction_parameters(3) + interaction_parameters(9) ./ T);
                tau_31 = exp(interaction_parameters(4) + interaction_parameters(10) ./ T);
                tau_23 = exp(interaction_parameters(5) + interaction_parameters(11) ./ T);
                tau_32 = exp(interaction_parameters(6) + interaction_parameters(12) ./ T);        
                tau = [1, tau_12, tau_13; tau_21, 1, tau_23; tau_31, tau_32, 1];
            end

        end
    end
end