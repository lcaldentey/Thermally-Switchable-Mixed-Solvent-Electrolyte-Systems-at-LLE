classdef GibbsMinimization

    methods

        % Solves for xE and xR gives a tie line slope, D
        function [xE, xR] = Gibbs_LIQUAC_withD(obj, z_exp_ternary, r, q, T, interaction_params, MW, valency, rho_solvent, dielec_solvent, salt, objMR, objSR, objLR, D)

            % Initializes interaction parameters, as specified in Gmehling
            % group papers
            b_ij = interaction_params(1:4);
            c_ij = interaction_params(5:8);
            b_jcja = interaction_params(9);
            c_jcja = interaction_params(10);
            ip_UNIQUAC = interaction_params(11:14);


            % Sets # of significant figures for the mole fractions to 5
            sigfigs = 5;


            % Initializes xE and xR, which will be solved for
            xE = zeros(1,4);
            xR = zeros(1,4);



            % Calculates minimum and maximum mole fractions of NaCl,
            % x_NaCl, given then slope of the tie line, D, and the feed
            % mole fractions, z_exp_ternary
            if D > 0
                x_NaCl_min = -(z_exp_ternary(2)/D - z_exp_ternary(3));
                x_NaCl_max = (D * z_exp_ternary(3) + 1 - z_exp_ternary(2)) / (1 + D);
            else
                x_NaCl_min = (D * z_exp_ternary(3) + 1 - z_exp_ternary(2)) / (1 + D);
                x_NaCl_max = z_exp_ternary(3) - z_exp_ternary(2) / D;
            end    
            


            % Rounds the minimum and maximum mole fractions of NaCl,
            % x_NaCl_min and x_NaCl_max, to the number of significant
            % figures determined above
            if (x_NaCl_min <= 0)
                x_NaCl_min = 10^-sigfigs;
            end
            if (x_NaCl_max > 1)
                x_NaCl_max = 1;
            end
            x_NaCl_min = round(x_NaCl_min, sigfigs);
            x_NaCl_max = round(x_NaCl_max, sigfigs);
            


            % Creates an array of mole fractions of NaCl, x_NaCl, based
            % upon the minimum and maximum values
            % Node_lengths is the spacing between each x_NaCl
            % Index is the number of each slot in the array
            x_NaCl = transpose(x_NaCl_min:(10^-sigfigs):(x_NaCl_max - (10^-sigfigs)));
            node_lengths = zeros(size(x_NaCl,1),1) + 10^-sigfigs;
            index = 1:size(x_NaCl,1);
        

            % Calculates the mole fractions of water and solvent, y_H2O and
            % z_solv, based on the mole fraction of NaCl and the slope of
            % the tie line
            y_H2O = -D * (x_NaCl_max - x_NaCl) + (1 - x_NaCl_max);
            z_solv = 1 - x_NaCl - y_H2O;
            x_tot = [z_solv, y_H2O, x_NaCl, x_NaCl];
            x_tot = x_tot ./ sum(x_tot,2);
    


            % Initializes key variable arrays
            temperature = zeros(size(x_NaCl,1),1) + T;
            rho_solv = zeros(size(x_NaCl,1),1) + rho_solvent;
            dielec_solv = zeros(size(x_NaCl,1),1) + dielec_solvent;
        


            % Calculates Long Range, Medium Range, and Short Range
            % contributions to the activity coefficients
            % Uses these contributions to calculate the activity
            % coefficient itself. Equation details are in Gmehling papers
            % or my paper
            ln_gamma_LR = objLR.func_LR(MW, valency, x_tot, temperature, rho_solv, dielec_solv, salt);
            ln_gamma_MR = objMR.func_MR(b_ij, c_ij, b_jcja, c_jcja, x_tot, MW, valency, salt);
            ln_gamma_SR = objSR.func_SR(x_tot, r, q, temperature, ip_UNIQUAC, MW);
            ln_gamma = ln_gamma_LR + ln_gamma_MR + ln_gamma_SR; 
            ln_gamma(:,3:4) = [abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,4), ...
                    abs(valency(3)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,3) + abs(valency(4)) / (abs(valency(3)) + abs(valency(4))) * ln_gamma(:,4)];
                
            


            % Calculates the delta_G_mix curve based on the mole fractions
            % of each component and the activity coefficients of each
            % component
            dg_mix = x_tot(:,1) .* log(x_tot(:,1)) + x_tot(:,2) .* log(x_tot(:,2)) + ...
                x_tot(:,3) .* log(x_tot(:,3)) + x_tot(:,1) .* ln_gamma(:,1) + ...
                x_tot(:,2) .* ln_gamma(:,2) + x_tot(:,3) .* ln_gamma(:,3) + ...
                x_tot(:,4) .* log(x_tot(:,4)) + x_tot(:,4) .* ln_gamma(:,4);
          
                

            % Creates an initial tangent line to the dg_mix curve using
            % x_NaCl_min and x_NaCl_max as the end points
            m1 = (dg_mix(size(x_NaCl,1)) - dg_mix(1)) / (x_NaCl(size(x_NaCl,1)) - x_NaCl(1));
            b1 = dg_mix(1) - m1 * x_NaCl(1);
            dg_line = m1 * x_NaCl + b1;



            % Finds a starting guess at minimums along the dg_mix curve
            deltas = dg_mix - dg_line;
            indices = index(islocalmin(deltas));
            


            % Checks whether there are two minimums on the dg_mix curve; if
            % not, then LLE equilibrium is not occuring
            if (size(indices, 2) ~= 2)
                indices = index(islocalmin(dg_mix));
            end
            


            % If two minimums do occur, use the Eubanks method to continue
            % refining the equilibrium molar composition
            if size(indices,2) == 2
                 
                guesses = indices(1:2);
    
                trap_area = abs(dg_mix(guesses(1)) + dg_mix(guesses(2))) * abs(x_NaCl(guesses(1)) - x_NaCl(guesses(2))) / 2;
                curve_area = abs(sum(dg_mix(guesses(1):guesses(2)) .* node_lengths((guesses(1):guesses(2)))));
                max_area_inner = sqrt(1 + D^2) * (trap_area - curve_area);
                xE = x_tot(guesses(1),:);
                xR = x_tot(guesses(2),:);
                                    
                for j = 1:(guesses(1) - 1)
        
                    trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                    curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                    difference = sqrt(1 + D^2) * (trap_area - curve_area);
        
                    if difference > max_area_inner
                        max_area_inner = difference;
                        xE = x_tot(j,:);
                        indices(1) = j;
                    elseif difference < max_area_inner
                        break
                    end
                end
        
                for j = guesses(1):(guesses(2) - 1)
        
                    trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                    curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                    difference = sqrt(1 + D^2) * (trap_area - curve_area);
        
                    if difference > max_area_inner
                        max_area_inner = difference;
                        xE = x_tot(j,:);
                        indices(1) = j;
                    elseif difference < max_area_inner
                        break
                    end
                end
    
                for k = (guesses(2) + 1):size(x_NaCl,1)
                    
                    trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                    curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                    difference = sqrt(1 + D^2) * (trap_area - curve_area);
                    
                    if difference > max_area_inner
                        max_area_inner = difference;
                        xR = x_tot(k,:);
                        indices(2) = k;
                    elseif difference < max_area_inner
                        break
                    end
                end
                
                
                for k = (indices(1) + 1):(guesses(2) - 1)
                    trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                    curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                    difference = sqrt(1 + D^2) * (trap_area - curve_area);
                    
                    if difference > max_area_inner
                        max_area_inner = difference;
                        xR = x_tot(k,:);
                        indices(2) = k;
                    elseif difference < max_area_inner
                        break
                    end
                end
                

            else
                xE = [100, 100, 100, 100];
                xR = [100, 100, 100, 100];
            end
        
            % Plots delta_G_mix equilibrium curve and tangent line
            %{
            m2 = (dg_mix(indices(1)) - dg_mix(indices(2))) / (x_NaCl(indices(1)) - x_NaCl(indices(2)));
            b2 = dg_mix(indices(1)) - m2 * x_NaCl(indices(1));
            dg_line2 = m2 * x_NaCl + b2;
            plot(x_NaCl, dg_mix)
            hold on
            plot(x_NaCl,dg_line2)
            %}
        
        
        end

        % There are two differences between this function and the one above
        % First, this function solves for the slope of the tie line, best_D
        % in addition to the equilibrium mole fractions xE and xR
        % In addition, you
        % can specify the mesh size (i.e., how many decimal places you want
        % it to solve x_NaCl to). Coarsegrain == 1 gives you a faster
        % search with x_NaCl resolved to 4 sig figs, while coarsegrain == 0
        % gives you a finer resolution on x_NaCl but is slower.
        function [xE, xR, best_D] = Gibbs_LIQUAC_Eubanks(obj, z_exp_ternary, r, q, T, interaction_params, MW, valency, rho_solvent, dielec_solvent, salt, objMR, objSR, objLR, coarsegrain)
            
            b_ij = interaction_params(1:4);
            c_ij = interaction_params(5:8);
            b_jcja = interaction_params(9);
            c_jcja = interaction_params(10);
            ip_UNIQUAC = interaction_params(11:14);
            

            % Specify the mesh size by passing coarsegrain as an input to
            % the function
            if coarsegrain == 1
                sigfigs = 4;
            else
               sigfigs = 5;
            end
            xE = zeros(1,4) + 100;
            xR = zeros(1,4) + 100;



            % Intial guess of the tie line slope, D, is zero. If the
            % algorithm finds a better value of D that maximizes the area
            % under the dG_mix curve, this value is saved as best_D
            % See Eubanks paper for additional details on the algorithm
            best_D = 0;
            D = 0;
            max_area = 0;


            % An upper bound to D is set to 100, but the user can override
            % this if there is a reason to
            while D < 100

                if D > -1
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
            
                x_NaCl = transpose(x_NaCl_min:10^-sigfigs:(x_NaCl_max - 10^-sigfigs));
                node_lengths = zeros(size(x_NaCl,1),1) + 10^-sigfigs;
                index = 1:size(x_NaCl,1);   
            
                y_H2O = -D * (x_NaCl_max - x_NaCl) + (1 - x_NaCl_max);
                z_solv = 1 - x_NaCl - y_H2O;
                x_tot = [z_solv, y_H2O, x_NaCl, x_NaCl];
                x_tot = x_tot ./ sum(x_tot,2);
            
                temperature = zeros(size(x_NaCl,1),1) + T;
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
            
                
                max_area_inner = 0;
                xE_inner = [0, 0, 0, 0];
                xR_inner = [0, 0, 0, 0];
            

                % not all values of D will result in a dg_mix curve with
                % real values
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
                        max_area_inner = sqrt(1 + D^2) * (trap_area - curve_area);
                        xE_inner = x_tot(guesses(1),:);
                        xR_inner = x_tot(guesses(2),:);
                                            
                        for j = 1:(guesses(1) - 1)
                
                            trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                            curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                            difference = sqrt(1 + D^2) * (trap_area - curve_area);
                
                            if difference > max_area_inner
                                max_area_inner = difference;
                                xE_inner = x_tot(j,:);
                                indices(1) = j;
                            elseif difference < max_area_inner
                                break
                            end
                        end
                
                        for j = guesses(1):(guesses(2) - 1)
                
                            trap_area = abs(dg_mix(guesses(2)) + dg_mix(j)) * abs(x_NaCl(guesses(2)) - x_NaCl(j)) / 2;
                            curve_area = abs(sum(dg_mix(j:guesses(2)) .* node_lengths(j:guesses(2))));
                            difference = sqrt(1 + D^2) * (trap_area - curve_area);
                
                            if difference > max_area_inner
                                max_area_inner = difference;
                                xE_inner = x_tot(j,:);
                                indices(1) = j;
                            elseif difference < max_area_inner
                                break
                            end
                        end
            
                        for k = (guesses(2) + 1):size(x_NaCl,1)
                            
                            trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                            curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                            difference = sqrt(1 + D^2) * (trap_area - curve_area);
                            
                            if difference > max_area_inner
                                max_area_inner = difference;
                                xR_inner = x_tot(k,:);
                                indices(2) = k;
                            elseif difference < max_area_inner
                                break
                            end
                        end
                        
                        
                        for k = (indices(1) + 1):(guesses(2) - 1)
                            trap_area = abs(dg_mix(k) + dg_mix(indices(1))) * abs(x_NaCl(k) - x_NaCl(indices(1))) / 2;
                            curve_area = abs(sum(dg_mix(indices(1):k) .* node_lengths(indices(1):k)));
                            difference = sqrt(1 + D^2) * (trap_area - curve_area);
                            
                            if difference > max_area_inner
                                max_area_inner = difference;
                                xR_inner = x_tot(k,:);
                                indices(2) = k;
                            elseif difference < max_area_inner
                                break
                            end
                        end
                    
                        if max_area_inner > max_area
                            max_area = max_area_inner;
                            xE = xE_inner;
                            xR = xR_inner;
                            best_D = D;
                            
                            % If a best value of D is found, only increase
                            % the next test of D by 1
                            D = D + 1;

                        % If the value of the Eubanks area starts
                        % decreasing, break the loop
                        elseif (max_area_inner > 0) && (max_area_inner < max_area)
                            D = 101;
                        else
                            D = D + 1;
                        end
                    else
                        D = D + 1;
                    end

                % if the guess for D resulting in a dg_mix curve with
                % imaginary numbers, increase the guess for D by 2.5
                % The user can adjust this guess as necessary
                else
                    D = D + 2.5;
                end

            end
        end


    end

end