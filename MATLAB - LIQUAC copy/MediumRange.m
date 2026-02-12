classdef MediumRange % based on LIQUAC model
    
    methods

        % Medium range activity coefficients, no temperature dependence, as
        % specified by Kiepe et al. LIQUAC
        function ln_gamma_MR = func_MR(obj, b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt)
            
            % Initializes key variables and interaction terms B_ij
            [m_ion, ionic_strength, x_saltfree, mean_MW] = variables(obj, x, MW, valency);
            [B_solv, dB_solv, B_ion, dB_ion] = interactionTerms(obj, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, salt);
        
            % Calculates solvent and ionic activity coefficients
            [ln_gamma_solv_MR] = solventGamma(obj, m_ion, x_saltfree, ionic_strength, MW, mean_MW, B_solv, dB_solv, B_ion, dB_ion);
            [ln_gamma_ions_MR] = ionsGamma(obj, mean_MW, m_ion, x_saltfree, B_solv, dB_solv, B_ion, dB_ion, MW, b_ij, c_ij, valency);
            
            ln_gamma_MR = [ln_gamma_solv_MR, ln_gamma_ions_MR];            
        end

        % Medium range activity coefficients, as specified by Kiepe et al.
        % LIQUAC, with temperature dependence added into the interaction
        % terms
        function ln_gamma_MR = func_MR_T(obj, b_ij, c_ij, b_jcja, c_jcja, x, MW, valency, salt, T)
            
            % Initializes key variables and interaction terms B_ij
            [m_ion, ionic_strength, x_saltfree, mean_MW] = variables(obj, x, MW, valency);
            [B_solv, dB_solv, B_ion, dB_ion] = interactionTerms_TempDependency(obj, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, T, salt);
        
            % Calculates solvent and ionic activity coefficients
            [ln_gamma_solv_MR] = solventGamma(obj, m_ion, x_saltfree, ionic_strength, MW, mean_MW, B_solv, dB_solv, B_ion, dB_ion);
            [ln_gamma_ions_MR] = ionsGamma(obj, mean_MW, m_ion, x_saltfree, B_solv, dB_solv, B_ion, dB_ion, MW, b_ij, c_ij, valency);
            
            ln_gamma_MR = [ln_gamma_solv_MR, ln_gamma_ions_MR];                  
        end


        % Initializes key variables for medium range activity coefficients
        function [m_ion, ionic_strength, x_saltfree, mean_MW] = variables(obj, x, MW, valency)
            
            kg_solvent = x(:, 1) * MW(1) + x(:, 2) * MW(2);
            m_ion = x(:,3:4) ./ kg_solvent;
            ionic_strength = 0.5 * sum(m_ion .* valency(3:4).^2,2);
            x_saltfree = [x(:,1) ./ (x(:,1) + x(:,2)), x(:,2) ./ (x(:,1) + x(:,2))];
            mean_MW = x_saltfree(:,1) * MW(1) + x_saltfree(:,2) * MW(2);

        end

        % Calculates B_solv, dB_solv, B_ion, dB_ion, no temperature
        % dependence
        function [B_solv, dB_solv, B_ion, dB_ion] = interactionTerms(obj, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, salt)
            % Set variables a1-a4 based on LIQUAC paper
            
            a_1 = -1.2;
            a_3 = -1;
            if salt == "LiCl"
                a_4 = 0.1451;
            elseif salt == "LiBr"
                a_4 = 0.0695;
            elseif salt == "LiI"
                a_4 = 0.1797;
            elseif salt == "LiOH"
                a_4 = 0.0894;
            elseif salt == "LiNO3"
                a_4 = 0.1820;
            elseif salt == "Li2SO4"
                a_4 = 0.2936;
            elseif salt == "LiClO3"
                a_4 = 0.1325;
            elseif salt == "CaCl2"
                a_4 = 0.2170;
            else
                a_4 = 0.1250;
            end
            a_2 = 2 * a_4;
            
            % [solvent-cation, solvent-anion, water-cation, water-anion]
            B_solv = b_ij + c_ij .* exp(a_1 * sqrt(ionic_strength) + a_2 * ionic_strength);
            dB_solv = (a_1 / 2 ./ sqrt(ionic_strength) + a_2) .* c_ij .* exp(a_1 * sqrt(ionic_strength) + a_2 * ionic_strength);
            B_ion = b_jcja + c_jcja .* exp(a_3 * sqrt(ionic_strength) + a_4 * ionic_strength);
            dB_ion = (a_3 / 2 ./ sqrt(ionic_strength) + a_4) .* c_jcja .* exp(a_3 * sqrt(ionic_strength) + a_4 * ionic_strength);            
        end

        % Calculates B_solv, dB_solv, B_ion, dB_ion, with temperature
        % dependence
        function [B_solv, dB_solv, B_ion, dB_ion] = interactionTerms_TempDependency(obj, b_ij, c_ij, b_jcja, c_jcja, ionic_strength, T, salt)
            
            % Set variables a1-a4 based on LIQUAC paper
            a_1 = -1.2;
            a_3 = -1;
            if salt == "LiCl"
                a_4 = 0.1451;
            elseif salt == "LiBr"
                a_4 = 0.0695;
            elseif salt == "LiI"
                a_4 = 0.1797;
            elseif salt == "LiOH"
                a_4 = 0.0894;
            elseif salt == "LiNO3"
                a_4 = 0.1820;
            elseif salt == "Li2SO4"
                a_4 = 0.2936;
            elseif salt == "LiClO3"
                a_4 = 0.1325;
            elseif salt == "CaCl2"
                a_4 = 0.2170;
            else
                a_4 = 0.1250;
            end
            a_2 = 2 * a_4;

            % [solv_1-cation, solv_1-anion, solv_2-cation, solv2_anion]
            B_solv = b_ij ./ T + c_ij ./ T .* exp(a_1 * sqrt(ionic_strength) + a_2 * ionic_strength);
            dB_solv = (a_1 / 2 ./ sqrt(ionic_strength) + a_2) .* c_ij ./ T .* exp(a_1 * sqrt(ionic_strength) + a_2 * ionic_strength);
            B_ion = b_jcja ./ T + c_jcja ./ T .* exp(a_3 * sqrt(ionic_strength) + a_4 * ionic_strength);
            dB_ion = (a_3 / 2 ./ sqrt(ionic_strength) + a_4) .* c_jcja ./ T .* exp(a_3 * sqrt(ionic_strength) + a_4 * ionic_strength);
           
        end

        % Calculates medium range activity coefficients for solvents
        function [ln_gamma_solv_MR] = solventGamma(obj, m_ion, x_saltfree, ionic_strength, MW, mean_MW, B_solv, dB_solv, B_ion, dB_ion)

            Solv1_Term1 = sum(m_ion .* B_solv(:,1:2),2);
            Solv1_Term2 = MW(1) ./ mean_MW .* (sum((x_saltfree(:,1) .* m_ion .* (B_solv(:,1:2) + ionic_strength .* dB_solv(:,1:2))),2) + sum((x_saltfree(:,2) .* m_ion .* (B_solv(:,3:4) + ionic_strength .* dB_solv(:,3:4))),2));
            
            Solv2_Term1 = sum(m_ion .* B_solv(:,3:4),2);
            Solv2_Term2 = MW(2) ./ mean_MW .* (sum((x_saltfree(:,1) .* m_ion .* (B_solv(:,1:2) + ionic_strength .* dB_solv(:,1:2))),2) + sum((x_saltfree(:,2) .* m_ion .* (B_solv(:,3:4) + ionic_strength .* dB_solv(:,3:4))),2));
            
            Solv_Term3 = (m_ion(:,1) .* m_ion(:,2)) .* (B_ion + ionic_strength .* dB_ion);
            
            ln_gamma_solv_MR = [Solv1_Term1 - Solv1_Term2 - MW(1) .* Solv_Term3, Solv2_Term1 - Solv2_Term2 - MW(2) .* Solv_Term3];           
        end

        % Calculates medium range activity coefficients for ions
        function [ln_gamma_ions_MR] = ionsGamma(obj, mean_MW, m_ion, x_saltfree, B_solv, dB_solv, B_ion, dB_ion, MW, b_ij, c_ij, valency)            
            
            Cation_Term1 = 1 ./ mean_MW .* sum(x_saltfree(:,1) .* B_solv(:,1) + x_saltfree(:,2) .* B_solv(:,3),2);
            Anion_Term1 = 1 ./ mean_MW .* sum(x_saltfree(:,1) .* B_solv(:,2) + x_saltfree(:,2) .* B_solv(:,4),2);
            
            Ions_Term2 = (valency(:,3:4).^2) ./ (2 .* mean_MW) .* (sum(x_saltfree(:,1) .* m_ion .* dB_solv(:,1:2),2) + sum(x_saltfree(:,2) .* m_ion .* dB_solv(:,3:4),2));
            Ions_Term3 = [m_ion(:,2) .* B_ion, m_ion(:,1) .* B_ion];
            Ions_Term4 = 0.5 * (valency(:,3:4).^2) .* sum(m_ion(:,1) .* m_ion(:,2).* dB_ion,2);

            % Reference states
            Ions_RefState_Cation = 1 ./ MW(2) .* (b_ij(:,3) + c_ij(:,3));
            Ions_RefState_Anion = 1 ./ MW(2) .* (b_ij(:,4) + c_ij(:,4));
            
            ln_gamma_ions_MR = [Cation_Term1 + Ions_Term2(:,1) + Ions_Term3(:,1) + Ions_Term4(:,1) - Ions_RefState_Cation, Anion_Term1 + Ions_Term2(:,2) + Ions_Term3(:,2) + Ions_Term4(:,2) - Ions_RefState_Anion];           
            
        end

        
    end
end
