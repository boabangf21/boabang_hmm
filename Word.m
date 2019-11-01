classdef Word < handle
    properties
      Nbstates    =  3; % number of states
        A     = []; % Transition probability matrix
        prior = []; % Intial state distribution
        mu    = []; % Mean
        Sigma = []; % Covariance matrix
        name  = '';
    end
    
    methods
        function current_word = Word(name)
            current_word.name = char(name);
        end
        
        function log_likelihood = log_likelihood(current_word, O)
            B = current_word.state_likelihood(O);
            log_likelihood = forward(current_word, O, B);
        end
        
        function [log_likelihood, alpha] = forward(current_word, O, B)
            log_likelihood = 0;
            T = size(O, 2);
            alpha = zeros(current_word.Nbstates, T);
            
           
                    % Initialization
                    alpha(:, 1) = B(:, 1) .* current_word.prior;
              for t=2:T
                    % Induction
                    alpha(:, t) = B(:, t) .* (current_word.A' * alpha(:, t - 1));
                
                
                % Scaling to avoid overflow
                alpha_sum      = sum(alpha(:, t));
                alpha(:, t)    = alpha(:, t) ./ alpha_sum;
                log_likelihood = log_likelihood + log(alpha_sum);
                end
            end
        
        
        function beta = backward(current_word, O, B)
            T = size(O, 2);
            beta = zeros(current_word.Nbstates, T);
            
            % Initialization
            beta(:, T) = ones(current_word.Nbstates, 1);
            
            for t = (T - 1):-1:1
                % Induction
                beta(:, t) = current_word.A * (beta(:, t + 1) .* B(:, t + 1));
                
                % Scaling
                beta(:, t) = beta(:, t) ./ sum(beta(:, t));
            end
        end
        
        % Evaluates the Gaussian pdfs for each state at the O
        % Returns a matrix containing B(s, t) = f(O_t | S_t = s)
        function B = state_likelihood(current_word, O)
            B = zeros(current_word.Nbstates, size(O, 2));
            
            for s = 1:current_word.Nbstates               
                B(s, :) = mvnpdf(O', current_word.mu(:, s)', current_word.Sigma(:, :, s));
            end
        end
        
        function em_initialize(current_word, O)
           
            current_word.prior = normalise(rand(current_word.Nbstates, 1));
            current_word.A     = mk_stochastic(rand(current_word.Nbstates));
            
            
            current_word.Sigma = repmat(diag(diag(cov(O'))), [1 1 current_word.Nbstates]);
            
           
            indices = randperm(size(O, 2));
            current_word.mu = O(:, indices(1:current_word.Nbstates));
        end
        
        function train(current_word, O)
            current_word.em_initialize(O);

            for i = 1:15
                log_likelihood = current_word.em_step(O);
                display(sprintf('Step %02d: log_likelihood = %f', i, log_likelihood))
                current_word.plot_gaussians(O);
            end
        end
        
        function log_likelihood = em_step(current_word, O)
            B = current_word.state_likelihood(O);
            D = size(O, 1);
            T = size(O, 2);
            
            [log_likelihood, alpha] = current_word.forward(O, B);
            beta                    = current_word.backward(O, B);
            
            xi_sum = zeros(current_word.Nbstates, current_word.Nbstates);
            gamma  = zeros(current_word.Nbstates, T);
            
            for t = 1:(T - 1)
                
                xi_sum      = xi_sum + normalise(current_word.A .* (alpha(:, t) * (beta(:, t + 1) .* B(:, t + 1))'));
                gamma(:, t) = normalise(alpha(:, t) .* beta(:, t));
            end
            
            gamma(:, T) = normalise(alpha(:, T) .* beta(:, T));
            
            expected_prior = gamma(:, 1);
            expected_A     = mk_stochastic(xi_sum);
            
            expected_mu    = zeros(D, current_word.Nbstates);
            expected_Sigma = zeros(D, D, current_word.Nbstates);
            
            gamma_state_sum = sum(gamma, 2);
            
           
            gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0);
            
            for s = 1:current_word.Nbstates
                gamma_observations = O .* repmat(gamma(s, :), [D 1]);
                expected_mu(:, s)  = sum(gamma_observations, 2) / gamma_state_sum(s);
                
                
                expected_Sigma(:, :, s) = symmetrize(gamma_observations * O' / gamma_state_sum(s) - ...
                                                     expected_mu(:, s) * expected_mu(:, s)');
            end
            
            % To maintain positive semidefiniteness
            expected_Sigma = expected_Sigma + repmat(0.01 * eye(D, D), [1 1 current_word.Nbstates]);
            
            
            current_word.prior = expected_prior;
            current_word.A     = expected_A;
            current_word.mu    = expected_mu;
            current_word.Sigma = expected_Sigma;
        end
        
        function plot_gaussians(current_word, O)
            
            
            plot(O(1, :), O(2, :), 'g+')
            hold on
            plot(current_word.mu(1, :), current_word.mu(2, :), 'r*')

            for s = 1:size(current_word.Sigma, 3)
                error_ellipse(current_word.Sigma(1:2, 1:2, s), 'mu', current_word.mu(1:2, s), 'style', 'r-', 'conf', .75)
            end

            axis([0 4000 0 4000])
            hold off
            title(sprintf('Training %s', current_word.name))
            xlabel('F1 [Hz]')
            ylabel('F2 [Hz]')
            drawnow
            
           % pause
        end
    end
end