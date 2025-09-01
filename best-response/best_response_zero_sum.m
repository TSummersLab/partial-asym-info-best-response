function [K1, K2, L1, L2, P1, P2, Sigma1, Sigma2, A1bar, A2bar, B1bar, B2bar, C1bar, C2bar, F1, F2, W1bar, W2bar, total_cost, LQG_cost1, LQG_cost2, iter] = best_response_zero_sum(A, B1, B2, C1, C2, D1, D2, Q, R1, R2, W, V1, V2, LQG_player, tolerance, max_num_iteration)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % A: system matrix
    % B1: Player 1's control matrix
    % B2: Player 2's control matrix
    % C1: Player 1's measurement matrix
    % C2: Player 2's measurement matrix
    % D1: Player 1's measurement attack matrix
    % D2: Player 2's measurement attack matrix
    % Q: state penalty
    % R1: player 1's input penalty
    % R2: player 2's input penalty
    % W: process noise covariance matrix
    % V1: player 1's measurement noise covariance matrix
    % V2: player 2's measurement noise covariance matrix
    % LQG_player: the player does LQG control at 1st iteration
    % tolerance: best response convergence tolerance
    % max_num_iteration: maximum number of iterations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % K1: player 1's policy gain
    % K2: player 2's policy gain
    % L1: player 1's state estimator gain
    % L2: player 2's state estimator gain
    % P1: player 1's value matrix
    % P2: player 2's value matrix
    % Sigma1: player 1's estimation error covariance matrix
    % Sigma2: player 2's estimation error covariance matrix
    % A1bar: player 1's augmented belief dynamics
    % A2bar: player 2's augmented belief dynamics
    % B1bar: player 1's augmented input matrix
    % B2bar: player 2's augmented input matrix
    % C1bar: player 1's augmented output matrix
    % C2bar: player 2's augmented output matrix
    % F1: player 1's augmented process noise matrix
    % F2: player 2's augmented process noise matrix
    % W1bar: player 1's augmented process noise covariance matrix
    % W2bar: player 2's augmented process noise covariance matrix
    % total_cost: zero-sum game's optimal value
    % LQG_cost1: player 1's optimal LQG cost
    % LQG_cost2: player 2's optimal LQG cost
    % iter: number of best response iterations

    % Parameter definition
    n = size(A, 1);
    p1 = size(C1, 1);
    p2 = size(C2, 1);
    A1bar = A;
    B1bar = B1;
    C1bar = C1;
    F1 = eye(n);
    W1bar = W;
    A2bar = A;
    B2bar = B2;
    C2bar = C2;
    F2 = eye(n);
    W2bar = W;
    total_cost = 0;
    LQG_cost1 = inf * ones(max_num_iteration, 1);
    LQG_cost2 = inf * ones(max_num_iteration, 1);

    if LQG_player == 1
        % Best response 1st iteration
        iter = 1;
        disp("Iteration: " + num2str(iter));

        % LQG control of player 1
        [Sigma1, ~, ~] = idare(A', C1', W, V1, [], []);
        L1 = A * Sigma1 * C1' / (V1 + C1 * Sigma1 * C1');

        [P1, ~, ~] = idare(A, B1, Q, R1, [], []);
        K1 = -(R1 + B1' * P1 * B1) \ B1' * P1 * A;

        % Closed-loop state and state estimate dynamics of player 1
        A1Bar = [A B1 * K1; L1 * C1 A + B1 * K1 - L1 * C1];
        Q1Bar = [Q zeros(n); zeros(n) K1' * R1 * K1];
        W1Bar = [W zeros(n); zeros(n) L1 * V1 * L1'];

        % Optimal cost of player 1
        P1Bar = dlyap(A1Bar', Q1Bar);
        LQG_cost1(iter, 1) = trace(P1Bar * W1Bar);

        % Best response from player 2
        A2bar = A1Bar;
        B2bar = [B2; L1 * D2];
        C2bar = [C2 D1 * K1];
        F2 = [eye(n) zeros(n, p1); zeros(n) L1];
        W2bar = [W zeros(n, p1); zeros(p1, n) V1];
        Q2bar = [Q zeros(n); zeros(n) K1' * R1 * K1];
        
        % Compute L2, Sigma2, K2, and P2
        [Sigma2, ~, ~] = idare(A2bar', C2bar', F2 * W2bar * F2', V2, [], []);
        L2 = A2bar * Sigma2 * C2bar' / (V2 + C2bar * Sigma2 * C2bar');

        [P2, ~, ~] = idare(A2bar, B2bar, Q2bar, R2, [], []);

        if isempty(P2)
            disp('Error: please increase the input penalty magnitude for player 2.');
            K2 = [];
        else
            K2 = -(R2 + B2bar' * P2 * B2bar) \ B2bar' * P2 * A2bar;

            % Closed-loop state and state estimate dynamics of player 2
            A2Bar = [A2bar B2bar * K2; L2 * C2bar A2bar + B2bar * K2 - L2 * C2bar];
            Q2Bar = [Q2bar zeros(2*n); zeros(2*n) K2' * R2 * K2];
            F2Bar = [F2 zeros(2*n, p2); zeros(2*n, n+p1) L2];
            W2Bar = F2Bar * [W2bar zeros(n+p1, p2); zeros(p2, n+p1) V2]* F2Bar';
    
            % Optimal cost of player 2
            P2Bar = dlyap(A2Bar', Q2Bar);
            LQG_cost2(iter, 1) = trace(P2Bar * W2Bar);
        end

        % Optimal cost difference between player 1 and player 2
        cost_diff = abs(LQG_cost2(iter, 1) - LQG_cost1(iter, 1));

        while cost_diff > tolerance && ~isempty(P2) && iter < max_num_iteration
            iter = iter + 1;
            disp("Iteration: " + num2str(iter));

            % Best reponse from player 1
            A1bar = [A B2 * K2; L2 * C2 A2bar + B2bar * K2 - L2 * C2bar];
            B1bar = [B1; L2 * D1];
            C1bar = [C1 D2 * K2];
            F1 = [eye(n) zeros(n, p2); zeros(2*(iter - 1)*n, n), L2];
            W1bar = [W zeros(n, p2); zeros(p2, n) V2];
            Q1bar = [Q zeros(n, 2*(iter - 1)*n); zeros(2*(iter - 1)*n, n) K2' * R2 * K2];

            % Compute L1, Sigma1, K1, and P1
            [Sigma1, ~, ~] = idare(A1bar', C1bar', F1 * W1bar * F1', V1, [], []);
            L1 = A1bar * Sigma1 * C1bar' / (V1 + C1bar * Sigma1 * C1bar');

            [P1, ~, ~] = idare(A1bar, B1bar, Q1bar, R1, [], []);

            if isempty(P1)
                disp('Error: please increase the input penalty magnitude for player 2.');
                K1 = [];
                break;
            else
                K1 = -(R1 + B1bar' * P1 * B1bar) \ B1bar' * P1 * A1bar;
            end

            % Closed-loop state and state estimate dynamics of player 1
            A1Bar = [A1bar B1bar * K1; L1 * C1bar A1bar + B1bar * K1 - L1 * C1bar];
            Q1Bar = [Q1bar zeros((2*iter - 1)*n); zeros((2*iter - 1)*n) K1' * R1 * K1];
            F1Bar = [F1 zeros((2*iter - 1)*n, p1); zeros((2*iter - 1)*n, n+p2) L1];
            W1Bar = F1Bar * [W1bar zeros(n+p2, p1); zeros(p1, n+p2) V1] * F1Bar';

            % Optimal cost of player 1
            P1Bar = dlyap(A1Bar', Q1Bar);
            LQG_cost1(iter, 1) = trace(P1Bar * W1Bar);

            % Best response from player 2
            A2bar = [A B1 * K1; L1 * C1 A1bar + B1bar * K1 - L1 * C1bar];
            B2bar = [B2; L1 * D2];
            C2bar = [C2 D1 * K1];
            F2 = [eye(n) zeros(n, p1); zeros((2*iter - 1)*n, n) L1];
            W2bar = [W zeros(n, p1); zeros(p1, n) V1];
            Q2bar = [Q zeros(n, (2*iter - 1)*n); zeros((2*iter - 1)*n, n) K1' * R1 * K1];

            % Compute L2, Sigma2, K2, and P2
            [Sigma2, ~, ~] = idare(A2bar', C2bar', F2 * W2bar * F2', V2, [], []);
            L2 = A2bar * Sigma2 * C2bar' / (V2 + C2bar * Sigma2 * C2bar');

            [P2, ~, ~] = idare(A2bar, B2bar, Q2bar, R2, [], []);
            
            if isempty(P2)                
                disp('Error: please increase the input penalty magnitude for player 2.');
                K2 = [];
                break;
            else
                K2 = -(R2 + B2bar' * P2 * B2bar) \ B2bar' * P2 * A2bar;
            end

            % Closed-loop state and state estimate dynamics of player 2
            A2Bar = [A2bar B2bar * K2; L2 * C2bar A2bar + B2bar * K2 - L2 * C2bar];
            Q2Bar = [Q2bar zeros(2*iter*n); zeros(2*iter*n) K2' * R2 * K2];
            F2Bar = [F2 zeros(2*iter*n, p2); zeros(2*iter*n, n+p1) L2];
            W2Bar = F2Bar * [W2bar zeros(n+p1, p2); zeros(p2, n+p1) V2] * F2Bar';

            % Optimal cost of player 2
            P2Bar = dlyap(A2Bar', Q2Bar);
            LQG_cost2(iter, 1) = trace(P2Bar * W2Bar);

            % Optimal cost difference between player 1 and player 2
            cost_diff = abs(LQG_cost2(iter, 1) - LQG_cost1(iter, 1));            

        end
        
        % Convergence criteria
        if cost_diff < tolerance
            disp("Success: the cost of zero-sum game converges!")
        end
        
        % Output game's optimal value and player 1's LQG cost
        total_cost = LQG_cost2(iter, 1);
        fprintf("The total number of iteration is: %d\n", iter)
        fprintf("The total cost of the zero-sum game is: %f\n", total_cost)
        fprintf("The LQG control cost of player 1 is: %f\n", LQG_cost1(1, 1))

    elseif LQG_player == 2 && max( abs( eig(A) ) ) < 1
        % Compute LQG control cost of player 1
        [Sigma1, ~, ~] = idare(A', C1', W, V1, [], []);
        L1 = A * Sigma1 * C1' / (V1 + C1 * Sigma1 * C1');

        [P1, ~, ~] = idare(A, B1, Q, R1, [], []);
        K1 = -(R1 + B1' * P1 * B1) \ B1' * P1 * A;

        A1Bar = [A B1 * K1; L1 * C1 A + B1 * K1 - L1 * C1];
        Q1Bar = [Q zeros(n); zeros(n) K1' * R1 * K1];
        W1Bar = [W zeros(n); zeros(n) L1 * V1 * L1'];
        
        P1Bar = dlyap(A1Bar', Q1Bar);
        LQG_cost_player1 = trace(P1Bar * W1Bar);
        
        % Best response 1st iteration
        iter = 1;
        disp("Iteration: " + num2str(iter));

        % LQG control of player 2
        [Sigma2, ~, ~] = idare(A', C2', W, V2, [], []);
        L2 = A * Sigma2 * C2' / (V2 + C2 * Sigma2 * C2');

        [P2, ~, ~] = idare(A, B2, Q, R2, [], []);

        if isempty(P2)
            disp('Error: please increase the input penalty magnitude for player 2.');
            disp('Or you can make the system matrix more stable.');
            K2 = [];
        else
            K2 = -(R2 + B2' * P2 * B2) \ B2' * P2 * A;
            
            % Closed-loop state and state estimate dynamics of player 2
            A2Bar = [A B2 * K2; L2 * C2 A + B2 * K2 - L2 * C2];
            Q2Bar = [Q zeros(n); zeros(n) K2' * R2 * K2];
            W2Bar = [W zeros(n); zeros(n) L2 * V2 * L2'];

            % Optimal cost of player 2
            P2Bar = dlyap(A2Bar', Q2Bar);
            LQG_cost2(iter, 1) = trace(P2Bar * W2Bar);

            % Best response from player 1
            A1bar = A2Bar;
            B1bar = [B1; L2 * D1];
            C1bar = [C1 D2 * K2];
            F1 = [eye(n) zeros(n, p2); zeros(n) L2];
            W1bar = [W zeros(n, p2); zeros(p2, n) V2];
            Q1bar = [Q zeros(n); zeros(n) K2' * R2 * K2];

            % Compute L1, Sigma1, K1 and P1
            [Sigma1, ~, ~] = idare(A1bar', C1bar', F1 * W1bar * F1', V1, [], []);
            L1 = A1bar * Sigma1 * C1bar' / (V1 + C1bar * Sigma1 * C1bar');
    
            [P1, ~, ~] = idare(A1bar, B1bar, Q1bar, R1, [], []);

            if isempty(P1)
                disp("Error: player 1's closed-loop system is unstable.");
                disp("Please increasre the input penalty magnitude for player 2");
                K1 = [];
            else
                K1 = -(R1 + B1bar' * P1 * B1bar) \ B1bar' * P1 * A1bar;
                
                % Closed-loop state and state estimate dynamics of player 1
                A1Bar = [A1bar B1bar * K1; L1 * C1bar A1bar + B1bar * K1 - L1 * C1bar];
                Q1Bar = [Q1bar zeros(2*n); zeros(2*n) K1' * R1 * K1];
                F1Bar = [F1 zeros(2*n, p1); zeros(2*n, n+p2) L1];
                W1Bar = F1Bar * [W1bar zeros(n+p2, p1); zeros(p1, n+p2) V1] * F1Bar';
        
                % Optimal cost of player 1
                P1Bar = dlyap(A1Bar', Q1Bar);
                LQG_cost1(iter, 1) = trace(P1Bar * W1Bar);           
            end
        end

        % Optimal cost difference between player 1 and player 2
        cost_diff = abs(LQG_cost2(iter, 1) - LQG_cost1(iter, 1));

        while cost_diff > tolerance && ~isempty(P1) && ~isempty(P2) && iter < max_num_iteration
            iter = iter + 1;
            disp("Iteration: " + num2str(iter));

            % Best response from player 2
            A2bar = [A B1 * K1; L1 * C1 A1bar + B1bar * K1 - L1 * C1bar];
            B2bar = [B2; L1 * D2];
            C2bar = [C2 D1 * K1];
            F2 = [eye(n) zeros(n, p1); zeros(2*(iter - 1)*n, n) L1];
            W2bar = [W zeros(n, p1); zeros(p1, n) V1];
            Q2bar = [Q zeros(n, 2*(iter - 1)*n); zeros(2*(iter - 1)*n, n) K1' * R1 * K1];

            % Compute L2, Sigma2, K2, and P2
            [Sigma2, ~, ~] = idare(A2bar', C2bar', F2 * W2bar * F2', V2, [], []);
            L2 = A2bar * Sigma2 * C2bar' / (V2 + C2bar * Sigma2 * C2bar');

            [P2, ~, ~] = idare(A2bar, B2bar, Q2bar, R2, [], []);
            
            if isempty(P2)
                disp('Error: please increase the input penalty magnitude for player 2.');
                K2 = [];
                break;
            else
                K2 = -(R2 + B2bar' * P2 * B2bar) \ B2bar' * P2 * A2bar;
            end

            % Closed-loop state and state estimate dynamics of player 2
            A2Bar = [A2bar B2bar * K2; L2 * C2bar A2bar + B2bar * K2 - L2 * C2bar];
            Q2Bar = [Q2bar zeros((2*iter - 1)*n); zeros((2*iter - 1)*n) K2' * R2 * K2];
            F2Bar = [F2 zeros((2*iter - 1)*n, p2); zeros((2*iter - 1)*n, n+p1) L2];
            W2Bar = F2Bar*[W2bar zeros(n+p1, p2); zeros(p2, n+p1) V2] * F2Bar';

            % Optimal cost of player 2
            P2Bar = dlyap(A2Bar', Q2Bar);
            LQG_cost2(iter, 1) = trace(P2Bar * W2Bar);

            % Best reponse from player 1
            A1bar = [A B2 * K2; L2 * C2 A2bar + B2bar * K2 - L2 * C2bar];
            B1bar = [B1; L2 * D1];
            C1bar = [C1 D2 * K2];
            F1 = [eye(n) zeros(n, p2); zeros((2*iter - 1)*n, n), L2];
            W1bar = [W zeros(n, p2); zeros(p2, n) V2];
            Q1bar = [Q zeros(n, (2*iter - 1)*n); zeros((2*iter - 1)*n, n) K2' * R2 * K2];

            % Compute L1, Sigma1, K1, and P1
            [Sigma1, ~, ~] = idare(A1bar', C1bar', F1 * W1bar * F1', V1, [], []);
            L1 = A1bar * Sigma1 * C1bar' / (V1 + C1bar * Sigma1 * C1bar');

            [P1, ~, ~] = idare(A1bar, B1bar, Q1bar, R1, [], []);

            if isempty(P1)
                disp("Error: player 1's closed-loop system is unstable.");
                disp("Please increasre the input penalty magnitude for player 2");                
                break;
            else
                K1 = -(R1 + B1bar' * P1 * B1bar) \ B1bar' * P1 * A1bar;
            end

            % Closed-loop state and state estimate dynamics of player 1
            A1Bar = [A1bar B1bar * K1; L1 * C1bar A1bar + B1bar * K1 - L1 * C1bar];
            Q1Bar = [Q1bar zeros(2*iter*n); zeros(2*iter*n) K1' * R1 * K1];
            F1Bar = [F1 zeros(2*iter*n, p1); zeros(2*iter*n, n+p2) L1];
            W1Bar = F1Bar * [W1bar zeros(n+p2, p1); zeros(p1, n+p2) V1] * F1Bar';

            % Optimal cost of player 1
            P1Bar = dlyap(A1Bar', Q1Bar);
            LQG_cost1(iter, 1) = trace(P1Bar * W1Bar);

            % Optimal cost difference between player 1 and player 2
            cost_diff = abs(LQG_cost2(iter, 1) - LQG_cost1(iter, 1));

        end

        % Convergence criteria
        if cost_diff < tolerance
            disp("Success: the cost of zero-sum game converges!")
        end

        total_cost = LQG_cost2(iter, 1);
        fprintf("The total number of iteration is: %d\n", iter)
        fprintf("The total cost of the zero-sum game is: %f\n", total_cost)
        fprintf("The LQG control cost of player 1 is: %f\n", LQG_cost_player1)

    elseif LQG_player == 2 && max( abs( eig(A) ) ) >= 1
        fprintf("Warning: A matrix is not strictly stable. Please select player 1 as the initial LQG control player.\n")
    else
        fprintf("Warning: Please specify the initial LQG control player.\n");
    end

end

