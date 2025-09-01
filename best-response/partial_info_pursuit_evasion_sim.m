function partial_info_pursuit_evasion_sim(A, B1, B2, C1, C2, D1, D2, A1bar, A2bar, B1bar, B2bar, C1bar, C2bar, W1bar, W2bar, V1, V2, L1, L2, K1, K2, F1, F2, T, x10, x20, X0, z10, z20, show_ellipses, show_time_plots)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % A: system matrix
    % B1: Player 1's control matrix
    % B2: Player 2's control matrix
    % C1: Player 1's measurement matrix
    % C2: Player 2's measurement matrix
    % D1: Player 1's measurement attack matrix
    % D2: Player 2's measurement attack matrix
    % A1bar: Player 1's higher-order system matrix
    % A2bar: Player 2's higher-order system matrix
    % B1bar: Player 1's higher-order control matrix
    % B2bar: Player 2's higher-order control matrix
    % C1bar: Player 1's higher-order measurement matrix
    % C2bar: Player 2's higher-order measurement matrix
    % W1bar: Player 1's noise matrix
    % W2bar: Player 2's noise matrix
    % V1: Player 1's measurement noise covariance matrix
    % V2: Player 2's measurement noise covariance matrix
    % L1: Player 1's optimal state estimator gain
    % L2: Player 2's optimal state estimator gain
    % K1: Player 1's optimal policy gain
    % K2: Player 2's optimal policy gain
    % F1: Player 1's noise scale matrix
    % F2: Player 2's noise scale matrix
    % T: planning horizon
    % x1: Player 1's initial state
    % x2: Player 2's initial state
    % x0: initial mean of state x
    % X0: initial covariance of state x
    % z10: Player 1's initial belief about x
    % z20: Player 2's initial belief about x

    t = 1; % time-step
    n = size(A, 1); % state dimension
    n1 = size(K1, 2); % player 1's belief dimension
    n2 = size(K2, 2); % player 2's belief dimension
    
    x1_traj = zeros(n, T);
    x2_traj = zeros(n, T);
    x_traj = zeros(n, T);
    z1_traj = zeros(n1, T);
    z2_traj = zeros(n2, T);
    z1_est_traj = zeros(n, T);
    z2_est_traj = zeros(n, T);
    Sigma1plus = zeros(n1);
    Sigma2plus = zeros(n2);
    Sigma1_traj = zeros(n1, n1, T);
    Sigma2_traj = zeros(n2, n2, T);
    
    % Actual initial state
    x1 = x10;
    x2 = x20;
    x = x1 - x2;

    % Player 1 and Player 2's initial belief
    z1 = zeros(n1, 1);
    z2 = zeros(n2, 1);
    for i = 1:n:n1
        z1(i:i+n-1) = z10;
        Sigma1plus(i:i+n-1, i:i+n-1) = X0;
    end
    for i = 1:n:n2
        z2(i:i+n-1) = z20;
        Sigma2plus(i:i+n-1, i:i+n-1) = X0;
    end

    x1_traj(:, t) = x1;
    x2_traj(:, t) = x2;
    x_traj(:, t) = x;
    z1_traj(:, t) = z1;
    z2_traj(:, t) = z2;
    z1_est_traj(:, t) = -z1(1:n) + x1;
    z2_est_traj(:, t) = z2(1:n) + x2;
    Sigma1_traj(:, :, t) = Sigma1plus;
    Sigma2_traj(:, :, t) = Sigma2plus;

    % Simulation
    while t < T
        % Player 1 and Player 2's inputs
        u1 = K1(:, 1:n1) * z1;
        u2 = K2(:, 1:n2) * z2;

        % Player 1 and Player 2's state dynamics
        x1 = A * x1 + B1 * u1;
        x2 = A * x2 + B2 * u2;
        x = x1 - x2;

        % Player 1 and Player 2's measurements
        y1 = C1 * x + D2 * u2;
        y2 = C2 * x + D1 * u1;

        % Player 1 and Player 2's belief dynamics
        z1 = A1bar(1:n1, 1:n1) * z1 + B1bar(1:n1, :) * u1 + L1(1:n1, :) * (y1 - C1bar(:, 1:n1) * z1);
        z2 = A2bar(1:n2, 1:n2) * z2 + B2bar(1:n2, :) * u2 + L2(1:n2, :) * (y2 - C2bar(:, 1:n2) * z2);

        % Player 1 and Player 2's posterior covariances
        Sigma1plus = F1(1:n1, :) * W1bar * F1(1:n1, :)' + A1bar(1:n1, 1:n1) * Sigma1plus * A1bar(1:n1, 1:n1)' ...
            - A1bar(1:n1, 1:n1) * Sigma1plus * C1bar(:, 1:n1)' / (V1 + C1bar(:, 1:n1) * Sigma1plus * C1bar(:, 1:n1)') * C1bar(:, 1:n1) * Sigma1plus * A1bar(1:n1, 1:n1)';
        Sigma2plus = F2(1:n2, :) * W2bar * F2(1:n2, :)' + A2bar(1:n2, 1:n2) * Sigma2plus * A2bar(1:n2, 1:n2)' ...
            - A2bar(1:n2, 1:n2) * Sigma2plus * C2bar(:, 1:n2)' / (V2 + C2bar(:, 1:n2) * Sigma2plus * C2bar(:, 1:n2)') * C2bar(:, 1:n2) * Sigma2plus * A2bar(1:n2, 1:n2)';
        
        t = t + 1;
        
        % Save trajectory data
        x1_traj(:, t) = x1;
        x2_traj(:, t) = x2;
        x_traj(:, t) = x;
        z1_traj(:, t) = z1;
        z2_traj(:, t) = z2;
        z1_est_traj(:, t) = -z1(1:n) + x1;
        z2_est_traj(:, t) = z2(1:n) + x2;
        Sigma1_traj(:, :, t) = Sigma1plus;
        Sigma2_traj(:, :, t) = Sigma2plus;
    end
    
    % Plot pursuit-evasion chase trajectories
    time_step = 10;
    plot_chase_trajectory(x1_traj, x2_traj, z1_est_traj, z2_est_traj, Sigma1_traj, Sigma2_traj, time_step, T, show_ellipses)

    % Plot actual state and belief trajectory
    if show_time_plots
        plot_estimation_error_trajectory(n, T, time_step, x_traj, z1_traj, z2_traj, Sigma1_traj, Sigma2_traj)
    end

end


function plot_chase_trajectory(x1_traj, x2_traj, z1_est_traj, z2_est_traj, Sigma1_traj, Sigma2_traj, time_step, T, show_ellipses)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % x1_traj: Player 1's state (mean) trajectory
    % x2_traj: Player 2's state (mean) trajectory
    % z1_est_traj: Player 1's estimate
    % z2_est_traj: Player 2's estimate
    % Sigma1_traj: Player 1's covariance trajectory
    % Sigma2_traj: Player 2's covariance trajectory
    % T: planning horizon

    % Plot pursuit-evasion chase trajectory
    figure
    hold on
    traj_line_width = 1.8;
    plot(x1_traj(1, 1:T), x1_traj(2, 1:T), 'Linewidth', traj_line_width)
    plot(x2_traj(1, 1:T), x2_traj(2, 1:T), 'Linewidth', traj_line_width);
    plot(z1_est_traj(1, 1:T), z1_est_traj(2, 1:T), '--', 'LineWidth', traj_line_width, 'Color', '#0072BD');
    plot(z2_est_traj(1, 1:T), z2_est_traj(2, 1:T), '--', 'LineWidth', traj_line_width, 'Color', '#D95319');
    scatter(x1_traj(1, 1), x1_traj(2, 1), 50, 's', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD'); 
    scatter(x2_traj(1, 1), x2_traj(2, 1), 50, 's', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319');
    scatter(z1_est_traj(1, 1), z1_est_traj(2, 1), 30, '^', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD'); 
    scatter(z2_est_traj(1, 1), z2_est_traj(2, 1), 30, '^', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319');
    scatter(x1_traj(1, T), x1_traj(2, T), 50, 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD');
    scatter(x2_traj(1, T), x2_traj(2, T), 50, 'o', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319');
    
    % Plot standard deviation ellipses
    if show_ellipses
        for i = 1:time_step:T
            k = -pi:0.01:pi;
        
            a1 = sqrt(Sigma1_traj(1, 1, i));
            b1 = sqrt(Sigma1_traj(2, 2, i));
            xc1 = z1_est_traj(1, i);
            yc1 = z1_est_traj(2, i);
            ellip_x1 = xc1 + a1*cos(k);
            ellip_y1 = yc1 + b1*sin(k);
        
            a2 = sqrt(Sigma2_traj(1, 1, i));
            b2 = sqrt(Sigma2_traj(2, 2, i));
            xc2 = z2_est_traj(1, i);
            yc2 = z2_est_traj(2, i);
            ellip_x2 = xc2 + a2*cos(k);
            ellip_y2 = yc2 + b2*sin(k);
        
            % plot(ellip_x1, ellip_y1, 'color', '#0072BD')
            % plot(ellip_x2, ellip_y2, 'color', '#D95319')
            plot(ellip_x1, ellip_y1, 'color', [.5 .5 .5])
            plot(ellip_x2, ellip_y2, 'color', [.5 .5 .5])
        end
    end

    hold off
    title("T = " + T)
    xlabel('X position')
    ylabel('Y position')
    legend('Pursuer', 'Evader', 'Pursuer estimate', 'Evader estimate', 'Location', 'southwest')
    axis equal
    box on
 
end


function plot_estimation_error_trajectory(n, T, time_step, x_traj, z1_traj, z2_traj, Sigma1_traj, Sigma2_traj)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % n: state dimension
    % T: planning horizon
    % x_traj: actual state trajectory
    % z1_traj: Player 1's belief trajectory
    % z2_traj: Player 2's belief trajectory
    % Sigma1_traj: Player 1's covariance trajectory
    % Sigma2_traj: Player 2's covariance trajectory

    % Compute Player 1 and Player 2's estimation error (standard deviation)
    est_err1 = zeros(n, T);
    est_err2 = zeros(n, T);
    for i = 1:n
        for j = 1:T
            est_err1(i, j) = sqrt(Sigma1_traj(i, i, j));
            est_err2(i, j) = sqrt(Sigma2_traj(i, i, j));
        end
    end

    figure
    subplot(4, 1, 1)
    hold on
    plot(0:T-1, x_traj(1, 1:T), 'Color', "#EDB120", 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z1_traj(1, 1:time_step:T), est_err1(1, 1:time_step:T), 'CapSize', 0, 'Color', '#0072BD', 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z2_traj(1, 1:time_step:T), est_err2(1, 1:time_step:T), 'Capsize', 0, 'Color', '#D95319', 'Linewidth', 1.5)
    hold off
    box on
    ylabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 11)
    legend('$x$', "$z^1$", '$z^2$', 'Interpreter', 'latex', 'location', 'southeast')

    subplot(4, 1, 2)
    hold on
    plot(0:T-1, x_traj(2, 1:T), 'Color', "#EDB120", 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z1_traj(2, 1:time_step:T), est_err1(2, 1:time_step:T), 'CapSize', 0, 'Color', '#0072BD', 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z2_traj(2, 1:time_step:T), est_err2(2, 1:time_step:T), 'Capsize', 0, 'Color', '#D95319', 'Linewidth', 1.5)
    hold off
    box on
    ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 11)
    
    subplot(4, 1, 3)
    hold on
    plot(0:T-1, x_traj(3, 1:T), 'Color', "#EDB120", 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z1_traj(3, 1:time_step:T), est_err1(3, 1:time_step:T), 'CapSize', 0, 'Color', '#0072BD', 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z2_traj(3, 1:time_step:T), est_err2(3, 1:time_step:T), 'Capsize', 0, 'Color', '#D95319', 'Linewidth', 1.5)
    hold off
    box on
    ylabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 11)
    
    subplot(4, 1, 4)
    hold on
    plot(0:T-1, x_traj(4, 1:T), 'Color', "#EDB120", 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z1_traj(4, 1:time_step:T), est_err1(4, 1:time_step:T), 'CapSize', 0, 'Color', '#0072BD', 'Linewidth', 1.5)
    errorbar(0:time_step:T-1, z2_traj(4, 1:time_step:T), est_err2(4, 1:time_step:T), 'Capsize', 0, 'Color', '#D95319', 'Linewidth', 1.5)
    hold off
    box on
    xlabel('Time t', 'Interpreter', 'latex', 'FontSize', 11)
    ylabel('$x_4$', 'Interpreter', 'latex', 'FontSize', 11)

end