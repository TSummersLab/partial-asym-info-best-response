% An example of a two-player linear quadratic pursuit-evasion game where
% players have perfect information about the states.

%% Problem Data Definition
clear; close all; clc;

% State and input data
n = 4;  % number of states
m1 = 2; % number of inputs of player 1
m2 = 2; % number of inputs of player 2

% System matrix (discretized double-integrator)
delt = 0.1;
A = [1, 0, delt, 0;
     0, 1, 0, delt;
     0, 0, 1, 0;
     0, 0, 0, 1];

% Input matrices
B1 = [0, 0;
      0, 0;
      delt, 0;
      0, delt];
B2 = [0, 0;
      0, 0;
      delt, 0;
      0, delt];

% Noise matrix
G = [0, 0;
     0, 0;
     1, 0;
     0, 1];

% State penalty
q = 1e-2;
Q = q * eye(n);

% Input penalties
r1 = 1;
r2 = -10;
R1 = r1 * eye(m1);
R2 = r2 * eye(m2);

% Process and measurement noise covariances
w1 = 1;
w2 = 1;
W1 = w1^2 * eye(m1);
W2 = w2^2 * eye(m2);
W = G * (W1 + W2) * G';

%% Zero-sum Game with Perfect Information
tol = 1e-6;
max_iter = 1e3;
[K1, K2, total_cost] = value_iteration_zero_sum(A, B1, -B2, Q, R1, R2, W, tol, max_iter);

%% Pursuit-evasion Simulation
T = 150; % time horizon
t = 1; % time step
cap_dist = 1; % capture distance;

% State and input trajectories
x1_traj = zeros(n, T);
x2_traj = zeros(n, T);
u1_traj = zeros(m1, T);
u2_traj = zeros(m2, T);

% Initial conditions
x10 = [0; 0; 0; 0];
x20 = [100; 0; 0; 0];
x1 = x10;
x2 = x20;
x = x1 - x2;

x1_traj(:, t) = x1;
x2_traj(:, t) = x2;

while t <= T
    % Player 1 and 2's inputs
    u1 = K1 * x;
    u2 = K2 * x;

    % Player 1 and 2's (mean) state and covariance propagation
    x1 = A * x1 + B1 * u1;
    x2 = A * x2 + B2 * u2;
    x = x1 - x2;

    % Terminate the game if player 2's is within player 1's capture zone
    if sqrt(x(1)^2 + x(2)^2) <= cap_dist
        disp("The evader is captured by the pursuer.");
        break;
    end

    t = t + 1;

    x1_traj(:, t) = x1;
    x2_traj(:, t) = x2;
    u1_traj(:, t) = u1;
    u2_traj(:, t) = u2;

end

% Plot the chase trajectories
figure
hold on;
box on;
plot(x1_traj(1, 1:t), x1_traj(2, 1:t), 'Linewidth', 1.8);
plot(x2_traj(1, 1:t), x2_traj(2, 1:t), 'Linewidth', 1.8);
scatter(x1_traj(1, 1), x1_traj(2, 1), 50, 's', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD'); 
scatter(x2_traj(1, 1), x2_traj(2, 1), 50, 's', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319');
scatter(x1_traj(1, t), x1_traj(2, t), 50, 'o', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', '#0072BD');
scatter(x2_traj(1, t), x2_traj(2, t), 50, 'o', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319');
hold off;
xlabel('X position', 'FontSize', 15);
ylabel('Y position', 'FontSize', 15);
legend('Pursuer', 'Evader', 'Fontsize', 15);
axis equal;
title("T = " + t);

%% Value Iteration
function [K1, K2, total_cost] = value_iteration_zero_sum(A, B1, B2, Q, R1, R2, W, tol, max_iter)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % A: system matrix
    % B1: player 1's input matrix
    % B2: player 2's input matrix
    % Q: state penalty
    % R1: player 1's input penalty
    % R2: player 2's input penalty
    % tol: value iteration convergence tolerance
    % max_iter: maximum number of iteration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % K1: player 1's policy gain
    % K2: player 2's policy gain
    % total_cost: game's total cost

    % Parameters definition
    n = size(A, 1);
    P = Q;
    Plast = 1e4 * eye(n);
    norm_error = norm(P - Plast);
    
    iter = 1;
    
    % Value iteration
    while norm_error > tol
        Plast = P;

        % Zero-sum case
        K1 = -(R1 + B1'*P*B1 - B1'*P*B2 * ((R2 + B2'*P*B2) \ B2'*P*B1)) \ (B1'*P*A - B1'*P*B2 * ((R2 + B2'*P*B2) \ B2'*P*A));
        K2 = -(R2 + B2'*P*B2 - B2'*P*B1 * ((R1 + B1'*P*B1) \ B1'*P*B2)) \ (B2'*P*A - B2'*P*B1 * ((R1 + B1'*P*B1) \ B1'*P*A));

        P = A'*P*A + Q - [A'*P*B1 A'*P*B2] * ([R1 + B1'*P*B1 B1'*P*B2; B2'*P*B1 R2 + B2'*P*B2] \ [B1'*P*A; B2'*P*A]);

        norm_error = norm(P - Plast);
        
        % Check the conditions that ensure the existence of the unique
        % solution
        if max(eig(R2 + B2'*P*B2)) > 0
            disp('Error: R2 needs to be sufficiently negative definite.')
            break
        elseif max(abs(eig(A + B1 * K1 + B2 * K2))) > 1
            disp('Error: the closed-loop system of the zero-sum game is unstable.')
            break
        elseif iter > max_iter
            disp('Warning: the value iteration algorithm does not converge.');
            break;
        end

        iter = iter + 1;

    end
    
    total_cost = trace(P * W);
end