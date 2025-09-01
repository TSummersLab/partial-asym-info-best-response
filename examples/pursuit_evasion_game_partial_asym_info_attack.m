% An example to test the best response algorithm for a two-player linear 
% quadratic pursuit-evasion game with partial and asymmetric information
% and each player's measurement can be attacked.

% Ensure the current directory is under "../examples"
addpath(fullfile(pwd, '..', 'best-response'));

%% Problem Data Definition
clear; close all;

% State and input data
n = 4;  % number of states
m1 = 4; % number of inputs of player 1
m2 = 4; % number of inputs of player 2
p1 = 2; % number of outputs of player 1
p2 = 2; % number of outputs of player 2

% System matrix (discretized double-integrator)
delt = 0.1;
A = [1, 0, delt, 0;
     0, 1, 0, delt;
     0, 0, 1, 0;
     0, 0, 0, 1];

% Input matrices
B1 = [0, 0, 0, 0;
      0, 0, 0, 0;
      delt, 0, 0, 0;
      0, delt, 0, 0];
B2 = [0, 0, 0, 0;
      0, 0, 0, 0;
      delt, 0, 0, 0;
      0, delt, 0, 0];

% Output matrices
C1 = [1, 0, 0, 0;
      0, 1, 0, 0];
C2 = [1, 0, 0, 0;
      0, 1, 0, 0];

% Feedthrough matrices
D1 = zeros(p2, m1);
D2 = [0, 0, 1, 0;
      0, 0, 0, 1];

% Noise matrix
G = [0, 0, 0, 0;
     0, 0, 0, 0;
     1, 0, 0, 0;
     0, 1, 0, 0];

% State penalty
q = 1e-2;
Q = q * eye(n);

% Input penalties
r1 = 1;
r2 = -10;
R1 = r1 * eye(m1);
R2 = [r2, 0, 0, 0;
      0, r2, 0, 0;
      0, 0, r2, 0;
      0, 0, 0, -0.6];

% Process and measurement noise covariances
w1 = 1;
w2 = 1;
v1 = 10;
v2 = 10;
W1 = w1^2 * eye(m1);
W2 = w2^2 * eye(m2);
W = G * (W1 + W2) * G';
V1 = v1^2 * eye(p1);
V2 = v2^2 * eye(p2);

% Best response algorithm parameters
LQG_player = 1; % select the player does LQG control at iteration 1
max_iter = 20;
tol = 1e-3;

%% Best Response Dynamics
[K1, K2, L1, L2, P1, P2, Sigma1, Sigma2, A1bar, A2bar, B1bar, B2bar, C1bar, C2bar, F1, F2, W1bar, W2bar, total_cost, LQG_cost1, LQG_cost2, num_iter] = best_response_zero_sum(A, B1, -B2, C1, C2, D1, D2, Q, R1, R2, W, V1, V2, LQG_player, tol, max_iter);

%% Plot Best Response Convergence Results
figure
hold on;
box on;
plot(1:num_iter, LQG_cost1(1:num_iter, :), 'LineWidth', 2.0)
plot(1:num_iter, LQG_cost2(1:num_iter, :), 'LineWidth', 2.0)
hold off
xlabel('Number of iterations', 'FontSize', 15)
ylabel("Players' optimal cost", 'FontSize', 15)
legend('Pursuer', 'Evader', 'Location', 'northeast', 'Fontsize', 15)
xticks(1:num_iter)

%% Pursuit-evasion Simulation
T = 150; % time horizon
x10 = [0; 0; 0; 0];   % pursuer's actual states at time 0
x20 = [100; 0; 0; 0]; % evader's actual states at time 0
z10 = [-80; 20; 0; -25]; % pursuer's initial belief about x10 - x20
z20 = [-100; 20; 0; 0];  % evader's initial belief about x10 - x20;
X0 = 100 * eye(n);
show_ellipses = false; % show position covariance ellipses
show_time_plots = false; % show estimation and estimation erros in time series
partial_info_pursuit_evasion_sim(A, B1, B2, C1, C2, D1, D2, A1bar, A2bar, B1bar, B2bar, C1bar, C2bar, W1bar, W2bar, V1, V2, L1, L2, K1, K2, F1, F2, T, x10, x20, X0, z10, z20, show_ellipses, show_time_plots)
