function [Y1, Y2] = dzialaj2 (W1, W2, X)

beta = 5;
X1 = [-1; X];
U1 = W1' * X1;
Y1 = 1 ./ (1 + exp (-beta * U1));
X2 = [-1; Y1];
U2 = W2' * X2;
Y2 = 1 ./ (1 + exp(-beta * U2));