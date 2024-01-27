function [Y] = dzialaj1(W, X)
    beta = 5;
    U = W' * X;  %co transponowac?
    Y = 1 ./ (1 + exp(-beta * U));
end