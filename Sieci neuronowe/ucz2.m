function [W1po, W2po] = ucz2(W1przed, W2przed, P, T, n)
liczbaPrzykladow = size(P, 2);
W1 = W1przed;
W2 = W2przed;
wspUcz = 0.1;
beta = 5;
for i = 1 : n ,
    nrPrzykladu = randi(liczbaPrzykladow);
    X = P (: , nrPrzykladu);
    X1 = [-1 ; X];
    [Y1, Y2] = dzialaj2(W1, W2, X);
    X2 = [-1 ; Y1];
    D2 = T(:, nrPrzykladu) - Y2;
    E2 = beta * D2 .* Y2 .* (1- Y2);
    D1 = W2 (2:end, :) * D2;
    

    E1 = beta * D1 .* Y1 .* (1 - Y1);
    dW1 = wspUcz * X1 * E1';
    dW2 = wspUcz * X2 * E2';
    W1 = W1 + dW1;
    W2 = W2 + dW2;
end
W1po = W1;
W2po = W2;