function [Wpo] = ucz1 (Wprzed, P, T, n)
    liczbaPrzykladow = size(P, 2);
    W = Wprzed;
    wspUcz = 0.1;  beta = 5;

    for i = 1 : n
        %losuj numer przykladu
        nrPrzykladu = randi(liczbaPrzykladow); 

        %podaj przyklad na wejscia i oblicz wyjscia
        X = P (:, nrPrzykladu);
        Y = dzialaj1(W, X);

        %oblicz bledy na wyjsciach
        D = T (:, nrPrzykladu) - Y;
        E = D .* beta .* Y .*(1-Y);
        %oblicz poprawki wag
        dW = wspUcz * X * E'; %co transponowac?

        %dodaj poprawki do wag
        W = W + dW;
    end
    Wpo = W;
end