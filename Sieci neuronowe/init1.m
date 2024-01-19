function [W] = init1 (S, K)
    % S - LICZBA WEJSC,
    % K - LICZBA NEURONOW
    W = rand(S, K) * 0.2 - 0.1;
end




