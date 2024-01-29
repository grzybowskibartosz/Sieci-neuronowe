function [W1, W2]  = init2 (S, K1, K2)

W1 = rand(S+1, K1) * 0.2 - 0.1;
W2 = rand(K1 + 1, K2) * 0.2 - 0.1;