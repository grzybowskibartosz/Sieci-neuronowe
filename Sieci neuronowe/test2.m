P = [ 0 0 1 1
      0 1 0 1];
T = [ 0 1 1 0];

[W1przed, W2przed] = init2(2, 2, 1)

[Y1, Y2a] = dzialaj2(W1przed, W2przed, P(:,1));
[Y1, Y2b] = dzialaj2(W1przed, W2przed, P(:,2));
[Y1, Y2c] = dzialaj2(W1przed, W2przed, P(:,3));
[Y1, Y2d] = dzialaj2(W1przed, W2przed, P(:,4));

Yprzed = [Y2a, Y2b, Y2c, Y2d]

[W1po, W2po] = ucz2(W1przed, W2przed, P, T, 5000);

[Y1, Y2a] = dzialaj2(W1po, W2po, P(:,1));
[Y1, Y2b] = dzialaj2(W1po, W2po, P(:,2));
[Y1, Y2c] = dzialaj2(W1po, W2po, P(:,3));
[Y1, Y2d] = dzialaj2(W1po, W2po, P(:,4));

Ypo = [Y2a, Y2b, Y2c, Y2d]