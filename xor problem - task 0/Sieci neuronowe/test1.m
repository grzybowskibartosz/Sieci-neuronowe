P = [4, 2, -1; 0.01, -1, 3.5; 0.01, 2, 0.01; -1, 2.5, -2; -1.5, 2, 1.5;]
T = [1, 0, 0; 0, 1, 0; 0, 0, 1;]


Wprzed = init1(5, 3)
Yprzed = dzialaj1(Wprzed, P)
Wpo = ucz1(Wprzed, P, T, 100)
Ypo = dzialaj1(Wpo, P)


ja = [2; 0.9; -1; 0; -5]
odp = dzialaj1 (Wpo, ja)
