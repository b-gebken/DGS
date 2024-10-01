function x0 = problem13_x0(n)

    x0 = zeros(n,1);
    x0(mod(1:n,4) == 0) = 0.8;
    x0(mod(1:n,4) == 1) = -0.8;
    x0(mod(1:n,4) == 2) = 1.2;
    x0(mod(1:n,4) == 3) = -1.2;

end

