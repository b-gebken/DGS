function x0 = problem2_x0(n)

    x0 = zeros(n,1);
    x0(1:n/2) = 1:n/2;
    x0((n/2+1):n) = -((n/2+1):n);

end

