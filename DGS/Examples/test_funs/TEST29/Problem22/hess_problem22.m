function hess = hess_problem22(x)
    
    [~,I] = problem22(x);
    
    n = size(x,1);
    hess = zeros(n);
    hess(I(n+1),I(n+1)) = I(I(n+1))*(6*1/(2*(n+1)^2)*(x(I(n+1)) + I(n+1)/(n+1) + 1));
    
end

