function hess = hess_problem24(x)
    
    [~,I] = problem24(x);
    
    n = size(x,1);
    hess = zeros(n);
    hess(I(n+1),I(n+1)) = I(I(n+1))*(10*10*10/((n+1)^2)*sinh(10*x(I(n+1))));
    
end

