function hess = hess_problem20(x)
    
    [~,I] = problem20(x);
    
    n = size(x,1);
    hess = zeros(n);
    hess(I(n+1),I(n+1)) = I(I(n+1));
    
end

