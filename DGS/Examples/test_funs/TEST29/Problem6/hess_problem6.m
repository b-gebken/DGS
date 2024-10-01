function hess = hess_problem6(x)
    
    [~,I] = problem6(x);
    
    n = size(x,1);
    hess = zeros(n);
    if(I(n+1) == 1)
        hess(1,1) = I(1)*(-4);
    elseif(I(n+1) == n)
        hess(n,n) = I(n)*(-4);
    else
        hess(I(n+1),I(n+1)) = I(I(n+1))*(-4);
    end
    
end

