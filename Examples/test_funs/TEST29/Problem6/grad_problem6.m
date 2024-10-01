function grad = grad_problem6(x)
    
    [~,I] = problem6(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    if(I(n+1) == 1)
        grad(1) = I(1)*(3 - 4*x(1));
        grad(2) = I(1)*(-1);
    elseif(I(n+1) == n)
        grad(n) = I(n)*(3 - 4*x(n));
        grad(n-1) = I(n)*(-1);
    else
        grad(I(n+1)-1) = I(I(n+1))*(-1); 
        grad(I(n+1)) = I(I(n+1))*(3 - 4*x(I(n+1)));
        grad(I(n+1)+1) = I(I(n+1))*(-1); 
    end
    
end

