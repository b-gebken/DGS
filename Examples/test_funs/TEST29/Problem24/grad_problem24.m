function grad = grad_problem24(x)
    
    [~,I] = problem24(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    if(I(n+1) == 1)
        grad(1) = I(1)*(2 + 10*10/((n+1)^2)*cosh(10*x(1)));
        grad(2) = I(1)*(-1);
    elseif(I(n+1) == n)
        grad(n) = I(n)*(2 + 10*10/((n+1)^2)*cosh(10*x(n)));
        grad(n-1) = I(n)*(-1);
    else
        grad(I(n+1)-1) = I(I(n+1))*(-1); 
        grad(I(n+1)) = I(I(n+1))*(2 + 10*10/((n+1)^2)*cosh(10*x(I(n+1))));
        grad(I(n+1)+1) = I(I(n+1))*(-1); 
    end
    
end

