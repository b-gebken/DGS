function grad = grad_problem20(x)
    
    [~,I] = problem20(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    if(I(n+1) == 1)
        grad(1) = I(1)*(x(1) - 3);
        grad(2) = I(1)*2;
    elseif(I(n+1) == n)
        grad(n) = I(n)*(x(n) - 3);
        grad(n-1) = I(n)*1;
    else
        grad(I(n+1)-1) = I(I(n+1))*1; 
        grad(I(n+1)) = I(I(n+1))*(x(I(n+1)) - 3);
        grad(I(n+1)+1) = I(I(n+1))*2; 
    end
    
end

