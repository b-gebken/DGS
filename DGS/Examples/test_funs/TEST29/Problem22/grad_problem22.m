function grad = grad_problem22(x)
    
    [~,I] = problem22(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    if(I(n+1) == 1)
        grad(1) = I(1)*(2 + 3*1/(2*(n+1)^2)*(x(1) + 1/(n+1) + 1)^2);
        grad(2) = I(1)*(-1);
    elseif(I(n+1) == n)
        grad(n) = I(n)*(2 + 3*1/(2*(n+1)^2)*(x(n) + n/(n+1) + 1)^2);
        grad(n-1) = I(n)*(-1);
    else
        grad(I(n+1)-1) = I(I(n+1))*(-1); 
        grad(I(n+1)) = I(I(n+1))*(2 + 3*1/(2*(n+1)^2)*(x(I(n+1)) + I(n+1)/(n+1) + 1)^2);
        grad(I(n+1)+1) = I(I(n+1))*(-1); 
    end
    
end

