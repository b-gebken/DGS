function grad = grad_problem19(x)
    
    [~,I] = problem19(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    
    if(I == 1)
        grad(1) = 2*((3 - 2*x(1))*x(1) - 2*x(2) + 1)*(3 - 4*x(1));
        grad(2) = 2*((3 - 2*x(1))*x(1) - 2*x(2) + 1)*(-2);
    elseif(I == n)
        grad(n-1) =  2*((3 - 2*x(n))*x(n) - x(n-1) + 1)*(-1);
        grad(n) = 2*((3 - 2*x(n))*x(n) - x(n-1) + 1)*(3 - 4*x(n));
    else
        grad(I-1) = 2*((3 - 2*x(I))*x(I) - x(I-1) - 2*x(I+1) + 1)*(-1);
        grad(I) = 2*((3 - 2*x(I))*x(I) - x(I-1) - 2*x(I+1) + 1)*(3 - 4*x(I));
        grad(I+1) = 2*((3 - 2*x(I))*x(I) - x(I-1) - 2*x(I+1) + 1)*(-2);
    end
    
    
end

