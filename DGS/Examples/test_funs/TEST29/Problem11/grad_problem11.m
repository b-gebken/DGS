function grad = grad_problem11(x)
    
    [~,I] = problem11(x);
    n = size(x,1);

    grad = zeros(n,1);
    for k = 1:2*(n-1)
        if(mod(k,2) == 0)
            i = k/2;
            grad(i) = grad(i) + I(k)*(1);
            grad(i+1) = grad(i+1) + I(k)*(3*x(i+1)^2 + 2*x(i+1) - 14);
        else
            i = (k+1)/2;
            grad(i) = grad(i) + I(k)*(1);
            grad(i+1) = grad(i+1) + I(k)*(-3*x(i+1)^2 + 10*x(i+1) - 2);
        end
    end
    
end

