function grad = grad_chained_CB3_I(x)
    
    [~,I] = chained_CB3_I(x);
    n = size(x,1);
    
    grad = zeros(n,1);
    
    for i = 1:n-1
        if(I(i) == 1)
            grad(i) = grad(i) + 4*x(i)^3;
            grad(i+1) = grad(i+1) + 2*x(i+1);
        elseif(I(i) == 2)
            grad(i) = grad(i) - 2*(2 - x(i));
            grad(i+1) = grad(i+1) - 2*(2 - x(i+1)); 
        else
            grad(i) = grad(i) - 2*exp(-x(i) + x(i+1));
            grad(i+1) = grad(i+1) + 2*exp(-x(i) + x(i+1)); 
        end
    end
    
end

