function grad = grad_crescent_II(x)
    
    [~,I] = crescent_II(x);
    n = size(x,1);
    
    grad = zeros(n,1);
    for i = 1:n-1
        if(I(i) == 1)
            grad(i) = grad(i) + 2*x(i);
            grad(i+1) = grad(i+1) + 2*(x(i+1) - 1) + 1;
        else
            grad(i) = grad(i) - 2*x(i);
            grad(i+1) = grad(i+1) - 2*(x(i+1) - 1) + 1; 
        end
    end
    
end

