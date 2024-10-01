function grad = grad_crescent_I(x)
    
    [~,I] = crescent_I(x);
    
    n = size(x,1);
    if(I(1) == 1)
        grad = [2*x(1);
            2*x(2:n-1) + 2*(x(2:n-1) - 1) + 1;
            2*(x(n) - 1) + 1];
    else
        grad = [-2*x(1);
            -2*x(2:n-1) - 2*(x(2:n-1) - 1) + 1;
            -2*(x(n) - 1) + 1];
    end
    
end

