function grad = grad_maxq(x)
    
    [~,I] = maxq(x);
    n = size(x,1);
    
    grad = zeros(n,1);
    grad(I(1)) = 2*x(I(1)); 
    
end

