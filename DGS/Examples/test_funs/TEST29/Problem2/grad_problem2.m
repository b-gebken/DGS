function grad = grad_problem2(x)
    
    [~,I] = problem2(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    
    grad(I) = sign(x(I));
    
end

