function grad = grad_problem5(x)
    
    [~,I] = problem5(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    
    for i = 1:n
        grad = grad + I(i)*1./(i+(1:n)'-1);
    end
    
end

