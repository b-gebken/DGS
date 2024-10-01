function grad = grad_problem17(x)
    
    [~,I] = problem17(x);
    
    n = size(x,1);
    grad = zeros(n,1);
    
    i_max = I(n+1);
    j = floor((i_max-1)/5);
    grad(i_max) = I(i_max)*(-(j+1)*sin(x(i_max))-cos(x(i_max)));
    for k = 5*j+1:5*j+5
        grad(k) = grad(k) + I(i_max)*(-(-sin(x(k))));
    end
    
end

