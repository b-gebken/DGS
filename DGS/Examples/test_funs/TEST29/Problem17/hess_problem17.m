function hess = hess_problem17(x)
    
    [~,I] = problem17(x);
    
    n = size(x,1);
    hess = zeros(n);
    
    i_max = I(n+1);
    j = floor((i_max-1)/5);
    hess(i_max,i_max) = I(i_max)*(-(j+1)*cos(x(i_max))+sin(x(i_max)));
    for k = 5*j+1:5*j+5
        hess(k,k) = hess(k,k) + I(i_max)*(-(-cos(x(k))));
    end
    
end

