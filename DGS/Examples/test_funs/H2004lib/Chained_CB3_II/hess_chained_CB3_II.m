function hess = hess_chained_CB3_II(x)
    
    [~,I] = chained_CB3_II(x);
    
    n = size(x,1);
    if(I(1) == 1)
        hess = diag([12*x(1)^2;12*x(2:n-1).^2 + 2;2]);
    elseif(I(1) == 2)
        hess = diag([2;4*ones(n-2,1);2]);
    else
        hess = zeros(n);
        hess(1,1:2) = [2*exp(-x(1) + x(2)),-2*exp(-x(1) + x(2))];
        for j = 2:n-1
            hess(j,j-1:j+1) = [-2*exp(-x(j-1) + x(j)),2*exp(-x(j) + x(j+1)) + 2*exp(-x(j-1) + x(j)), -2*exp(-x(j) + x(j+1))];
        end
        hess(n,n-1:n) = [-2*exp(-x(n-1)+x(n)),2*exp(-x(n-1)+x(n))];  
    end
    
end

