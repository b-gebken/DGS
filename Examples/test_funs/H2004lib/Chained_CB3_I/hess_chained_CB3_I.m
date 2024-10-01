function hess = hess_chained_CB3_I(x)
    
    [~,I] = chained_CB3_I(x);
    n = size(x,1);
    
    hess = zeros(n);
    
    for i = 1:n-1
        if(I(i) == 1)
            hess(i,i) = hess(i,i) + 12*x(i)^2;
            hess(i+1,i+1) = hess(i+1,i+1) + 2;
        elseif(I(i) == 2)
            hess(i,i) = hess(i,i) + 2;
            hess(i+1,i+1) = hess(i+1,i+1) + 2; 
        else
            hess(i,i) = hess(i,i) + 2*exp(-x(i) + x(i+1));
            hess(i,i+1) = hess(i,i+1) - 2*exp(-x(i) + x(i+1));
            hess(i+1,i) = hess(i+1,i) - 2*exp(-x(i) + x(i+1));
            hess(i+1,i+1) = hess(i+1,i+1) + 2*exp(-x(i) + x(i+1)); 
        end
    end
    
end

