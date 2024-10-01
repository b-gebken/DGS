function hess = hess_crescent_II(x)
    
    [~,I] = crescent_II(x);
    n = size(x,1);
    
    hess = zeros(n);
    
    for i = 1:n-1
        if(I(i) == 1)
            hess(i,i) = hess(i,i) + 2;
            hess(i+1,i+1) = hess(i+1,i+1) + 2;
        else
            hess(i,i) = hess(i,i) - 2;
            hess(i+1,i+1) = hess(i+1,i+1) - 2; 
        end
    end
    
end

