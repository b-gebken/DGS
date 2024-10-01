function hess = hess_crescent_I(x)
    
    [~,I] = crescent_I(x);
    
    n = size(x,1);
    if(I(1) == 1)
        hess = diag([2,4*ones(1,n-2),2]);
    else
        hess = diag([-2,-4*ones(1,n-2),-2]); 
    end
    
end

