function hess = hess_maxq(x)
    
    [~,I] = maxq(x);
    n = size(x,1);
    
    hess = zeros(n);
    hess(I(1),I(1)) = 2; 
    
end

