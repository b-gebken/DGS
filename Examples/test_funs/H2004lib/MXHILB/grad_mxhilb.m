function grad = grad_mxhilb(x)
    
    [~,I] = mxhilb(x);
    n = size(x,1);
    
    grad = sign(I(1)) * 1./(abs(I(1)) + (1:n)' - 1);
    
end

