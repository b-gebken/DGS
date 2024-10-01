function hess = hess_active_faces(x)
    
    [~,I] = active_faces(x);
    
    n = size(x,1);
    if(I(1) == 1)
        hess = -1/(abs(-sum(x,1)) + 1)^2*ones(n);
    else
        hess = zeros(n);
        hess(I(1)-1,I(1)-1) = -1/(abs(x(I(1)-1)) + 1)^2;
    end
    
end

