function grad = grad_active_faces(x)
    
    [~,I] = active_faces(x);
    
    n = size(x,1);
    if(I(1) == 1)
        s = sign(-sum(x,1));
        grad = -s*1/(abs(-sum(x,1))+1)*ones(n,1);
    else
        grad = zeros(n,1);
        grad(I(1)-1) = sign(x(I(1)-1))*1/(abs(x(I(1)-1))+1);
    end
    
end

