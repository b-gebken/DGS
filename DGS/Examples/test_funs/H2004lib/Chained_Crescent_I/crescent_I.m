function [y,I] = crescent_I(x)

    n = size(x,1);
    tmp1 = sum(x(1:n-1).^2 + (x(2:n) - 1).^2 + x(2:n) - 1,1); 
    tmp2 = sum(-x(1:n-1).^2 - (x(2:n) - 1).^2 + x(2:n) + 1,1);
    tmp = [tmp1;tmp2];
    
    [y,I] = max(tmp,[],1);
    
end

