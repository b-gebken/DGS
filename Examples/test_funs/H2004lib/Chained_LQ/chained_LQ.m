function [y,I] = chained_LQ(x)

    n = size(x,1);
    tmp = zeros(n-1,1);
    I = zeros(n-1,1);
    for i = 1:n-1
        [tmp_val,tmp_I] = max([-x(i)-x(i+1); -x(i) - x(i+1) + (x(i)^2 + x(i+1)^2 - 1)],[],1);
        tmp(i) = tmp_val;
        I(i) = tmp_I(1);
    end

    y = sum(tmp,1);
    
end

