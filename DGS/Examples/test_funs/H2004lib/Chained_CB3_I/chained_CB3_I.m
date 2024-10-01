function [y,I] = chained_CB3_I(x)

    n = size(x,1);
    tmp = zeros(n-1,1);
    I = zeros(n-1,1);
    for i = 1:n-1
        [tmp_val,tmp_I] = max([x(i)^4 + x(i+1)^2;
            (2 - x(i))^2 + (2 - x(i+1))^2;
            2*exp(-x(i) + x(i+1))],[],1);
        tmp(i) = tmp_val;
        I(i) = tmp_I(1);
    end

    y = sum(tmp,1);
    
end

