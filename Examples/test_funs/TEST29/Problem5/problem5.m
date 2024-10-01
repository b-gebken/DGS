function [y,I] = problem5(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n,1);
    for i = 1:n
        tmp(i) = abs(sum(x./(i+(1:n)'-1),1));
        I(i) = sign(sum(x./(i+(1:n)'-1),1));
    end

    y = sum(tmp,1);
    
end

