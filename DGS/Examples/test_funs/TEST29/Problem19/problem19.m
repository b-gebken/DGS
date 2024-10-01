function [y,I] = problem19(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    
    tmp(1) = ((3 - 2*x(1))*x(1) - 2*x(2) + 1)^2;
    for i = 2:n-1   
        tmp(i) = ((3 - 2*x(i))*x(i) - x(i-1) - 2*x(i+1) + 1)^2;
    end
    tmp(n) = ((3 - 2*x(n))*x(n) - x(n-1) + 1)^2;
    
    [y,i] = max(tmp,[],1);
    I = i(1);
    
end

