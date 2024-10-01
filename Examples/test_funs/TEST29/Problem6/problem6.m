function [y,I] = problem6(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n+1,1);
    
    tmp(1) = abs((3-2*x(1))*x(1) + 1 - x(2));
    I(1) = sign((3-2*x(1))*x(1) + 1 - x(2));
    for i = 2:n-1
        tmp(i) = abs((3-2*x(i))*x(i) + 1 - x(i-1) - x(i+1));
        I(i) = sign((3-2*x(i))*x(i) + 1 - x(i-1) - x(i+1));
    end
    tmp(n) = abs((3-2*x(n))*x(n) + 1 - x(n-1));
    I(n) = sign((3-2*x(n))*x(n) + 1 - x(n-1));
    
    
    [y,i] = max(tmp,[],1);
    I(n+1) = i(1);
    
end

