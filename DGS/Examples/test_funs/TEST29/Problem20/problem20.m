function [y,I] = problem20(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n+1,1);
    
    tmp(1) = abs((0.5*x(1) - 3)*x(1) - 1 + 2*x(2));
    I(1) = sign((0.5*x(1) - 3)*x(1) - 1 + 2*x(2));
    for i = 2:n-1
        tmp(i) = abs((0.5*x(i) - 3)*x(i) - 1 + x(i-1) + 2*x(i+1));
        I(i) = sign((0.5*x(i) - 3)*x(i) - 1 + x(i-1) + 2*x(i+1));
    end
    tmp(n) = abs((0.5*x(n) - 3)*x(n) - 1 + x(n-1));
    I(n) = sign((0.5*x(n) - 3)*x(n) - 1 + x(n-1));
    
    
    [y,i] = max(tmp,[],1);
    I(n+1) = i(1);
    
end

