function [y,I] = problem22(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n+1,1);
    
    tmp(1) = abs(2*x(1) + 1/(2*(n+1)^2)*(x(1) + 1/(n+1) + 1)^3 - x(2));
    I(1) = sign(2*x(1) + 1/(2*(n+1)^2)*(x(1) + 1/(n+1) + 1)^3 - x(2));
    for i = 2:n-1
        tmp(i) = abs(2*x(i) + 1/(2*(n+1)^2)*(x(i) + i/(n+1) + 1)^3 - x(i-1) - x(i+1));
        I(i) = sign(2*x(i) + 1/(2*(n+1)^2)*(x(i) + i/(n+1) + 1)^3 - x(i-1) - x(i+1));
    end
    tmp(n) = abs(2*x(n) + 1/(2*(n+1)^2)*(x(n) + n/(n+1) + 1)^3 - x(n-1));
    I(n) = sign(2*x(n) + 1/(2*(n+1)^2)*(x(n) + n/(n+1) + 1)^3 - x(n-1));
    
    
    [y,i] = max(tmp,[],1);
    I(n+1) = i(1);
    
end

