function [y,I] = problem24(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n+1,1);
    
    tmp(1) = abs(2*x(1) + 10/((n+1)^2)*sinh(10*x(1)) - x(2));
    I(1) = sign(2*x(1) + 10/((n+1)^2)*sinh(10*x(1)) - x(2));
    for i = 2:n-1
        tmp(i) = abs(2*x(i) + 10/((n+1)^2)*sinh(10*x(i)) - x(i-1) - x(i+1));
        I(i) = sign(2*x(i) + 10/((n+1)^2)*sinh(10*x(i)) - x(i-1) - x(i+1));
    end
    tmp(n) = abs(2*x(n) + 10/((n+1)^2)*sinh(10*x(n)) - x(n-1) - 1);
    I(n) = sign(2*x(n) + 10/((n+1)^2)*sinh(10*x(n)) - x(n-1) - 1);
    
    
    [y,i] = max(tmp,[],1);
    I(n+1) = i(1);
    
end

