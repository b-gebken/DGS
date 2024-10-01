function [y,I] = problem17(x)
   
    n = size(x,1);
    tmp = zeros(n,1);
    I = zeros(n+1,1);
    
    for i = 1:n
        j = floor((i-1)/5);
               
        tmp(i) = abs(5 - (j+1)*(1 - cos(x(i))) - sin(x(i)) - sum(cos(x(5*j+1:5*j+5))));
        I(i) = sign(5 - (j+1)*(1 - cos(x(i))) - sin(x(i)) - sum(cos(x(5*j+1:5*j+5))));
    end
    
    [y,i] = max(tmp,[],1);
    I(n+1) = i(1);
    
end

