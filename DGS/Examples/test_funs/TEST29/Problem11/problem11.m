function [y,I] = problem11(x)
   
    n = size(x,1);

    I = zeros(2*(n-1),1);
    tmp = 0;
    for k = 1:2*(n-1)
        if(mod(k,2) == 0)
            i = k/2;
            tmp = tmp + abs(x(i) + x(i+1)*((1 + x(i+1))*x(i+1) - 14) - 29);
            I(k) = sign(x(i) + x(i+1)*((1 + x(i+1))*x(i+1) - 14) - 29);
        else
            i = (k+1)/2;
            tmp = tmp + abs(x(i) + x(i+1)*((5 - x(i+1))*x(i+1) - 2) - 13);
            I(k) = sign(x(i) + x(i+1)*((5 - x(i+1))*x(i+1) - 2) - 13);
        end
    end
    
    y = tmp;
end

