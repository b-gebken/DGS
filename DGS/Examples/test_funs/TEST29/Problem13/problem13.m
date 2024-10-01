function [y,I] = problem13(x)
        
    n = size(x,1);
    I = zeros(2*(n-2),1);
    y_const = [-14.4,-6.8,-4.2,-3.2];
    
    y = 0;
    for k = 1:2*(n-2)
        i = 2*floor((k+3)/4)-2;
        l = mod(k-1,4)+1;
        
        tmp_sum = 0;
        for h = 1:3
            tmp_prod = 1;
            for j = 1:4
                tmp_prod = tmp_prod*sign(x(i+j))*abs(x(i+j))^(j/(h*l));
            end
            
            tmp_sum = tmp_sum + h^2/l * tmp_prod;
        end
        
        I(k) = sign(y_const(l) + tmp_sum);
        y = y + abs(y_const(l) + tmp_sum);
    end

end

