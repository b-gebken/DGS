function hess = hess_problem13(x)
        
    n = size(x,1);
    
    [~,I] = problem13(x);
    hess = zeros(n);
    
    for k = 1:2*(n-2)
        i = 2*floor((k+3)/4)-2;
        l = mod(k-1,4)+1;
        
        for h = 1:3
            tmp_prod = 1;
            for j = 1:4
                tmp_prod = tmp_prod*sign(x(i+j))*abs(x(i+j))^(j/(h*l));
            end
            
            for j1 = 1:4
                for j2 = 1:4
                    if(j1 ~= j2)
                        hess(i+j1,i+j2) = hess(i+j1,i+j2) + I(k) * h^2/l * j1/(h*l) * j2/(h*l) * sign(x(i+j1)) * sign(x(i+j2)) * tmp_prod/(abs(x(i+j1)) * abs(x(i+j2)));
                    else
                        hess(i+j1,i+j1) = hess(i+j1,i+j1) + I(k) * h^2/l * j1/(h*l) * (j1/(h*l) - 1) * tmp_prod/(abs(x(i+j1))^2);
                    end
                end
            end
            
        end
        
    end

end

