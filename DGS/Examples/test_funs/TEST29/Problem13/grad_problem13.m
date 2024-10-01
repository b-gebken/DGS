function grad = grad_problem13(x)
        
    n = size(x,1);
    
    [~,I] = problem13(x);
    grad = zeros(n,1);
    
    for k = 1:2*(n-2)
        i = 2*floor((k+3)/4)-2;
        l = mod(k-1,4)+1;
        
        for h = 1:3
            tmp_prod = 1;
            for j = 1:4
                tmp_prod = tmp_prod*sign(x(i+j))*abs(x(i+j))^(j/(h*l));
            end
            
            for j = 1:4
                grad(i+j) = grad(i+j) + I(k) * h^2/l * j/(h*l) * sign(x(i+j)) * tmp_prod/abs(x(i+j));
            end
            
        end
        
    end

end

