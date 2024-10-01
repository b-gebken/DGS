function hess = hess_problem11(x)
    
    [~,I] = problem11(x);
    n = size(x,1);

    hess = zeros(n);
    for k = 1:2*(n-1)
        if(mod(k,2) == 0)
            i = k/2;
            hess(i+1,i+1) = hess(i+1,i+1) + I(k)*(6*x(i+1) + 2);
        else
            i = (k+1)/2;
            hess(i+1,i+1) = hess(i+1,i+1) + I(k)*(-6*x(i+1) + 10);
        end
    end
    
end

