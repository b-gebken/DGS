function hess = hess_problem19(x)
    
    [~,I] = problem19(x);
    
    n = size(x,1);
    hess = zeros(n);
    
    if(I == 1)
        hess(1,1) = 2*(24*x(1)^2 - 36*x(1) + 8*x(2) + 5);
        hess(1,2) = 4*(4*x(1) - 3);
        hess(2,1) = hess(1,2);
        hess(2,2) = 8;
    elseif(I == n)
        hess(n-1,n-1) = 2;
        hess(n,n-1) = 8*x(n) - 6;
        hess(n-1,n) = hess(n,n-1);
        hess(n,n) = 48*x(n)^2 - 72*x(n) + 8*x(n-1) + 10;
    else
        hess(I-1,I-1) = 2;
        hess(I-1,I) = 8*x(I) - 6;
        hess(I,I-1) = hess(I-1,I);
        hess(I-1,I+1) = 4;
        hess(I+1,I-1) = hess(I-1,I+1);
        
        hess(I,I) = 2*(24*x(I)^2 - 36*x(I) + 4*x(I-1) + 8*x(I+1) + 5);
        hess(I,I+1) = 4*(4*x(I) - 3);
        hess(I+1,I) = hess(I,I+1);
        
        hess(I+1,I+1) = 8;
    end
    
end

