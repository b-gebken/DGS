function grad = grad_chained_CB3_II(x)
    
    [~,I] = chained_CB3_II(x);

    if(I(1) == 1)
        grad = [4*x(1)^3;
            4*x(2:end-1).^3 + 2*x(2:end-1);
            2*x(end)];
    elseif(I(1) == 2)
        grad = [-2*(2 - x(1));
            -4*(2 - x(2:end-1));
            -2*(2 - x(end))];
    else
        grad = [-2*exp(-x(1) + x(2));
            -2*exp(-x(2:end-1) + x(3:end)) + 2*exp(-x(1:end-2) + x(2:end-1));
            2*exp(-x(end-1)+x(end))];
    end
    
end

