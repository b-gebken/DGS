function [y,I] = chained_CB3_II(x)

    [y,I] = max([sum(x(1:end-1).^4 + x(2:end).^2,1);sum((2 - x(1:end-1)).^2 + (2 - x(2:end)).^2,1);sum(2*exp(-x(1:end-1)+x(2:end)),1)],[],1);
    
end

