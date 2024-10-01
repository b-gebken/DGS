function [y,I] = maxq(x)

    [y,I] = max(x.^2,[],1);
    
end

