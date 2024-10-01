function [y,I] = active_faces(x)

    g = @(in) log(abs(in) + 1); 
    tmp = [g(-sum(x,1));g(x)];
    
    [y,I] = max(tmp,[],1);
    
end

