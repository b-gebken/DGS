function [val,I] = nest_fun(x,m)

[max_val,I] = max(x(1:m,:),[],1);
val = max_val + 1/2*vecnorm(x,2,1).^2;

end

