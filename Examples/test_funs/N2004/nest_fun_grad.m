function grad = nest_fun_grad(x,m)

[~,I] = nest_fun(x,m);

tmp = zeros(size(x,1),1);
tmp(I) = 1;
grad = tmp + x;

end

