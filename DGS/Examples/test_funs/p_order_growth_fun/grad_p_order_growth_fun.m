function grad = grad_Ex6(x,p)

% global x_arr_glob
% x_arr_glob = [x_arr_glob,x];

grad = zeros(2,1);
if(abs(x(1))^p >= abs(x(2)))
    grad(1) = sign(x(1))*abs(x(1))^(p-1);
else
    grad(2) = sign(x(2));
end

end

