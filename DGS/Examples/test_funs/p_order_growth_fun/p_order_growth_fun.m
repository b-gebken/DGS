function y = Ex6(x,p)

% global x_arr_glob
% x_arr_glob = [x_arr_glob,x];

y = max([1/p*abs(x(1,:)).^p;abs(x(2,:))],[],1);

end

