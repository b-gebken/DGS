x_sym = sym('x',[n,1]);

f_sym = sum(abs(x_sym(1:n-1)).^(x_sym(2:n).^2 + 1) + abs(x_sym(2:n)).^(x_sym(1:n-1).^2 + 1),1);
brown_2 =  matlabFunction(f_sym,'Vars',{x_sym(:)});

grad_f_sym = gradient(f_sym,x_sym);
grad_brown_2 = matlabFunction(grad_f_sym,'Vars',{x_sym(:)});

hess_f_sym = hessian(f_sym,x_sym);
hess_brown_2 = matlabFunction(hess_f_sym,'Vars',{x_sym(:)});