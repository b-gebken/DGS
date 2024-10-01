x_sym = sym('x',[n,1]);

f_sym = sum(-x_sym(1:n-1) + 2*(x_sym(1:n-1).^2 + x_sym(2:n).^2 - 1) + 1.75*abs(x_sym(1:n-1).^2 + x_sym(2:n).^2 - 1),1);
mifflin_2 =  matlabFunction(f_sym,'Vars',{x_sym(:)});

grad_f_sym = gradient(f_sym,x_sym);
grad_mifflin_2 = matlabFunction(grad_f_sym,'Vars',{x_sym(:)});

hess_f_sym = hessian(f_sym,x_sym);

c = cell(1,n-1);
for i = 1:n-1
    c{i} = dirac(x_sym(i)^2 + x_sym(i+1)^2 - 1);
end

hess_f_sym = subs(hess_f_sym,c,zeros(1,n-1));

hess_mifflin_2 = matlabFunction(hess_f_sym,'Vars',{x_sym(:)});