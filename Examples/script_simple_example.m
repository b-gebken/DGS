% This script applies the DGS to a simple example.

clear all
rng('default')

% Chained Crescent II
addpath('test_funs/H2004lib/Chained_Crescent_II')
n = 2;
problem_data.n = n;
problem_data.f = @(x) crescent_II(x);
problem_data.subgrad_f = @(x) grad_crescent_II(x);
problem_data.x0 = crescent_II_x0(n);

% Set parameters for DGS 
j_max = 10;
kappa_eps = 0.1;
eps0 = 10;
eps_fun = @(j) eps0*kappa_eps.^(j-1);
del_fun = eps_fun;
dgs_options.eps_arr = eps_fun(1:j_max);
dgs_options.delta_arr = del_fun(1:j_max);

dgs_options.rand_sample_N = 0;
dgs_options.memory_size = 0;
dgs_options.c = 0.5;
dgs_options.ls_flag = 'armijo';
dgs_options.max_iter = 10000;
dgs_options.disp_flag = 2;

%% Run the algorithm
addpath('..')
[x_opt,f_opt,x_cell,eval_counter] = eps_descent_method(problem_data,dgs_options);

%% Process the results
f = problem_data.f;

xj_arr = zeros(n,j_max);
Nj_arr = zeros(1,j_max);
zl_arr = [];
ij_arr = zeros(1,j_max);
for j = 1:j_max
    xj_arr(:,j) = x_cell{j}(:,end);
    Nj_arr(j) = size(x_cell{j},2)-1;
    zl_arr = [zl_arr,x_cell{j}];
    ij_arr(j) = size(zl_arr,2);
end
l_max = size(zl_arr,2);

fzl_arr = zeros(1,l_max);
for l = 1:l_max
    fzl_arr(l) = f(zl_arr(:,l));
end

N_bar = max(Nj_arr);

%% Visualization

p = 2;
M = 10;
x_min = [zeros(n,1)];

sp_size_1 = 2;
sp_size_2 = 4;

figure
subplot(sp_size_1,sp_size_2,1) %%%%%%%%%%%%%%%%%%%%
plot(1:j_max,log10(vecnorm(xj_arr - x_min,2,1)),'k.-','MarkerSize',10);
hold on
h1 = plot(1:j_max,log10(M*dgs_options.eps_arr.^(1/p)),'ro-');
if(p > 1)
    h2 = plot(1:j_max,log10(M*dgs_options.delta_arr.^(1/(p-1))),'bx-');
    legend([h1,h2],{'$M \varepsilon_j^{1/p}$','$M \delta_j^{1/(p-1)}$'},'Interpreter','latex','Location','sw','FontSize',20)
else
    legend([h1],{'$M \varepsilon_j^{1/p}$'},'Interpreter','latex','Location','sw','FontSize',20)
end
grid on
title('$\log_{10}(\| x^j - x^* \|)$','Interpreter','latex','FontSize',20)
xlabel('$j$','Interpreter','latex');

subplot(sp_size_1,sp_size_2,2) %%%%%%%%%%%%%%%%%%%%
plot(1:j_max,Nj_arr,'k.-','MarkerSize',10);
grid on
title('$N_j$','Interpreter','latex','FontSize',20)
xlabel('$j$','Interpreter','latex');

subplot(sp_size_1,sp_size_2,[3,4,7,8]) %%%%%%%%%%%%%%%%%%%%
[X1,X2] = ndgrid(linspace(-2,2,100));
X = [X1(:)';X2(:)'];
fX = zeros(1,size(X,2));
for i = 1:size(X,2)
    fX(i) = f(X(:,i));
end
surf(X1,X2,reshape(fX,size(X1,2),size(X1,2)),'FaceAlpha',0);
hold on
plot3(zl_arr(1,:),zl_arr(2,:),fzl_arr,'r.-','MarkerSize',10,'LineWidth',1.5);
title('Graph','Interpreter','latex','FontSize',20)
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');

subplot(sp_size_1,sp_size_2,5) %%%%%%%%%%%%%%%%%%%%
nonzero_ind = find(ij_arr ~= 0,1,'first');
plot(1:l_max,log10(vecnorm(zl_arr - x_min,2,1)),'k.-');
hold on
plot(ij_arr(nonzero_ind:end),log10(vecnorm(zl_arr(:,ij_arr(nonzero_ind:end)) - x_min,2,1)),'ko')
grid on
title('$\log_{10}(\| z^l - x^* \|)$','Interpreter','latex','FontSize',20)
xlabel('$l$','Interpreter','latex');

subplot(sp_size_1,sp_size_2,6) %%%%%%%%%%%%%%%%%%%%
r = @(l) max([eps_fun(l);del_fun(l)],[],1);
plot(1:l_max,log10(abs(fzl_arr - f(x_min))),'k.-');
hold on
plot(ij_arr(nonzero_ind:end),log10(fzl_arr(ij_arr(nonzero_ind:end)) - f(x_min)),'ko')
grid on
title('$\log_{10}(f(z^l) - f(x^*))$','Interpreter','latex','FontSize',20)
xlabel('$l$','Interpreter','latex');