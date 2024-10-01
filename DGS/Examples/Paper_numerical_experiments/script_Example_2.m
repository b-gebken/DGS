% This script reproduces the numerical results shown in the second example
% in [G2024a], Section 6.

clear all
rng('default')

% Define the problem (MAXQ)
addpath('../test_funs/H2004lib/MAXQ')
n = 10;
problem_data.n = n;
problem_data.f = @(x) maxq(x);
problem_data.subgrad_f = @(x) grad_maxq(x);
problem_data.x0 = maxq_x0(n);
x_min = zeros(n,1);

% Set parameters for DGS
kappa_eps = 0.5;
eps0 = 10;
kappa_del = kappa_eps;
del0 = eps0;
j_max = 40;
eps_fun = @(j) eps0*kappa_eps.^(j);
del_fun = @(j) del0*kappa_del.^(j);
dgs_options.eps_arr = eps_fun(0:j_max-1);
dgs_options.delta_arr = del_fun(0:j_max-1);

dgs_options.rand_sample_N = 100;
dgs_options.memory_size = 0;
dgs_options.c = 0.9;
dgs_options.ls_flag = 'armijo';
dgs_options.max_iter = 100000;
dgs_options.disp_flag = 2;

%% Run the algorithm
addpath('../..')

[~,~,x_cell] = eps_descent_method(problem_data,dgs_options); 

%% Process the results
f = problem_data.f;

xj_arr = zeros(n,j_max);
Nj_arr = zeros(1,j_max);
zl_arr = [];
ij_arr = zeros(1,j_max);
for j = 1:j_max
    xj_arr(:,j) = x_cell{j}(:,end);
    Nj_arr(j) = size(x_cell{j},2)-1;
    zl_arr = [zl_arr,x_cell{j}(:,1:end)];
    ij_arr(j) = size(zl_arr,2);
end
l_max = size(zl_arr,2);

fzl_arr = zeros(1,l_max);
for l = 1:l_max
    fzl_arr(l) = f(zl_arr(:,l));
end

N_bar = max(Nj_arr);

M = 10;
p = 2;

%% Plot (a)

lw = 1.5;

figure
h0 = plot(1:j_max,log10(vecnorm(xj_arr - x_min,2,1)),'k.-','MarkerSize',15,'LineWidth',lw);
hold on
h1 = plot(1:j_max,log10(M*dgs_options.eps_arr.^(1/p)),'r-','LineWidth',lw);
h2 = plot(1:j_max,log10(M*dgs_options.delta_arr.^(1/(p-1))),'b-','LineWidth',lw);
legend([h0,h1,h2],{'$\| x^j - x^* \|$','$M \varepsilon_j^{1/p}$','$M \delta_j^{1/(p-1)}$'},'Interpreter','latex','Location','ne','FontSize',20)

grid on
axis square
xlim([1,j_max])
ylim([-11,7])
yticks(-12:3:7)
xlabel('$j$','Interpreter','latex')
set(gca,'linewidth',1.1)
set(gca,'fontsize',15)

% Log tick labeling
old_ticks = yticks;
new_ticks_cell = cell(numel(old_ticks),1);
for i = 1:numel(old_ticks)
    new_ticks_cell{i} = ['10^{',num2str(old_ticks(i)),'}'];
end
yticklabels(new_ticks_cell)

%% Plot (b)

lw = 1.5;
ms = 15;

figure
plot(1:j_max,Nj_arr,'k.-','LineWidth',lw,'MarkerSize',ms);

grid on
xlim([1,j_max])
xlabel('$j$','Interpreter','latex')
ylabel('$N_j$','Interpreter','latex')
ylim([0,6.5])
set(gca,'linewidth',1.1)
set(gca,'fontsize',15)

%% Plot (c)

r = @(l) max([M*eps_fun(l).^(1/p);M*del_fun(l).^(1/(p-1))],[],1);
r_tilde = @(l) r(l/(N_bar+1) - 1);
r_tilde_arr = r_tilde(1:size(zl_arr,2));

lw = 1.75;
ms = 12;

figure
h1 = plot(1:l_max,log10(abs(fzl_arr - f(x_min))),'k.-','LineWidth',lw,'MarkerSize',ms);
hold on
h2 = plot(1:l_max,log10(r_tilde_arr),'r-','LineWidth',lw,'MarkerSize',ms);
plot(ij_arr(2:end),log10(abs(fzl_arr(ij_arr(2:end)) - f(x_min))),'ko','LineWidth',lw,'MarkerSize',8);

xlabel('$l$','Interpreter','latex');
legend([h1,h2],{'$f(z^l) - f(x^*)$','$\tilde{r}(l)$'},'Interpreter','latex','Location','sw','FontSize',20)
xlim([1,l_max])
ylim([-22,2.5])
grid off
grid on
set(gca,'linewidth',1.1)
set(gca,'fontsize',15)

% Log tick labeling
old_ticks = yticks;
new_ticks_cell = cell(numel(old_ticks),1);
for i = 1:numel(old_ticks)
    new_ticks_cell{i} = ['10^{',num2str(old_ticks(i)),'}'];
end
yticklabels(new_ticks_cell)
