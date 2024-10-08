% This script reproduces the numerical results shown in the first example
% in [G2024a], Section 6.

clear all
rng('default')

% Define the problem
addpath('../test_funs/p_order_growth_fun')
n = 2;
order = 3;
problem_data.n = n;
problem_data.f = @(x) p_order_growth_fun(x,order);
problem_data.subgrad_f = @(x) grad_p_order_growth_fun(x,order);
problem_data.x0 = [10.0;0.0];

% Set parameters for DGS 
kappa_eps = 0.85;
eps0 = 0.01;
kappa_del = 0.75;
del0 = 20;
j_max = 70;
eps_fun = @(j) eps0*kappa_eps.^(j);
del_fun = @(j) del0*kappa_del.^(j);
dgs_options.eps_arr = eps_fun(0:j_max-1);
dgs_options.delta_arr = del_fun(0:j_max-1);

dgs_options.rand_sample_N = 100;
dgs_options.memory_size = 0;
dgs_options.c = 0.9;
dgs_options.ls_flag = 'armijo';
dgs_options.max_iter = 100000;
dgs_options.disp_flag = 1;

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

p = order;
x_min = zeros(n,1);

%% Plot (a)

lw = 1.5;

figure
h0 = plot(1:j_max,log10(vecnorm(xj_arr - x_min,2,1)),'k.-','MarkerSize',15,'LineWidth',lw);
hold on
h1 = plot(1:j_max,log10(dgs_options.eps_arr.^(1/p)),'r-','LineWidth',lw);
h2 = plot(1:j_max,log10(dgs_options.delta_arr.^(1/(p-1))),'b-','LineWidth',lw);
legend([h0,h1,h2],{'$\| x^j - x^* \|$','$\varepsilon_j^{1/p}$','$\delta_j^{1/(p-1)}$'},'Interpreter','latex','Location','ne','FontSize',20)

grid on
axis square
xlim([1,70])
ylim([-4.25,1.75])
xlabel('$j$','Interpreter','latex')
xticks(0:10:70)
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

figure
plot(1:l_max,log10(vecnorm(zl_arr - x_min,2,1)),'k.-','MarkerSize',10,'LineWidth',lw);
hold on
plot(ij_arr,log10(vecnorm(zl_arr(:,ij_arr) - x_min,2,1)),'ko','MarkerSize',6)
xline(ij_arr(35),'k--','LineWidth',2)

h0 = plot(1,1,'k.-','MarkerSize',15,'LineWidth',lw);
legend(h0,{'$\| z^l - x^* \|$'},'Interpreter','latex','Location','ne','FontSize',20)

grid on
axis square
xlim([1,l_max])
ylim([-4.25,1.75])
xlabel('$l$','Interpreter','latex')
set(gca,'linewidth',1.1)
set(gca,'fontsize',15)

% Log tick labeling
old_ticks = yticks;
new_ticks_cell = cell(numel(old_ticks),1);
for i = 1:numel(old_ticks)
    new_ticks_cell{i} = ['10^{',num2str(old_ticks(i)),'}'];
end
yticklabels(new_ticks_cell)

