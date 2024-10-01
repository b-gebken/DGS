% This script applies the DGS method to the 20 test problems from
% [H2004,HMM2004].

% [H2004] Haarala (2004): Large-scale nonsmooth optimization: variable
% metric bundle method with limited memory
% [HMM2004] Haarala, Miettinen, Mäkelä (2004): New limited memory bundle
% method for large-scale nonsmooth optimization. doi:
% 10.1080/10556780410001689225

clear all
rng('default');

% Prepare test problems. (Derivatives for "brown_2" and "mifflin_2" are
% computed symbolically.)
disp('Preparing test problems...')
problem_arr = struct('n', repmat({[]}, 1, 20),'x0',[],'f',[],'subgrad_f',[]);

% Problems 1-10
addpath(genpath('test_funs/H2004lib/'));
func_name_cell = {'maxq','mxhilb','chained_LQ','chained_CB3_I','chained_CB3_II','active_faces','brown_2','mifflin_2','crescent_I','crescent_II'};
for prob_no = 1:10
    problem_arr(prob_no).n = 50;
    if(prob_no == 7 || prob_no == 8)
        n = problem_arr(prob_no).n;
        eval(['init_',func_name_cell{prob_no}])
        f = eval(func_name_cell{prob_no});
        subgrad_f = eval(['grad_',func_name_cell{prob_no}]);
        problem_arr(prob_no).f = f;
        problem_arr(prob_no).subgrad_f = subgrad_f;
        clear f subgrad_f
    else
        problem_arr(prob_no).f = @(in) feval(func_name_cell{prob_no},in);
        problem_arr(prob_no).subgrad_f = @(in) feval(['grad_',func_name_cell{prob_no}],in);
    end

    problem_arr(prob_no).x0 = feval([func_name_cell{prob_no},'_x0'],problem_arr(prob_no).n);
end

% Problems 11-20
addpath(genpath('test_funs/TEST29/'));
prob_no_shift = [2,5,6,11,13,17,19,20,22,24];
dim_arr = 50*ones(1,10);

for prob_no = 1:10
    problem_arr(prob_no+10).n = dim_arr(prob_no);

    problem_arr(prob_no+10).f = @(in) feval(['problem',num2str(prob_no_shift(prob_no))],in);
    problem_arr(prob_no+10).subgrad_f = @(in) feval(['grad_problem',num2str(prob_no_shift(prob_no))],in);

    problem_arr(prob_no+10).x0 = feval(['problem',num2str(prob_no_shift(prob_no)),'_x0'],problem_arr(prob_no+10).n);
end

disp('...done!')

% Set parameters for DGS 
j_max = 7;
kappa_eps = 0.1;
eps0 = 10;
eps_fun = @(j) eps0*kappa_eps.^(j);
del_fun = @(j) 10^-3 * ones(size(j));
dgs_options.eps_arr = eps_fun(0:j_max-1);
dgs_options.delta_arr = del_fun(0:j_max-1);

dgs_options.rand_sample_N = 0;
dgs_options.memory_size = Inf;
dgs_options.c = 0.5;
dgs_options.ls_flag = 'armijo_normal';
dgs_options.max_iter = 10000;
dgs_options.disp_flag = 1;

% Run the benchmark
addpath('..');
num_problems = numel(problem_arr);
dgs_output = struct('x_arr', repmat({[]}, 1, num_problems),'f_eval',[],'subgrad_eval',[],'runtime',[],'final_f',[]);

totalT = tic;
for i = 1:num_problems
    disp(['----- Problem ',num2str(i),' -----'])
    
    tic
    [x_opt,f_opt,x_cell,eval_counter] = eps_descent_method(problem_arr(i),dgs_options);
    dgs_output(i).runtime = toc;
    dgs_output(i).x_arr = [x_cell{:}];
    dgs_output(i).f_eval = eval_counter.f_eval;
    dgs_output(i).subgrad_eval = eval_counter.subgrad_eval;
    dgs_output(i).iter = size([x_cell{:}],2);
    dgs_output(i).final_f = problem_arr(i).f(x_cell{end}(:,end));
    
end
bench_time = toc(totalT);
disp(['Benchmark completed in ',num2str(bench_time/60),' minutes.'])

% Convert results to a table
name_cell = {'MAXQ';
    'MXHILB';
    'Chained LQ';
    'Chained CB3 I';
    'Chained CB3 II';
    'No. of Act. Faces';
    'Brown fun. 2';
    'Chained Mifflin 2';
    'Chained Crescent I';
    'Chained Crescent II';
    'P2, TEST29';
    'P5, TEST29';
    'P6, TEST29';
    'P11, TEST29';
    'P13, TEST29';
    'P17, TEST29';
    'P19, TEST29';
    'P20, TEST29';
    'P22, TEST29';
    'P24, TEST29'};

% (Approximated) minimal values for all test problems
opt_vals = [0;
    0;
    -(50-1)*2^(1/2);
    2*(50-1);
    2*(50-1);
    0;
    0;
    -34.795178390715485;
    0;
    0;
    0;
    1.811173540856273e-11;
    1.584669563348129e-06;
    5.839003843430868e+02;
    27.227867622575232;
    3.746943679772130e-07;
    1.703219237679747e-10;
    6.696865284538944e-12;
    2.683326094064450e-04;
    0];

T = table(name_cell,...
    [dgs_output(:).final_f]',...
    [dgs_output(:).final_f]' - opt_vals,...
    [dgs_output(:).f_eval]',...
    [dgs_output(:).subgrad_eval]',...
    [dgs_output(:).iter]',...
    [dgs_output(:).runtime]');
T.Properties.VariableNames = {'Problem','Obj. val.','Acc.','f eval.','subgrad eval.','Iter.','Runtime'};

disp(T)