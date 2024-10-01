% Deterministic gradient sampling method
%   The implementation of the deterministic gradient sampling method for
%   the solution of nonsmooth, nonconvex optimization problems that was
%   used for the numerical experiments in [G2024a]. It combines the method
%   proposed in [GP2021,G2022] (for single-objective problems) with the
%   bisection method from [G2024b]. Consists of an outer loop ("for j =
%   1:j_max"), which loops over all given pairs (eps,delta), and an inner
%   loop ("for i = 1:max_iter"), which performs descent steps with fixed
%   eps and delta until an (eps,delta)-critical point (cf. Def. 3.2 in
%   [GP2021]) is found (or the maximum number of iterations is reached). In
%   addition to the initial approximation of the Goldstein
%   eps-subdifferential with a single subgradient at the current point, one
%   has the options to randomly sample a number of points from the eps-ball
%   and to use subgradients that were evaluated in previous iterations that
%   still lie in the current eps-ball. Independent of the chosen
%   initialization, the algorithm then enriches the approximation until it
%   is sufficient (cf. (8) in [G2024b]). For the step length, one has the
%   choice of either using a normalized or non-normalized Armijo
%   backtracking line search, or using the explicit step length eps/norm(v)
%   that yields guaranteed descent due to inequality (8) in [G2024b]. (The
%   latter option means that the method always performs steps of length
%   eps. If backtracking would result in a step short than eps/norm(v),
%   then eps/norm(v) is used instead.)
%
% Input:
%   problem_data: A struct with the following fields:
%       n: Number of variables. 
%       x0: Initial point (as a column vector).
%       f: Function handle that returns the objective function at a
%           given point.
%       subgrad_f: Function handle that returns an arbitrary element
%           of the subdifferential at a given point.
%   dgs_options: A struct with the following fields:
%       eps_arr: Vector of eps values for the Goldstein
%       eps-subdifferential. (Must be the same length as delta_arr.)
%       delta_arr: Vector of delta values for the stopping condition.
%           (Must be the same length as eps_arr.) 
%       rand_sample_N: Number of randomly sampled points for the initial
%           approximation of the eps-subdifferential. ("0" means fully
%           deterministic approximation as in [GP2021,G2022].)
%       memory_max_size: The maximum number of subgradients that are stored
%           for the initial approximation of the eps-subdifferential. If
%           the maximum size is reached and a new subgradient is to be
%           stored, then the oldest subgradient is removed. "0" means no
%           memory is used (as in [GP2021,G2022]), "Inf" means that there
%           is no limit on the memory size. (Note that this does not limit
%           the size of the individual approximations of the
%           eps-subdifferential.)
% 		ls_flag: Can be "armijo", "armijo_normal" or "eps".
% 			armijo: Armijo backtracking line search with initial
% 			    step length 1.
%           armijo_normal: Normalized Armijo backtracking line search with
%               initial step length 1/norm(v).
%           eps: Normalized step length eps/norm(v).
% 		c: A number from (0,1) that controls the quality of the
% 		    eps-subdifferential approximation (cf. (8) in [G2024b]).
% 		    Choosing c close to 1 means (potentially) steeper descent in
% 		    each iteration, at the cost of a more expensive approximation
% 		    of the eps-subdifferential. Additionally, c is used as an
% 		    Armijo-type parameter in the line search (if the corresponding
% 		    options is chosen).
%       max_iter: The maximum number of iterations for each pair
% 		    (eps,delta). (I.e., the total number of iterations of the
% 		    method is at most numel(eps_arr)*max_iter.)
%       disp_flag: Controls the amount of information that is displayed
%           during a run.
% 			0 - No output.
%           1 - Only the final results are displayed.
%           2 - Output after every iteration of the outer loop.
%           3 - Output after every iteration of the inner and outer loop.
% 
% Output:
%   x_opt: Final iterate. Let eps* and delta* be the final elements of
%       eps_arr and delta_arr. Then x_opt is (eps*,delta*)-critical (unless
%       the maximum number of iterations was reached for the final outer
%       loop).
% 	f_opt: Objective value in the final iterate x_opt. 
%   x_cell: A cell array of length numel(eps_arr), where x_cell{j} contains
%       the descent sequence corresponding to the pair
%       (eps_arr(j),delta_arr(j)).
% 
% 
% [G2024a] Gebken (2024): Analyzing the speed of convergence in nonsmooth
% optimization via the Goldstein subdifferential with application to
% descent methods (to be submitted)
% [GP2021] Gebken, Peitz (2021): An Efficient Descent Method for Locally
% Lipschitz Multiobjective Optimization Problems. doi:
% 0.1007/s10957-020-01803-w
% [G2022] Gebken (2022): Computation and analysis of pareto critical sets
% in smooth and nonsmooth multiobjective optimization. doi:
% 10.17619/UNIPB/1-1327
% [G2024b] Gebken (2024): A note on the convergence of deterministic
% gradient sampling in nonsmooth optimization. doi:
% 10.1007/s10589-024-00552-0
		
function [x_opt,f_opt,x_cell,eval_counter] = eps_descent_method(problem_data,dgs_options)

% Read input
n = problem_data.n;
f = problem_data.f;
subgrad_f = problem_data.subgrad_f;
x0 = problem_data.x0;

eps_arr = dgs_options.eps_arr;
delta_arr = dgs_options.delta_arr;
rand_sample_N = dgs_options.rand_sample_N;
memory_max_size = dgs_options.memory_size;
c = dgs_options.c;
ls_flag = dgs_options.ls_flag;
max_iter = dgs_options.max_iter;
disp_flag = dgs_options.disp_flag;

% Initialization
j_max = numel(eps_arr);
x_cell = cell(j_max,1);
qp_optns = optimoptions('quadprog','Display','off','OptimalityTolerance',10^-16);

memory.sample_pts = [];
memory.subgrads = [];
memory.max_size = memory_max_size;

eval_counter.f_eval = 0;
eval_counter.subgrad_eval = 0;

if(disp_flag >= 1)
    start_tic = tic;
end

if(disp_flag >= 1)
    disp('Deterministic gradient sampling...')
end

% Loop over all (eps,delta) pairs
for j = 1:j_max
    
    if(disp_flag >= 2)
        disp(['    Iteration j = ',num2str(j),'/',num2str(j_max),'...']);
        disp(['        eps_j   = ',num2str(eps_arr(j))]);
        disp(['        delta_j = ',num2str(delta_arr(j))]);
        disp('        Running descent iterations...');
    end
    
    % Starting point is either x0 (if j = 1) or the final iterate of the
    % previous descent sequence.
    if(j == 1)
        x_arr = [x0,zeros(n,max_iter)];
        f_xi = f(x0); eval_counter.f_eval = eval_counter.f_eval + 1;
    else
        x_arr = [x_cell{j-1}(:,end),zeros(n,max_iter)];
        f_xi = f_x_new;
    end
    
    if(disp_flag >= 2)
        desc_tic = tic;
    end

    % Descent loop for fixed eps and delta
    for i = 1:max_iter
        
        if(disp_flag >= 3)
            tmp_counter = eval_counter.subgrad_eval;
            disp(['            Iteration i = ',num2str(i),' (j = ',num2str(j),')...']);
            disp(['                Computing descent direction...']);
        end
        
        % Step 2 in [G2024a]
        [v,f_eps_v,memory,eval_counter] = descent_direction(x_arr(:,i),f_xi,f,subgrad_f,eps_arr(j),delta_arr(j),c,rand_sample_N,memory,qp_optns,eval_counter);

        if(disp_flag >= 3)
            disp(['                    ...done!']);
            disp(['                Req. subgrad. eval.: ',num2str(eval_counter.subgrad_eval - tmp_counter)]);
        end
        
        % Step 3 in [G2024a]
        if(norm(v,2) <= delta_arr(j))
            % Step 4 in [G2024a]
            x_arr = x_arr(:,1:i);
            f_x_new = f_xi;
            break
        % Step 5 in [G2024a]
        else
            % Step 6 in [G2024a]
            if(strcmp(ls_flag,'eps'))
                t = eps_arr(j)/norm(v,2);
                f_x_new = f_eps_v;
            elseif(strcmp(ls_flag,'armijo') || strcmp(ls_flag,'armijo_normal'))
                if(strcmp(ls_flag,'armijo'))
                    t = 1;
                elseif(strcmp(ls_flag,'armijo_normal'))
                    t = 1/norm(v,2);
                end

                if(t <= eps_arr(j)/norm(v,2))
                    t = eps_arr(j)/norm(v,2);
                    f_x_new = f_eps_v;
                else
                    f_x_new = f(x_arr(:,i) + t*v); eval_counter.f_eval = eval_counter.f_eval + 1;
                    while(f_x_new - f_xi > -c*t*norm(v,2)^2)
                        t = t/2;
                        
                        if(t <= eps_arr(j)/norm(v,2))
                            t = eps_arr(j)/norm(v,2);
                            f_x_new = f_eps_v;
                            break;
                        end

                        f_x_new = f(x_arr(:,i) + t*v); eval_counter.f_eval = eval_counter.f_eval + 1;
                    end
                end
            end
 
            % Step 7 in [G2024a]
            x_arr(:,i+1) = x_arr(:,i) + t*v;
            
            if(disp_flag >= 3)
                disp(['                norm(v,2)  = ',num2str(norm(v,2)),' (delta_j = ',num2str(delta_arr(j)),', eps_j = ',num2str(eps_arr(j)),')']);
                disp(['                New f val. = ',num2str(f_x_new)]);
                disp(['                f decr.    = ',num2str(f_xi - f_x_new)]);
            end
            
            f_xi = f_x_new;
        end
    end
    
    if(disp_flag >= 2)
        desc_time = toc(desc_tic);
        disp(['        ...done in N_j = ',num2str(size(x_arr,2)-1),' iterations (in ',num2str(desc_time),'s).']);
    end
    
    if(i == max_iter)
        disp(['Warning: Maximum number of iterations reached for j = ',num2str(j),' before eps-delta-critical point was found.'])
    end

    x_cell{j} = x_arr;

end

x_opt = x_cell{end}(:,end);
f_opt = f_x_new;

if(disp_flag >= 1)
    total_time = toc(start_tic);
    disp('    ...done!')
    disp(['    Final obj. value:    ',num2str(f_x_new)])
    disp(['    Total iterations:    ',num2str(sum(cellfun(@(in) size(in,2),x_cell))),' = ',num2str(sum(cellfun(@(in) size(in,2)-1,x_cell))),' + ',num2str(j_max)])
    disp(['    Total f eval.:       ',num2str(eval_counter.f_eval)])
    disp(['    Total subgrad eval.: ',num2str(eval_counter.subgrad_eval)])
    disp(['    Total time:          ',num2str(total_time)])
end

end

