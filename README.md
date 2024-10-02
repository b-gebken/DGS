<h1>Deterministic gradient sampling method</h1>

An implementation of the deterministic gradient sampling method for the solution of nonsmooth, nonconvex optimization problems that was used for the numerical experiments in [G2024a]. It combines the method proposed in [GP2021,G2022] (for single-objective problems) with the bisection method from [G2024b].

In addition to the initial approximation of the Goldstein eps-subdifferential with a single subgradient at the current point, one has the options to randomly sample a number of points from the eps-ball and to use subgradients that were evaluated in previous iterations that still lie in the current eps-ball. Independent of the chosen initialization, the algorithm then enriches the approximation until it is sufficient (cf. (8) in [G2024b]). See the comments in eps_descent_method.m for more details.

The folder Examples contains a script that performs a benchmark in which the method is applied to the 20 test problems of [H2004].

(If you are using this implementation in your research, please cite [GP2021] and [G2024b].)

[G2024a] Gebken (2024): Analyzing the speed of convergence in nonsmooth optimization via the Goldstein subdifferential with application to descent methods (to be submitted)<br/>
[GP2021] Gebken, Peitz (2021): An Efficient Descent Method for Locally Lipschitz Multiobjective Optimization Problems. doi:0.1007/s10957-020-01803-w<br/>
[G2022] Gebken (2022): Computation and analysis of pareto critical sets in smooth and nonsmooth multiobjective optimization. doi:10.17619/UNIPB/1-1327<br/>
[G2024b] Gebken (2024): A note on the convergence of deterministic gradient sampling in nonsmooth optimization. doi:10.1007/s10589-024-00552-0<br/>
[H2004] Haarala (2004): Large-scale nonsmooth optimization: variable metric bundle method with limited memory<br/>
