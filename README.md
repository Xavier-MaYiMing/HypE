### HypE: Hypervolume estimation algorithm for multiobjective optimization

##### Reference: Bader J, Zitzler E. HypE: An algorithm for fast hypervolume-based many-objective optimization[J]. Evolutionary Computation, 2011, 19(1): 45-76.

##### HypE is an indicator-based multi-objective evolutionary algorithm (MOEA) using hypervolume in mating selection and environmental selection.

| Variables   | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| npop        | Population size                                      |
| iter        | Iteration number                                     |
| lb          | Lower bound                                          |
| ub          | Upper bound                                          |
| nobj        | The dimension of objective space (default = 3)       |
| eta_c       | Spread factor distribution index (default = 20)      |
| eta_m       | Perturbance factor distribution index (default = 20) |
| ns          | Sample number in HV calculation (default = 10000)    |
| nvar        | The dimension of decision space                      |
| pop         | Population                                           |
| objs        | Objectives                                           |
| ref         | Reference point                                      |
| HV          | Hypervolume                                          |
| mating_pool | Mating pool                                          |
| off         | Offsprings                                           |
| off_objs    | The objective of offsprings                          |
| pf          | Pareto front                                         |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(50, 200, np.array([0] * 100), np.array([1] * 100))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/HypE/blob/main/Pareto%20front.png)



