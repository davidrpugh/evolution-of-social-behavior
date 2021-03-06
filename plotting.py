"""
Module containing various plotting functions.

@author davidrpugh

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

import models
import selection_functions
import symbolics


def plot_generalized_sexual_selection(x1, x2, x3, selection_function, d1, d3,
                                      T, R, P, S, M, m, epsilon, max_time, method="RK45"):

    fig, axes = plt.subplots(1,2, figsize=(15,8), sharex=True)

    # prepare the axes
    #axes[0].set_ylim((0, 1.05))
    axes[0].set_xlabel(r"Time, $t$", fontsize=15)
    axes[0].set_ylabel(r"Offspring genotype shares, $x_i$", fontsize=15)

    axes[1].set_xlabel(r"Time, $t$", fontsize=15)
    axes[1].set_ylabel(r"Total Offspring (Fitness), $N(t)$", fontsize=15)

    # create the initial condition
    x4 = 1 - x1 - x2 - x3
    y0 = np.array([x1, x2, x3, x4])
    assert y0.sum() <= 1

    # create the payoff kernel
    assert (T > R) and (R > P) and (R > S), "Payoffs must satisfy either Prisoner's Dilemma or Stag Hunt constraints! T={}, R={}, P={}, S={}".format(T, R, P, S)
    payoff_kernel = np.array([[R, S], [T, P]])

    # create the selection functions
    if selection_function == "kirkpatrick":
        UGA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d3)
    elif selection_function == "seger":
        UGA = lambda x_A: selection_functions.seger_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.seger_selection(x_A, d3)
    elif selection_function == "wright":
        UGA = lambda x_A: selection_functions.wright_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.wright_selection(x_A, d3)
    else:
        valid_funcs = ("kirkpatrick", "seger", "wright")
        msg = "Selection_function must be one of {}, {}, or {}.".format(*valid_funcs)
        raise ValueError(msg)

    # simulate the model starting from a random initial condition
    def f(t, y):
        W = models.generalized_sexual_selection(y, UGA, UgA, payoff_kernel, M, m, epsilon)
        y_dot = models.offspring_genotypes_evolution(W, y)
        return y_dot

    def f_jac(t, y):
        return None

    solution = integrate.solve_ivp(f, t_span=(0, max_time), y0=y0, method=method,
                                   rtol=1e-9, atol=1e-12, dense_output=True, vectorized=True)

    axes[0].plot(solution.t, solution.y[0], label="GA")
    axes[0].plot(solution.t, solution.y[1], label="Ga")
    axes[0].plot(solution.t, solution.y[2], label="gA")
    axes[0].plot(solution.t, solution.y[3], label="ga")
    axes[0].legend()

    def total_offspring(yt):
        W = models.generalized_sexual_selection(yt, UGA, UgA, payoff_kernel, M, m, epsilon)
        N = models.total_offspring(W, yt)
        return N

    def fitness(y):
        _, T = y.shape
        Ns = []
        for t in range(T):
            yt = y[:,[t]]
            N = total_offspring(yt)
            Ns.append(N)
        return np.array(Ns)

    axes[1].plot(solution.t, fitness(solution.y))

    optimize_result = optimize.minimize(lambda y: -total_offspring(y.reshape(-1, 1)),
                                        x0=0.25 * np.ones(4),
                                        bounds=[(0,1), (0,1), (0,1), (0,1)],
                                        constraints={"type": "eq", "fun": lambda y: y.sum() - 1})
    axes[1].axhline(-optimize_result.fun, color='k', linestyle="--", label=r"$\bar{N}^*$")
    axes[1].legend()

    plt.show()

    return (solution, optimize_result)
