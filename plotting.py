"""
Module containing various plotting functions.

@author davidrpugh

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import models
import selection_functions


def plot_generalized_sexual_selection(x0, x1, x2, selection_function, d0, d2,
                                      T, R, P, S, mutation_rate, max_time):

    fig, ax = plt.subplots(1,1, figsize=(10,8))

    # prepare the axes
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"Time, $t$", fontsize=15)
    ax.set_ylabel(r"Female genotype shares, $x_i$", fontsize=15)

    # create the initial condition
    x3 = 1 - (x0 + x1 + x2)
    y0=np.array([x0,x1,x2,x3])
    assert np.allclose(y0.sum(), 1)

    # create the payoff kernel
    assert T > R and R > P and R > S, "Payoffs must satisfy either Prisoner's Dilemma or Stag Hunt constraints!"
    payoff_kernel = np.array([[R, S],
                              [T, P]])

    # create the selection functions
    if selection_function == "kirkpatrick":
        UGA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d0)
        UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d2)
    elif selection_function == "seger":
        UGA = lambda x_A: selection_function.seger_selection(x_A, d0)
        UgA = lambda x_A: selection_function.seger_selection(x_A, d2)
    else:
      raise ValueError("selection_function must be one of \"kirkpatrick\" or \"seger\".")

    # simulate the model starting from a random initial condition
    f = lambda t, y: models.generalized_sexual_selection(y, UGA, UgA, payoff_kernel, mutation_rate)
    result = integrate.solve_ivp(f, t_span=(0, max_time), y0=y0, rtol=1e-9, atol=1e-12,
                                 dense_output=True, vectorized=True)

    ax.plot(result.t, result.y[0], label="GA")
    ax.plot(result.t, result.y[1], label="Ga")
    ax.plot(result.t, result.y[2], label="gA")
    ax.plot(result.t, result.y[3], label="ga")
    ax.legend()
    plt.show()

    return result
