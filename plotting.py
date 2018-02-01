"""
Module containing various plotting functions.

@author davidrpugh

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import models
import selection_functions
import symbolics


def plot_generalized_sexual_selection(x1, x2, x3, selection_function, d1, d3,
                                      T, R, P, S, max_time):

    fig, ax = plt.subplots(1,1, figsize=(10,8))

    # prepare the axes
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"Time, $t$", fontsize=15)
    ax.set_ylabel(r"Offspring genotype shares, $x_i$", fontsize=15)

    # create the initial condition
    x4 = 1 - (x1 + x2 + x3)
    y0=np.array([x1,x2,x3,x4])
    assert np.allclose(y0.sum(), 1)

    # create the payoff kernel
    assert (T > R) and (R > P) and (R > S), "Payoffs must satisfy either Prisoner's Dilemma or Stag Hunt constraints! T={}, R={}, P={}, S={}".format(T, R, P, S)
    payoff_kernel = np.array([[R, S],
                              [T, P]])

    # create the selection functions
    if selection_function == "kirkpatrick":
        UGA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d3)
    elif selection_function == "seger":
        UGA = lambda x_A: selection_functions.seger_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.seger_selection(x_A, d3)
    else:
      raise ValueError("selection_function must be one of \"kirkpatrick\" or \"seger\".")

    # simulate the model starting from a random initial condition
    f = lambda t, y: symbolics.generalized_sexual_selection(y, UGA, UgA, payoff_kernel)
    result = integrate.solve_ivp(f, t_span=(0, max_time), y0=y0, rtol=1e-9, atol=1e-12,
                                 dense_output=True, vectorized=True)

    ax.plot(result.t, result.y[0], label="GA")
    ax.plot(result.t, result.y[1], label="Ga")
    ax.plot(result.t, result.y[2], label="gA")
    ax.plot(result.t, result.y[3], label="ga")
    ax.legend()
    plt.show()

    return result


def plot_monomorphic_gamma_sexual_selection(x1, selection_function, d1, d3,
                                            T, R, P, S, max_time):

    fig, ax = plt.subplots(1,1, figsize=(10,8))

    # prepare the axes
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"Time, $t$", fontsize=15)
    ax.set_ylabel(r"Offspring genotype shares, $x_i$", fontsize=15)

    # create the initial condition
    x2 = 1 - x1
    y0=np.array([[x1], [x2]])
    assert np.allclose(y0.sum(), 1)

    # create the payoff kernel
    assert (T > R) and (R > P) and (R > S), "Payoffs must satisfy either Prisoner's Dilemma or Stag Hunt constraints! T={}, R={}, P={}, S={}".format(T, R, P, S)
    payoff_kernel = np.array([[R, S],
                              [T, P]])

    # create the selection functions
    if selection_function == "kirkpatrick":
        UGA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d3)
    elif selection_function == "seger":
        UGA = lambda x_A: selection_functions.seger_selection(x_A, d1)
        UgA = lambda x_A: selection_functions.seger_selection(x_A, d3)
    else:
      raise ValueError("selection_function must be one of \"kirkpatrick\" or \"seger\".")

    # simulate the model starting from a random initial condition
    f = lambda t, y: symbolics.monomorphic_gamma_sexual_selection(y, UGA, UgA, payoff_kernel)
    result = integrate.solve_ivp(f, t_span=(0, max_time), y0=y0, rtol=1e-9, atol=1e-12,
                                 dense_output=True, vectorized=True)

    ax.plot(result.t, result.y[0], label="GA")
    ax.plot(result.t, result.y[1], label="Ga")
    ax.plot(result.t, result.y[2], label="gA")
    ax.plot(result.t, result.y[3], label="ga")
    ax.legend()
    plt.show()

    return result
