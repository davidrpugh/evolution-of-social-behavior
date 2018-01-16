import numpy as np


def prisoners_dilemma_payoffs(random_state, max_payoff=10):
    T = random_state.randint(3, max_payoff)
    R = random_state.randint(2, T)
    P = random_state.randint(1, R)
    S = random_state.randint(P)
    payoffs = np.array([[R, S], [T, P]])
    return payoffs


def stag_hunt_payoffs(random_state, max_payoff=10):
    T = random_state.randint(3, max_payoff)
    R = random_state.randint(2, T)
    S = random_state.randint(1, R)
    P = random_state.randint(S)
    payoffs = np.array([[R, S], [T, P]])
    return payoffs
