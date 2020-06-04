import numpy as np

import models
import payoffs
import selection_functions


prng = np.random.RandomState(42)
payoff_kernel = payoffs.prisoners_dilemma_payoffs(prng)

d0, d2 = 1.5, 1.0
UGA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d0)
UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d2)

number_of_genotypes = 4
y0, = prng.dirichlet(np.ones(number_of_genotypes), 1)
f = lambda t, y: models.generalized_sexual_selection(y, UGA, UgA, payoff_kernel)

# confirm equilibrium with all GA
eps = 1e-12
equilibrium = np.array([1-eps, 0, 0, eps])
assert np.allclose(f(0, equilibrium.reshape(-1, 1)), np.zeros(number_of_genotypes).reshape(-1,1))

# confirm equilibrium with all ga
eps = 1e-12
equilibrium = np.array([eps, 0, 0, 1-eps])
assert np.allclose(f(0, equilibrium.reshape(-1, 1)), np.zeros(number_of_genotypes).reshape(-1,1))
