"""
Code defining the models.

@author davidrpugh

"""
import numpy as np
import sympy as sym


def _haploid_inheritance_probabilities(mutation_rate, incumbent_only):
    """
    R[i,j,k] is offspring genotype i from mother with genotype j and father with genotype k.

    """
    if incumbent_only:
        R = np.array([[[1 - mutation_rate, 1/2, (1/2) * (1 - mutation_rate), 1/4],
                       [1/2, 0, 1/4, 1/4 * mutation_rate],
                       [(1/2) * (1 - mutation_rate), 1/4, 0, 0],
                       [1/4, (1/4) * mutation_rate, 0, 0]],
                      [[mutation_rate, 1/2, 1/2 * mutation_rate, 1/4],
                       [1/2, 1, 1/4, (1/2) * (1 - (1/2) * mutation_rate)],
                       [1/2 * mutation_rate, 1/4, 0, 0],
                       [1/4, (1/2) * (1 - (1/2) * mutation_rate), 0, 0]],
                      [[0, 0, (1/2) * (1 - mutation_rate), 1/4],
                       [0, 0, 1/4, (1/4) * mutation_rate],
                       [(1/2) * (1 - mutation_rate), 1/4, 1-mutation_rate, 1/2],
                       [1/4, (1/4) * mutation_rate, 1/2, 0]],
                      [[0, 0, 1/2 * mutation_rate, 1/4],
                       [0, 0, 1/4, (1/2) * (1 - (1/2) * mutation_rate)],
                       [(1/2) * mutation_rate, 1/4, mutation_rate, 1/2],
                       [1/4, (1/2) * (1 - (1/2) * mutation_rate), 1/2, 1]]])
    else:
        R = np.array([[[1 - mutation_rate, 1/2, (1/2) * (1 - mutation_rate), 1/4],
                       [1/2, mutation_rate, 1/4, (1/2) * mutation_rate],
                       [(1/2) * (1 - mutation_rate), 1/4, 0, 0],
                       [1/4, (1/2) * mutation_rate, 0, 0]],
                      [[mutation_rate, 1/2, (1/2) * mutation_rate, 1/4],
                       [1/2, 1 - mutation_rate, 1/4, (1/2) * (1 - mutation_rate)],
                       [(1/2) * mutation_rate, 1/4, 0, 0],  # discrepancy here!
                       [1/4, (1/2) * (1 - mutation_rate), 0, 0]],
                      [[0, 0, (1/2) * (1 - mutation_rate), 1/4],
                       [0, 0, 1/4, (1/2) * mutation_rate],
                       [(1/2) * (1 - mutation_rate), 1/4, 1 - mutation_rate, 1/2],
                       [1/4, (1/2) * mutation_rate, 1/2, mutation_rate]],
                      [[0, 0, (1/2) * mutation_rate, 1/4],
                       [0, 0, 1/4, (1/2) * (1 - mutation_rate)],
                       [(1/2) * mutation_rate, 1/4, mutation_rate, 1/2],
                       [1/4, (1/2) * (1 - mutation_rate), 1/2, 1 - mutation_rate]]])
        
    return R


def _net_payoffs(payoff_kernel, M, m):
    # indexed female genotype j, male genotype k, male genotype l
    gross_payoffs = np.tile(payoff_kernel, (4, 2, 2))
    metabolic_costs = np.array([M, M, m, m]).reshape(4, 1, 1)
    net_payoffs = gross_payoffs - metabolic_costs

    return net_payoffs


def _offspring_by_genotype(W, x):
    """Number of offspring by genotype."""
    assert W.shape == (4, 4, 4)
    offspring_by_mother = W.sum(axis=2)
    offspring_by_genotype = offspring_by_mother.dot(x)
    return offspring_by_genotype


def total_offspring(W, x):
    """Total offspring across all genotypes."""
    assert W.shape == (4, 4, 4)
    offspring_by_genotype = _offspring_by_genotype(W, x)
    total_offspring = offspring_by_genotype.sum(axis=0)
    return total_offspring


def offspring_genotypes_evolution(W, x):
    """Equation of motion for offspring genotypes."""
    assert W.shape == (4, 4, 4)
    x_dot = (_offspring_by_genotype(W, x) / total_offspring(W, x)) - x
    return x_dot


def generalized_sexual_selection(x, UGA, UgA, payoff_kernel, M=0, m=0, mutation_rate=0.0, incumbent_only=False):
    number_of_genotypes, _ = x.shape
    x_A, x_a = np.sum(x[::2]), np.sum(x[1::2])
    phenotype_selection_kernel = np.vstack((np.tile(np.array([UGA(x_A), 1 - UGA(x_A)]), (2,2)),
                                            np.tile(np.array([UgA(x_A), 1 - UgA(x_A)]), (2,2))))
    S = np.tile(x, number_of_genotypes).T / np.array([x_A, x_a, x_A, x_a])
    R = _haploid_inheritance_probabilities(mutation_rate, incumbent_only)
    net_payoffs = _net_payoffs(payoff_kernel, M, m)
    W = R * ((phenotype_selection_kernel * S) * (phenotype_selection_kernel[:, np.newaxis, :] * net_payoffs).sum(axis=2))[np.newaxis, :, :]
    return W


def make_F_jac(UGA, UgA, payoff_kernel, metabolic_costs, mutation_rate):
    x1, x2, x3, x4 = sym.symbols("x1, x2, x3, x4", real=True, nonnegative=True)
    T, R, P, S = sym.symbols("T, R, P, S", real=True, positive=True)
    M, m = sym.symbols("M, m", real=True, nonnegative=True)
    epsilon = sym.symbols("epsilon", real=True, nonnegative=True)

    x = np.array([[x1], [x2], [x3], [x4]])
    W = generalized_sexual_selection(x, UGA, UgA, payoff_kernel, M, m, mutation_rate)
    x_dot = offspring_genotypes_evolution(W, x)
    F = sym.Matrix(x_dot)
    F_jac = F.jacobian((x1, x2, x3, x4))

    _F_jac = sym.lambdify((x1, x2, x3, x4, T, R, P, S, M, m, epsilon), F_jac, modules="numpy")

    def F_jac(t, y):
        ((R, S), (T, P)) = payoff_kernel
        M, m = metabolic_costs
        return _F_jac(y[0], y[1], y[2], y[3], T, R, P, S, M, m, mutation_rate)

    return F_jac
