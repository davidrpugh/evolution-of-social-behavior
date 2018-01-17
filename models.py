"""
Code defining the models.

@author davidrpugh

"""
import numpy as np


def _haploid_inheritance_probabilities(mutation_rate):
    # axis 0 is indexed by male genotype; axis 1 is indexed by female genotype; axis 2 is indexed by offspring genotype
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
    assert np.allclose(R.sum(axis=0), np.ones((4,4))), "{}, {}".format(R.sum(axis=0), R.sum(axis=1))
    return R


def generalized_sexual_selection(x, UGA, UgA, payoff_kernel, mutation_rate=0.0,
                                 number_of_potential_mates=2):
    number_of_genotypes = 4
    assert x.shape == (number_of_genotypes, 1)

    # mating probabilities are frequency dependent
    x_A, x_a = np.sum(x[::2]), np.sum(x[1::2])
    assert np.allclose(x_a, 1 - x_A, atol=1e-5), "1-x_A should equal x_a; actual difference is {}".format(1 - x_A - x_a)

    # determine the payoffs
    Pi = np.tile(payoff_kernel, (2, 2))

    # determine the shares of a particular genotype in the alpha(k) phenotype
    S = np.tile(x, number_of_genotypes).T / np.array([x_A, x_a, x_A, x_a])
    assert np.allclose(np.ones(number_of_genotypes), S[:, ::2].sum(axis=1))
    assert np.allclose(np.ones(number_of_genotypes), S[:, 1::2].sum(axis=1))

    # determine mate selection probabilities
    phenotype_selection_probabilities = np.vstack((np.tile(np.array([UGA(x_A), 1 - UGA(x_A)]), (2,2)),
                                                   np.tile(np.array([UgA(x_A), 1 - UgA(x_A)]), (2,2))))
    genotype_selection_probabilities = S * phenotype_selection_probabilities
    U = np.tile(genotype_selection_probabilities , (number_of_potential_mates, 1, 1))
    U = U[:, :, :, np.newaxis] # insert additional axis for correct broadcasting...

    # actual mate selection: female selects one particular male to mate with!
    uniform_random_mate_selection = lambda mates: np.sum(mates / mates.size)
    V = np.apply_along_axis(uniform_random_mate_selection, axis=0, arr=U)

    # compute the haploid inheritance probabilities
    R = _haploid_inheritance_probabilities(mutation_rate)

    # define the replicator equation
    W = R * V * Pi
    offspring_by_genotype = W.sum(axis=0).dot(x)
    total_offspring = offspring_by_genotype.sum(axis=0)
    x_dot = (offspring_by_genotype / total_offspring) - x
    assert np.allclose(x_dot.sum(), 0.0, atol=1e-5), "Derivatives should sum to one; actual sum is {}".format(x_dot.sum())

    return x_dot
