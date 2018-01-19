"""
Module containing various selection functions previously used in the literature.

@author davidrpugh

"""
def kirkpatrick_selection(x_A, d=1):
    return d * x_A / (1 + (d - 1) * x_A)


def kirkpatrick_selection_derivative(x_A, d=1):
    return (1 + (d - 1) * x_A)**-2


def seger_selection(x_A, d=0):
    return x_A * (1 + d * (1 - x_A))


def seger_selection_derivative(x_A, d=0):
    return 1 + d * (1 - 2 * x_A)


def wright_bergstrom_matching(x_A, d=0):
    return d + (1 - d) * xA
