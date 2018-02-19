"""
Module containing various selection functions previously used in the literature.

@author davidrpugh

"""
def kirkpatrick_selection(x_A, d=1):
    return d * x_A / (1 + (d - 1) * x_A)


def kirkpatrick_selection_derivative(x_A, d=1):
    return d / (1 + (d - 1) * x_A)**2


def perfect_selection(x_A, d):
    return 1


def perfect_selection_derivative(x_A, d):
    return 0


def random_selection(x_A, d):
    return x_A


def random_selection_derivative(x_A, d):
    return 1


def seger_selection(x_A, d=0):
    return x_A * (1 + d * (1 - x_A))


def seger_selection_derivative(x_A, d=0):
    return 1 + d * (1 - 2 * x_A)


def wright_selection(x_A, d=0):
    return (d * x_A + (1 - d) * x_A**2)**0.5


def wright_selection_derivative(x_A, d=0):
    return 0.5 * (d + 2 * (1 - d) * x_A) * (d * x_A + (1 - d) * x_A**2)**(-0.5)
