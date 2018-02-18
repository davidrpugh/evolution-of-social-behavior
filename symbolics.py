import sympy as sym


class U(sym.Function):
    """Generic matching function"""

    is_real = True

    is_positive = True

    @classmethod
    def eval(cls, x):
        """We require the U(0)=0 and U(1)=1"""
        if x.is_Number and x is sym.S.Zero:
            return sym.S.Zero
        elif x.is_Number and x is sym.S.One:
            return sym.S.One


class UGA(U):
    """Matching function for G females."""


class UgA(U):
    """Matching function for g females."""


def total_offspring(x1, x2, x3, UGA, UgA, T, R, P, S):
    out = (
           2 * R *((x1 + x2) * UGA(x1 + x3)**2 + (1 - (x1 + x2)) * UgA(x1 + x3)**2) +
           2 * P * ((x1 + x2) * (1 - UGA(x1 + x3))**2 + (1 - (x1 + x2)) * (1 - UgA(x1 + x3))**2) +
           2 * (S + T) * ((x1 + x2) * UGA(x1 + x3) * (1 - UGA(x1 + x3)) + (1 - (x1 + x2)) * UgA(x1 + x3) * (1 - UgA(x1 + x3)))
          )
    return out


def equation_motion_GA_share(x1, x2, x3, UGA, UgA, T, R, P, S):
    numerator = (
                 x1 * UGA(x1 + x3)**2 * (1) * x1 / (x1 + x3) * 2*R +  # 2*R
                 x1 * UGA(x1 + x3)**2 * (1/2) * x3 / (x1 + x3) * 2*R + # 0

                 x1 * (1 - UGA(x1 + x3))**2 * (1/2) * x2 / (1 - x1 - x3) * 2*P + # 0
                 x1 * (1 - UGA(x1 + x3))**2 * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P + # 0

                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1) * x1 / (x1 + x3) * S + # 0
                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x3 / (x1 + x3) * S + # 0
                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x2 / (1 - x1 - x3) * T + # 0
                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T + # 0

                 x2 * UGA(x1 + x3)**2 * (1/2) * x1 / (x1 + x3) * 2*R + # 0
                 x2 * UGA(x1 + x3)**2 * (1/4) * x3 / (x1 + x3) * 2*R + # 0

                 x2 * (1 - UGA(x1 + x3))**2 * (0) + # 0

                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x1 / (x1 + x3) * S + # 0
                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * x3 / (x1 + x3) * S + # 0

                 x3 * UgA(x1 + x3)**2 * (1/2) * x1 / (x1 + x3) * 2*R + # 0

                 x3 * (1 - UgA(x1 + x3))**2 * (1/4) * x2 / (1 - x1 - x3) * 2* P + # 0

                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x1 / (x1 + x3) * S + # 0
                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x2 / (1 - x1 - x3) * T + # 0

                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/4) * x1 / (x1 + x3) * 2*R + # 0

                 (1 - x1 - x2 - x3) * (1 - UgA(x1 + x3))**2 * (0) + # 0

                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x1 / (x1 + x3) * S # 0
                 )

    x1_dot = (numerator / total_offspring(x1, x2, x3, UGA, UgA, T, R, P, S)) - x1
    return x1_dot


def equation_motion_Ga_share(x1, x2, x3, UGA, UgA, T, R, P, S):
    numerator = (
                 x1 * UGA(x1 + x3)**2 * (0) +

                 x1 * (1 - UGA(x1 + x3))**2 * (1/2) * x2 / (1 - x1 - x3) * 2*P +
                 x1 * (1 - UGA(x1 + x3))**2 * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x2 / (1 - x1 - x3) * T +
                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 x2 * UGA(x1 + x3)**2 * (1/2) * x1 / (x1 + x3) * 2*R +
                 x2 * UGA(x1 + x3)**2 * (1/4) * x3 / (x1 + x3) * 2*R +

                 x2 * (1 - UGA(x1 + x3))**2 * (1) * x2 / (1 - x1 - x3) * 2*P +
                 x2 * (1 - UGA(x1 + x3))**2 * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x1 / (x1 + x3) * S +
                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * x3 / (x1 + x3) * S +
                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1) * x2 / (1 - x1 - x3) * T +
                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 x3 * UgA(x1 + x3)**2 * (0) +

                 x3 * (1 - UgA(x1 + x3))**2 * (1/4) * x2 / (1 - x1 - x3) * 2*P +

                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x2 / (1 - x1 - x3) * T +

                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/4) * x1 / (x1 + x3) * 2*R +

                 (1 - x1 - x2 - x3) * (1 - UgA(x1 + x3))**2 * (1/2) * x2 / (1 - x1 - x3) * 2*P +

                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x1 / (x1 + x3) * S +
                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x2 / (1 - x1 - x3) * T

                 )

    x2_dot = (numerator / total_offspring(x1, x2, x3, UGA, UgA, T, R, P, S)) - x2
    return x2_dot


def equation_motion_gA_share(x1, x2, x3, UGA, UgA, T, R, P, S):
    numerator = (
                 x1 * UGA(x1 + x3)**2 * (1/2) * x3 / (x1 + x3) * 2*R +

                 x1 * (1 - UGA(x1 + x3))**2 * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * x3 / (x1 + x3) * S +
                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 x2 * UGA(x1 + x3)**2 * (1/4) * x3 / (x1 + x3) * 2*R +

                 x2 * (1 - UGA(x1 + x3))**2 * (0) +

                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * x3 / (x1 + x3) * S +

                 x3 * UgA(x1 + x3)**2 * (1/2) * x1 / (x1 + x3) * 2*R +
                 x3 * UgA(x1 + x3)**2 * (1) * x3 / (x1 + x3) * 2*R +

                 x3 * (1 - UgA(x1 + x3))**2 * (1/4) * x2 / (1 - x1 - x3) * 2*P +
                 x3 * (1 - UgA(x1 + x3))**2 * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x1 / (x1 + x3) * S +
                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1) * x3 / (x1 + x3) * S +
                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x2 / (1 - x1 - x3) * T +
                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/4) * x1 / (x1 + x3) * 2*R +
                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/2) * x3 / (x1 + x3) * 2*R +

                 (1 - x1 - x2 - x3) * (1 - UgA(x1 + x3))**2 * (0) +

                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x1 / (x1 + x3) * S +
                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x3 / (x1 + x3) * S

                 )

    x3_dot = (numerator / total_offspring(x1, x2, x3, UGA, UgA, T, R, P, S)) - x3
    return x3_dot


def equation_motion_ga_share(x1, x2, x3, UGA, UgA, T, R, P, S):
    numerator = (
                 x1 * UGA(x1 + x3)**2 * (0) +

                 x1 * (1 - UGA(x1 + x3))**2 * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x1 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 x2 * UGA(x1 + x3)**2 * (1/4) * x3 / (x1 + x3) * 2*R +

                 x2 * (1 - UGA(x1 + x3))**2 * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/4) * x3 / (x1 + x3) * S +
                 x2 * 2 * UGA(x1 + x3) * (1 - UGA(x1 + x3)) * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 x3 * UgA(x1 + x3)**2 * (0) +

                 x3 * (1 - UgA(x1 + x3))**2 * (1/4) * x2 / (1 - x1 - x3) * 2*P +
                 x3 * (1 - UgA(x1 + x3))**2 * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x2 / (1 - x1 - x3) * T +
                 x3 * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T +

                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/4) * x1 / (x1 + x3) * 2*R +
                 (1 - x1 - x2 - x3) * UgA(x1 + x3)**2 * (1/2) * x3 / (x1 + x3) * 2*R +

                 (1 - x1 - x2 - x3) * (1 - UgA(x1 + x3))**2 * (1/2) * x2 / (1 - x1 - x3) * 2*P +
                 (1 - x1 - x2 - x3) * (1 - UgA(x1 + x3))**2 * (1) *(1 - x1 - x2 - x3) / (1 - x1 - x3) * 2*P +

                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/4) * x1 / (x1 + x3) * S +
                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x3 / (x1 + x3) * S +
                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1/2) * x2 / (1 - x1 - x3) * T +
                 (1 - x1 - x2 - x3) * 2 * UgA(x1 + x3) * (1 - UgA(x1 + x3)) * (1) * (1 - x1 - x2 - x3) / (1 - x1 - x3) * T
                 )

    x4_dot = (numerator / total_offspring(x1, x2, x3, UGA, UgA, T, R, P, S)) - (1 - x1 - x2 - x3)
    return x4_dot

# define symbolic variables
x1, x2, x3, xA = sym.symbols('x1, x2, x3, xA')
T, R, P, S = sym.symbols('T, R, P, S')

# equations for generalized sexual selection model
F = sym.Matrix([equation_motion_GA_share(x1, x2, x3, UGA, UgA, T, R, P, S),
                equation_motion_Ga_share(x1, x2, x3, UGA, UgA, T, R, P, S),
                equation_motion_gA_share(x1, x2, x3, UGA, UgA, T, R, P, S)])
F_jac = (F.jacobian((x1, x2, x3))
          .subs({x1+x3: xA})
          .doit())

_F = sym.lambdify((x1, x2, x3, UGA, UgA, T, R, P, S), F, modules="numpy")
_F_jac = sym.lambdify((x1, x2, x3, xA, UGA, sym.Derivative(UGA(xA), xA), UgA, sym.Derivative(UgA(xA), xA), T, R, P, S), F_jac, modules="numpy")

# equations for the monomorphic_gamma model
f = sym.Matrix([equation_motion_GA_share(x1, 1-x1, 0, UGA, UgA, T, R, P, S)])
f_jac = sym.diff(f, x1, 1)

_f = sym.lambdify((x1, UGA, UgA, T, R, P, S), f, modules="numpy")
_f_jac = sym.lambdify((x1, UGA, UgA, T, R, P, S), f_jac, modules="numpy")

# equations for computing optimal fitness
UGA_star, UgA_star, xG_star = sym.symbols("UGA_star, UgA_star, xG_star")
equilibrium_values = {UGA(x1+x3): UGA_star, UgA(x1+x3): UgA_star, x1+x2: xG_star}
N = total_offspring(x1, x2 , x3, UGA, UgA, T, R, P, S)
equilibrium_total_offspring = N.subs(equilibrium_values)

_equilibrium_total_offspring = sym.lambdify((xG_star, UGA_star, UgA_star, T, R, P, S),
                                            equilibrium_total_offspring,
                                            modules="numpy")

def generalized_sexual_selection(x, UGA, UgA, payoff_kernel):
    ((R, S), (T, P)) = payoff_kernel
    return _F(x[0], x[1], x[2], UGA, UgA, T, R, P, S)


def generalized_sexual_selection_jac(x, UGA, UgA, payoff_kernel):
    ((R, S), (T, P)) = payoff_kernel
    return _F_jac(x[0], x[1], x[2], UGA, UgA, T, R, P, S)


def monomorphic_gamma_sexual_selection(x, UGA, UgA, payoff_kernel):
    ((R, S), (T, P)) = payoff_kernel
    return _f(x[0], UGA, UgA, T, R, P, S)


def monomorphic_gamma_sexual_selection_jac(x, UGA, UgA, payoff_kernel):
    ((R, S), (T, P)) = payoff_kernel
    return _f_jac(x[0], UGA, UgA, T, R, P, S)
