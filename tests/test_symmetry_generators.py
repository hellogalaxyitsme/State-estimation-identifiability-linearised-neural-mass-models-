"""Every continuous symmetry generator annihilates the coefficient Jacobian identically."""

import sympy as sp

from neural_observability.theory.atlas import wendling_2002
from neural_observability.theory.families import JansenRit, NextGenerationEI
from neural_observability.theory.linear_families import linear_column
from neural_observability.theory.symmetry import _symbolic_jacobian, named_generators


def _column(factory, drive):
    model = factory()
    return linear_column(model, model.equilibria(drive)[0])


def test_scaling_generators_are_exact_identities():
    for factory, drive in ((JansenRit, 90.0), (wendling_2002, 90.0), (NextGenerationEI, -0.65)):
        col = _column(factory, drive)
        _, jac, _, _ = _symbolic_jacobian(col)
        for g in named_generators(col)[1]:
            res = jac * sp.Matrix([sp.Integer(int(x)) for x in g])
            assert all(sp.simplify(e) == 0 for e in res)


def test_mean_field_time_scale_generator_is_an_exact_identity():
    col = _column(NextGenerationEI, -0.65)
    params, jac, _, _ = _symbolic_jacobian(col)
    p = {str(s): s for s in params}
    d, k, u, w, jii = p["d_i"], p["k_si"], p["u_i"], p["w_i"], p["J_ii"]
    # the four coefficients of S_i/S_e stay fixed while k_si changes
    v = {name: sp.Integer(0) for name in p}
    v["k_si"] = 1 / k
    v["d_i"] = -sp.Rational(1, 2) / d
    v["w_i"] = (k - d) / (u * w)
    v["J_ii"] = (-(d**2 + u * w + u * jii) / k + d - (k - d)) / (u * jii)
    v["J_ie"] = -1 / k
    res = jac * sp.Matrix([v[str(s)] for s in params])
    assert all(sp.cancel(sp.together(e)) == 0 for e in res)
