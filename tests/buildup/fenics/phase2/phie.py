import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import utilities
from mtnlion.newman import equations


def phi_e():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm
    Lc, a_s, eps_e, sigma_eff, brug_kappa = common.collect(cmn.params, 'L', 'a_s', 'eps_e', 'sigma_eff', 'brug_kappa')
    F, t_plus, R, T = common.collect(cmn.const, 'F', 't_plus', 'R', 'Tref')

    x = sym.Symbol('ce')
    y = sym.Symbol('x')
    kp = cmn.const.kappa_ref[0].subs(y, x)

    dfdc = sym.Symbol('dfdc')
    # dfdc = 0
    kd = fem.Constant(2) * R * T / F * (fem.Constant(1) + dfdc) * (t_plus - fem.Constant(1))
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)

    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    phie = fem.TrialFunction(V)
    phie_s = fem.Function(V)
    v = fem.TestFunction(V)
    jbar = fem.Function(V)
    ce = fem.Function(V)

    kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce, degree=1)
    kappa_eff = kappa_ref * eps_e ** brug_kappa
    kappa_Deff = kappa_D * kappa_ref * eps_e

    # phi_e = Phie(sigma_eff, Lc, a_s, F, eps_e, t_plus, brug_kappa, kappa_eff, kappa_Deff, phie, v, dx, ds)
    # F = phi_e.get(ce, jbar, fem.Constant(0))

    a, Lin = equations.phie(jbar, ce, Lc, a_s, F, kappa_eff, kappa_Deff, phie, v, dx, nonlin=False)

    for i, j in enumerate(comsol_sol.data.j):
        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, comsol_sol.data.phie[i, 0], bm, 1)]
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')
        ce.vector()[:] = comsol_sol.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')

        # a, L = fem.lhs(F), fem.rhs(F)

        # Solve
        fem.solve(a == Lin, phie_s, bc)
        u_nodal_values = phie_s.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]

    utilities.report(comsol_sol.mesh, time, u_array, comsol_sol.data.phie, '$\Phi_e$')

    plt.savefig('comsol_compare_phie.png')
    plt.show()


if __name__ == '__main__':
    phi_e()
