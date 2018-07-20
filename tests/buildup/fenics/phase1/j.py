import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import utilities


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    number = sym.Symbol('n')
    nabs = ((sym.sign(number) + 1) / 2) * sym.Abs(number)

    csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
    x, f, r, Tref = sym.symbols('x[0], F, R, Tref')

    s1 = nabs.subs(number, (csmax - cse) / csmax) ** (1 - alpha)
    s2 = nabs.subs(number, cse / csmax) ** alpha
    s3 = nabs.subs(number, ce / ce0) ** (1 - alpha)
    sym_flux = k_norm_ref * s1 * s2 * s3
    soc = cse / csmax

    tmpx = sym.Symbol('x')
    uocpneg = cmn.params.neg.Uocp[0].subs(tmpx, soc)
    uocppos = cmn.params.pos.Uocp[0].subs(tmpx, soc)

    uocp = sym.Piecewise((uocpneg, x <= 1 + fem.DOLFIN_EPS), (uocppos, x >= 2 - fem.DOLFIN_EPS), (0, True))
    eta = phis - phie - uocp
    sym_j = sym_flux * (sym.exp((1 - alpha) * f * eta / (r * Tref)) - sym.exp(-alpha * f * eta / (r * Tref)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh = cmn.mesh

    k_norm_ref, csmax, alpha, ce0 = cmn.k_norm_ref, cmn.csmax, cmn.alpha, cmn.ce0
    F, R, Tref = cmn.F, cmn.R, cmn.Tref

    for i, j in enumerate(comsol_sol.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)

        # Turn COMSOL solutions into FEniCS functions
        cse = fem.Function(V)
        ce = fem.Function(V)
        phis = fem.Function(V)
        phie = fem.Function(V)
        cse.vector()[:] = comsol_sol.data.cse[i, fem.dof_to_vertex_map(V)].astype('double')
        ce.vector()[:] = comsol_sol.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')
        phie.vector()[:] = comsol_sol.data.phie[i, fem.dof_to_vertex_map(V)].astype('double')
        phis.vector()[:] = comsol_sol.data.phis[i, fem.dof_to_vertex_map(V)].astype('double')

        # Evaluate expressions
        csmax = fem.interpolate(csmax, V)
        alpha = fem.interpolate(alpha, V)

        j = fem.Expression(sym.printing.ccode(sym_j), csmax=csmax, cse=cse, ce=ce, ce0=ce0, alpha=alpha,
                           k_norm_ref=k_norm_ref, phie=phie, phis=phis, R=R, F=F, Tref=Tref, degree=1)
        u_array[i, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]

    # utilities.overlay_plt(comsol_sol.mesh, time, '$j$', u_array, comsol_sol.data.j)
    utilities.report(comsol_sol.mesh[comsol_sol.neg_ind][:-1], time, u_array[:, comsol_sol.neg_ind][:, :-1],
                     comsol_sol.data.j[:, comsol_sol.neg_ind][:, :-1], '$j_neg$')
    utilities.report(comsol_sol.mesh[comsol_sol.pos_ind], time, u_array[:, comsol_sol.pos_ind],
                     comsol_sol.data.j[:, comsol_sol.pos_ind], '$j_pos$')
    plt.savefig('comsol_compare_j.png')
    plt.show()

    return


if __name__ == '__main__':
    sys.exit(main())
