import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import mtnlion.engine as engine
import utilities


def main():
    import timeit
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    number = sym.Symbol('n')
    nabs = ((sym.sign(number) + 1) / 2) * sym.Abs(number)

    csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
    s1 = nabs.subs(number, (csmax - cse) / csmax) ** (1 - alpha)
    s2 = nabs.subs(number, cse / csmax) ** alpha
    s3 = nabs.subs(number, ce / ce0) ** (1 - alpha)
    sym_flux = k_norm_ref * s1 * s2 * s3
    soc = cse / csmax

    x, f, r, Tref = sym.symbols('x[0], F, R, Tref')
    uocpneg = -0.16 + 1.32 * sym.exp(-3.0 * soc) + 10.0 * sym.exp(-2000.0 * soc)
    uocppos = 4.19829 + 0.0565661 * sym.tanh(-14.5546 * soc + 8.60942) - 0.0275479 * (
        1. / (0.998432 - soc) ** 0.492465 - 1.90111) - \
              0.157123 * sym.exp(-0.04738 * soc ** 8) + 0.810239 * sym.exp(-40 * (soc - 0.133875))
    uocp = sym.Piecewise((uocpneg, x <= 1 + fem.DOLFIN_EPS), (uocppos, x >= 2 - fem.DOLFIN_EPS), (0, True))
    eta = phis - phie - uocp
    fluix = sym_flux * (sym.exp((1 - alpha) * f * eta / (r * Tref)) - sym.exp(-alpha * f * eta / (r * Tref)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm

    k_norm_ref, csmax, alpha, ce0 = cmn.k_norm_ref, cmn.csmax, cmn.alpha, cmn.ce0
    F, R, Tref = cmn.F, cmn.R, cmn.Tref

    for i, j in enumerate(comsol_sol.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)

        cse = fem.Function(V)
        ce = fem.Function(V)
        phis = fem.Function(V)
        phie = fem.Function(V)
        cse.vector()[:] = comsol_sol.data.cse[i, fem.dof_to_vertex_map(V)].astype('double')
        ce.vector()[:] = comsol_sol.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')
        phie.vector()[:] = comsol_sol.data.phie[i, fem.dof_to_vertex_map(V)].astype('double')
        phis.vector()[:] = comsol_sol.data.phis[i, fem.dof_to_vertex_map(V)].astype('double')

        csmax = fem.interpolate(csmax, V)
        alpha = fem.interpolate(alpha, V)

        # flux = fem.Expression(sym.printing.ccode(sym_flux), csmax=csmax, cse=cse, ce=ce, ce0=ce0, alpha=alpha, k_norm_ref=k_norm_ref, degree=1)
        # eta2 = fem.Expression(sym.printing.ccode(eta), phie=phie, phis=phis, cse=cse, csmax=csmax, degree=1)
        j = fem.Expression(sym.printing.ccode(fluix), csmax=csmax, cse=cse, ce=ce, ce0=ce0, alpha=alpha,
                           k_norm_ref=k_norm_ref, phie=phie, phis=phis, R=R, F=F, Tref=Tref, degree=1)
        u_array[i, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]

    utilities.overlay_plt(comsol_sol.mesh, time, '$j$', u_array, comsol_sol.data.j)
    plt.show()

    exit(0)

    st = timeit.default_timer()
    jneg, jpos = calculate_j(d_comsol, params)
    # plot_j(time, d_comsol, params, jneg, jpos)
    print('Time: {}'.format(timeit.default_timer()-st))
    print(engine.rmse(jneg, d_comsol.data.j[:, d_comsol.neg_ind]))
    print(engine.rmse(jpos, d_comsol.data.j[:, d_comsol.pos_ind]))

    nan_ins = np.array([[0]*len(d_comsol.sep)])
    sep = np.repeat(nan_ins, len(jpos), axis=0)
    utilities.overlay_plt(d_comsol.mesh, time, '$j$', np.concatenate((jneg, sep, jpos), axis=1), d_comsol.data.j)
    plt.savefig('comsol_compare_j.png')
    plt.show()

    # jneg_orig = d_comsol.data.get_solution_in_neg().get_solution_at_time_index(list(map(lambda x: x*10, time))).j
    # jpos_orig = d_comsol.data.get_solution_in_pos().get_solution_at_time_index(list(map(lambda x: x*10, time))).j
    # rmsn = np.sum(np.abs(jneg_orig - jneg), axis=1) / len(d_comsol.data.mesh.neg)
    # maxn = np.max(np.abs(jneg_orig), axis=1)
    # rmsp = rmse(jpos, jpos_orig)
    # maxp = np.max(jpos_orig, axis=1)

    # print('Neg rms: {}'.format(np.log10(rmsn / maxn)))
    # print('Pos rms: {}'.format(np.log10(rmsp / maxp)))

    return


if __name__ == '__main__':
    sys.exit(main())
