import sys
from functools import partial

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import utilities
from mtnlion.newman import equations


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    k_norm_ref, csmax, alpha, ce0 = cmn.k_norm_ref, cmn.csmax, cmn.alpha, cmn.ce0
    F, R, Tref = cmn.F, cmn.R, cmn.Tref
    V = domain.V
    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)
    bc = [fem.DirichletBC(V, 0.0, domain.boundary_markers, 1), 0]

    Acell, sigma_eff, L, a_s, F = cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, F

    cse_f = fem.Function(V)
    ce_f = fem.Function(V)
    phis_f = fem.Function(V)  # "previous solution"
    phie_f = fem.Function(V)
    j_f = fem.Function(V)
    phis = fem.Function(V)  # current solution

    j = equations.j(ce_f, cse_f, phie_f, phis_f, csmax, ce0, alpha, k_norm_ref, F, R, Tref, cmn.params.neg.Uocp[0],
                    cmn.params.pos.Uocp[0])
    phis_form = partial(equations.phis, j, a_s, F, sigma_eff, L, phis_f, v, domain.dx((1, 3)), domain.ds(4),
                        nonlin=True)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(comsol.mesh)))
    u_array2 = np.empty((len(time), len(comsol.mesh)))

    comsol_data = zip(comsol.data.cse, comsol.data.ce, comsol.data.phis, comsol.data.phie)
    for i, (cse_t, ce_t, phis_t, phie_t) in enumerate(comsol_data):
        cse_f.vector()[:] = cse_t[fem.dof_to_vertex_map(V)].astype('double')
        ce_f.vector()[:] = ce_t[fem.dof_to_vertex_map(V)].astype('double')
        phie_f.vector()[:] = phie_t[fem.dof_to_vertex_map(V)].astype('double')
        phis_f.vector()[:] = phis_t[fem.dof_to_vertex_map(V)].astype('double')

        bc[1] = fem.DirichletBC(V, phis_t[-1], domain.boundary_markers, 4)
        Feq = phis_form(neumann=fem.Constant(Iapp[i]) / Acell)

        J = fem.derivative(Feq, phis_f, du)
        problem = fem.NonlinearVariationalProblem(Feq, phis_f, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-5
        prm['newton_solver']['relative_tolerance'] = 1e-4
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()

        u_array[i, :] = phis_f.vector().get_local()[fem.vertex_to_dof_map(domain.V)]
        # j_neg = fem.interpolate(j, cmn.neg_V).vector().get_local()[fem.vertex_to_dof_map(cmn.neg_V)]
        # j_pos = fem.interpolate(j, cmn.pos_V).vector().get_local()[fem.vertex_to_dof_map(cmn.pos_V)]
        # jout = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]
        #
        # u_array2[i, :] = -99
        # u_array2[i, 0:len(j_neg)] = j_neg
        # u_array2[i, -len(j_pos):] = j_pos
        # comp = np.array([u_array2[i], jout])
        u_array2[i, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]
        u_array2[i, 20] = j(1)

    utilities.report(comsol.neg, time, u_array[:, comsol.neg_ind],
                     comsol.data.phis[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array[:, comsol.pos_ind],
                     comsol.data.phis[:, comsol.pos_ind], '$\Phi_s^{pos}$')
    plt.show()

    utilities.report(comsol.neg, time, u_array2[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind], '$j^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array2[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind], '$j^{pos}$')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
