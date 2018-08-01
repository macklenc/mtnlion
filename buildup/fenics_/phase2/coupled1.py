import sys
from functools import partial

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def main():
    # Times at which to run solver
    time_in = [0.1, 5, 9.9, 10, 10.1, 15, 20]

    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    k_norm_ref, csmax, alpha, L, a_s, sigma_eff = \
        common.collect(cmn.fenics_params, 'k_norm_ref', 'csmax', 'alpha', 'L', 'a_s', 'sigma_eff')
    F, R, Tref, ce0, Acell = common.collect(cmn.fenics_consts, 'F', 'R', 'Tref', 'ce0', 'Acell')
    V = domain.V
    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)
    bc = [fem.DirichletBC(V, 0.0, domain.boundary_markers, 1), 0]

    cse_f = fem.Function(V)
    ce_f = fem.Function(V)
    phis_f = fem.Function(V)  # "previous solution"
    phie_f = fem.Function(V)

    j = equations.j(ce_f, cse_f, phie_f, phis_f, csmax, ce0, alpha, k_norm_ref, F, R, Tref,
                    cmn.fenics_params.Uocp[0][0],
                    cmn.fenics_params.Uocp[2][0], dm=domain.domain_markers)
    phis_form = partial(equations.phis, j, phis_f, v, domain.dx((0, 2)),
                        **cmn.fenics_params, **cmn.fenics_consts, ds=domain.ds(4), nonlin=True)

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(comsol.mesh)))
    u_array2 = np.empty((len(time_in), len(comsol.mesh)))

    k = 0
    for i in range(len(time_in)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step

        cse_f.vector()[:] = comsol.data.cse[i][fem.dof_to_vertex_map(V)].astype('double')
        ce_f.vector()[:] = comsol.data.ce[i][fem.dof_to_vertex_map(V)].astype('double')
        phie_f.vector()[:] = comsol.data.phie[i][fem.dof_to_vertex_map(V)].astype('double')
        phis_f.vector()[:] = comsol.data.phis[i_1][fem.dof_to_vertex_map(V)].astype('double')

        bc[1] = fem.DirichletBC(V, comsol.data.phis[i][-1], domain.boundary_markers, 4)
        Feq = phis_form(neumann=fem.Constant(Iapp[i]) / Acell) + fem.inner(phis_f, v) * domain.dx(1)

        J = fem.derivative(Feq, phis_f, du)
        problem = fem.NonlinearVariationalProblem(Feq, phis_f, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()

        u_array[k, :] = phis_f.vector().get_local()[fem.vertex_to_dof_map(domain.V)]
        u_array2[k, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]
        k += 1

    d = dict()
    d['x'] = comsol.mesh
    d['ce'] = comsol.data.ce[1::2]
    d['cse'] = comsol.data.cse[1::2]
    d['phie'] = comsol.data.phie[1::2]
    d['phis'] = u_array

    neg_params = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in cmn.params.items()}
    d = dict(d, **neg_params)

    def filter(x, sel='neg'):
        if sel is 'neg':
            ind0 = 0
            ind1 = cmn.comsol_solution.neg_ind
        else:
            ind0 = 2
            ind1 = cmn.comsol_solution.pos_ind

        if isinstance(x, list):
            return x[ind0]

        if isinstance(x, np.ndarray):
            if len(x.shape) > 1:
                return x[:, ind1]

            return x[ind1]

        return x

    neg = dict(map(lambda x: (x[0], filter(x[1], 'neg')), d.items()))
    dta = equations.eval_j(**neg, **cmn.consts)

    utilities.report(comsol.neg, time_in, u_array[:, comsol.neg_ind],
                     comsol.data.phis[:, comsol.neg_ind][1::2], '$\Phi_s^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time_in, u_array[:, comsol.pos_ind],
                     comsol.data.phis[:, comsol.pos_ind][1::2], '$\Phi_s^{pos}$')
    plt.show()

    utilities.report(comsol.neg, time_in, dta,
                     comsol.data.j[:, comsol.neg_ind][1::2], '$j^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time_in, u_array2[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind][1::2], '$j^{pos}$')
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
