import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def find_cse_from_cs(comsol):
    data = np.append(comsol.pseudo_mesh, comsol.data.cs[1::2].T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    return xcoor, cse, neg_ind, pos_ind


# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    xbar = fem.Expression(cppcode=utilities.expressions.xbar, markers=dest_markers,
                          neg=dest_x_neg, sep=dest_x_sep, pos=dest_x_pos, degree=1)
    return fem.Expression(cppcode=utilities.expressions.composition, inner=xbar, outer=func, degree=1)


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain

    cs_sol = utilities.create_solution_matrices(int(len(time) / 2), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = \
        utilities.create_solution_matrices(int(len(time) / 2), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(int(len(time) / 2), len(domain.mesh.coordinates()), 1)[0]

    d_cs = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs, cs_f = utilities.create_functions(pseudo_domain.V, 3)
    cse_1, cse = utilities.create_functions(electrode_domain.V, 2)
    cs_cse = utilities.create_functions(cse_domain.V, 1)[0]
    phis_c, phie_c, ce_c = utilities.create_functions(domain.V, 3)

    cse.set_allow_extrapolation(True)
    cse_1.set_allow_extrapolation(True)

    cse_f = cross_domain(cs_f, electrode_domain.domain_markers,
                         fem.Expression(('x[0]', '1.0'), degree=1),
                         fem.Expression(('0.5*(x[0]+1)', '1.0'), degree=1),
                         fem.Expression(('x[0] - 0.5', '1.0'), degree=1))

    # Uocp = equations.Uocp(cse_1, **cmn.fenics_params)
    asdf = utilities.piecewise(cmn.electrode_domain.mesh, cmn.electrode_domain.domain_markers, cmn.electrode_domain.V,
                               *cmn.params.csmax)
    cmn.fenics_params.csmax = asdf
    Uocp = equations.Uocp_interp(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos,
                                 cse_f, cmn.fenics_params.csmax, utilities)
    j = equations.j(ce_c, cse_f, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts,
                    dm=domain.domain_markers, V=domain.V)

    jhat = cross_domain(j, pseudo_domain.domain_markers,
                        fem.Expression('x[0]', degree=1),
                        fem.Expression('2*x[0]-1', degree=1),
                        fem.Expression('x[0] + 0.5', degree=1))

    ds = pseudo_domain.ds
    dx = pseudo_domain.dx

    neumann = jhat * v * ds(5)

    euler = equations.euler(cs_f, cs_1, dtc)
    lhs, rhs = equations.cs(cs_f, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = lhs * euler * dx - rhs * dx + neumann

    # cse_1.vector()[:] = np.append(comsol.data.cse[0, comsol.neg_ind], comsol.data.cse[0, comsol.pos_ind])

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step
        utilities.assign_functions([comsol.data.ce, comsol.data.phis, comsol.data.phie],
                                   [ce_c, phis_c, phie_c], domain.V, i)
        utilities.assign_functions(
            [np.append(comsol.data.cse[:, comsol.neg_ind], comsol.data.cse[:, comsol.pos_ind], axis=1)], [cse_1],
            electrode_domain.V, i_1)

        cs_1.vector()[:] = comsol.data.cs[i_1].astype('double')

        # utilities.assign_functions([comsol.data.j], [j_c_1], domain.V, i_1)
        cs_f.assign(cs_1)

        J = fem.derivative(F, cs_f, d_cs)

        # utilities.newton_solver(F, phie_c_, bc, J, domain.V, relaxation=0.1)
        problem = fem.NonlinearVariationalProblem(F, cs_f, J=J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-12
        prm['newton_solver']['relative_tolerance'] = 1e-12
        prm['newton_solver']['maximum_iterations'] = 5000
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()

        cs_cse.assign(fem.interpolate(cs_f, cse_domain.V))
        cse.assign(fem.interpolate(cse_f, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs.vector().get_local()  # used to prove that cs computed correctly
        k += 1

    if return_comsol:
        return cs_sol, pseudo_cse_sol, cse_sol, comsol
    else:
        return cs_sol, pseudo_cse_sol, cse_sol


def main():
    # Quiet
    fem.set_log_level(fem.INFO)

    # Times at which to run solver
    time_in = [15, 25, 35, 45]
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(time, dt, return_comsol=True)

    print('cs total normalized RMSE%: {}'.format(utilities.norm_rmse(cs_sol, comsol.data.cs[1::2])))

    xcoor, cse, neg_ind, pos_ind = find_cse_from_cs(comsol)
    utilities.report(xcoor[neg_ind], time_in, pseudo_cse_sol[:, neg_ind],
                     cse.T[:, neg_ind], 'pseudo $c_{s,e}^{neg}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_neg.png')
    utilities.report(xcoor[pos_ind], time_in, pseudo_cse_sol[:, pos_ind],
                     cse.T[:, pos_ind], 'pseudo $c_{s,e}^{pos}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_pos.png')

    utilities.report(comsol.mesh[comsol.neg_ind], time_in, cse_sol[:, comsol.neg_ind],
                     comsol.data.cse[1::2, comsol.neg_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time_in, cse_sol[:, comsol.pos_ind],
                     comsol.data.cse[1::2, comsol.pos_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_pos.png')

    plt.show()


if __name__ == '__main__':
    main()
