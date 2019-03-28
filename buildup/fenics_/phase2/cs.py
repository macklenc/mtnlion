import dolfin as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    xbar = fem.Expression(cppcode=utilities.expressions.xbar, markers=dest_markers,
                          neg=dest_x_neg, sep=dest_x_sep, pos=dest_x_pos, degree=1)
    return fem.Expression(cppcode=utilities.expressions.composition, inner=xbar, outer=func, degree=1)


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain

    cs_sol = utilities.create_solution_matrices(len(time), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = \
        utilities.create_solution_matrices(len(time), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]

    d_cs = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs, cs_f = utilities.create_functions(pseudo_domain.V, 3)
    cse = utilities.create_functions(electrode_domain.V, 1)[0]
    cs_cse = utilities.create_functions(cse_domain.V, 1)[0]
    phis_c, phie_c, ce_c = utilities.create_functions(domain.V, 3)

    cse.set_allow_extrapolation(True)

    cse_f = cross_domain(cs_f, electrode_domain.domain_markers,
                         fem.Expression(('x[0]', '1.0'), degree=1),
                         fem.Expression(('0.5*(x[0]+1)', '1.0'), degree=1),
                         fem.Expression(('x[0] - 0.5', '1.0'), degree=1))

    # Uocp = equations.Uocp(cse_1, **cmn.fenics_params)
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
    J = fem.derivative(F, cs_f, d_cs)
    problem = fem.NonlinearVariationalProblem(F, cs_f, J=J)
    solver = fem.NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-10
    prm['newton_solver']['relative_tolerance'] = 1e-9
    prm['newton_solver']['maximum_iterations'] = 5000
    prm['newton_solver']['relaxation_parameter'] = 1.0

    # cse_1.vector()[:] = np.append(comsol.data.cse[0, comsol.neg_ind], comsol.data.cse[0, comsol.pos_ind])

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_ce(t), comsol_phis(t), comsol_phie(t)],
                                   [ce_c, phis_c, phie_c], domain.V, ...)
        cs_1.vector()[:] = comsol_cs(t - dt).astype('double')

        cs_f.assign(cs_1)
        solver.solve()

        cs_cse.assign(fem.interpolate(cs_f, cse_domain.V))
        cse.assign(fem.interpolate(cse_f, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs_f.vector().get_local()  # used to prove that cs computed correctly

    if return_comsol:
        return utilities.interp_time(time, cs_sol), utilities.interp_time(time, pseudo_cse_sol), utilities.interp_time(
            time, cse_sol), comsol
    else:
        return utilities.interp_time(time, cs_sol), utilities.interp_time(time, pseudo_cse_sol), utilities.interp_time(
            time, cse_sol)


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.INFO)
    import numpy as np

    # Times at which to run solver
    if time is None:
        time = np.arange(0.1, 50, 1)
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(time, dt, return_comsol=True)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)

    if not get_test_stats:
        print('cs total normalized RMSE%: {}'.format(utilities.norm_rmse(cs_sol(plot_time), comsol_cs(plot_time))))

        xcoor, cse, neg_ind, pos_ind = utilities.find_cse_from_cs(comsol)
        comsol_pseudo_cse = utilities.interp_time(comsol.time_mesh, cse)

        utilities.report(xcoor[neg_ind], plot_time, pseudo_cse_sol(plot_time)[:, neg_ind],
                         comsol_pseudo_cse(plot_time)[:, neg_ind], 'pseudo $c_{s,e}^{neg}$')
        utilities.save_plot(__file__, 'plots/compare_pseudo_cse_neg.png')
        utilities.report(xcoor[pos_ind], plot_time, pseudo_cse_sol(plot_time)[:, pos_ind],
                         comsol_pseudo_cse(plot_time)[:, pos_ind], 'pseudo $c_{s,e}^{pos}$')
        utilities.save_plot(__file__, 'plots/compare_pseudo_cse_pos.png')

        utilities.report(comsol.mesh[comsol.neg_ind], plot_time, cse_sol(plot_time)[:, comsol.neg_ind],
                         comsol_cse(plot_time)[:, comsol.neg_ind], '$c_{s,e}$')
        utilities.save_plot(__file__, 'plots/compare_cse_neg.png')
        plt.show()
        utilities.report(comsol.mesh[comsol.pos_ind], plot_time, cse_sol(plot_time)[:, comsol.pos_ind],
                         comsol_cse(plot_time)[:, comsol.pos_ind], '$c_{s,e}$')
        utilities.save_plot(__file__, 'plots/compare_cse_pos.png')

        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, cs_sol, comsol_cs)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data


if __name__ == '__main__':
    main()
