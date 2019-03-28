import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from buildup import common, utilities
from mtnlion.newman import equations


# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    xbar = fem.Expression(
        cppcode=utilities.expressions.xbar,
        markers=dest_markers,
        neg=dest_x_neg,
        sep=dest_x_sep,
        pos=dest_x_pos,
        degree=1,
    )
    return fem.Expression(cppcode=utilities.expressions.composition, inner=xbar, outer=func, degree=1)


def run(start_time, dt, stop_time, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()
    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain
    time = np.arange(start_time, stop_time, dt)

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    cs_sol = utilities.create_solution_matrices(len(time), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = utilities.create_solution_matrices(len(time), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]

    cs_u = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs = utilities.create_functions(pseudo_domain.V, 2)
    jbar_c, cse_c = utilities.create_functions(electrode_domain.V, 2)
    cse = utilities.create_functions(electrode_domain.V, 1)[0]
    cs_cse = utilities.create_functions(cse_domain.V, 1)[0]
    phis_c, phie_c, ce_c = utilities.create_functions(domain.V, 3)

    cse.set_allow_extrapolation(True)

    cse_f = cross_domain(
        cs,
        electrode_domain.domain_markers,
        fem.Expression(("x[0]", "1.0"), degree=1),
        fem.Expression(("0.5*(x[0]+1)", "1.0"), degree=1),
        fem.Expression(("x[0] - 0.5", "1.0"), degree=1),
    )

    # Uocp = equations.Uocp(cse_1, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(
        cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos, cse, cmn.fenics_params.csmax, utilities
    )
    j = equations.j(
        ce_c, cse, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts, dm=domain.domain_markers, V=domain.V
    )

    jhat = cross_domain(
        j,
        pseudo_domain.domain_markers,
        fem.Expression("x[0]", degree=1),
        fem.Expression("2*x[0]-1", degree=1),
        fem.Expression("x[0] + 0.5", degree=1),
    )

    ds = pseudo_domain.ds
    dx = pseudo_domain.dx

    if start_time < dt:  # TODO implement real cs0 here
        # cs_1.assign(cmn.fenics_params.cs_0)
        # cs0 = np.empty(domain.mesh.coordinates().shape).flatten()
        # cs0.fill(cmn.consts.ce0)
        cs0 = comsol_cs(start_time)
        cse0 = comsol_cse(start_time)
    else:
        cs0 = comsol_cs(start_time)
        cse0 = comsol_cse(start_time)

    neumann = jhat * v * ds(5)

    euler = equations.euler(cs, cs_1, dtc)
    lhs, rhs = equations.cs(cs, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = lhs * euler * dx - rhs * dx + neumann

    J = fem.derivative(F, cs, cs_u)
    problem = fem.NonlinearVariationalProblem(F, cs, J=J)
    solver = fem.NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 2e-7
    prm["newton_solver"]["relative_tolerance"] = 1e-6
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["relaxation_parameter"] = 1.0

    cs_1.vector()[:] = cs0
    cs_sol[0, :] = cs0
    cse_sol[0, :] = cse0

    cse_x = interpolate.interp1d(comsol.mesh, comsol_cse(start_time), fill_value="extrapolate")
    cse_coor = cse_domain.mesh.coordinates()[fem.dof_to_vertex_map(cse_domain.V)]
    cse_tr_coor = np.array(
        [cse_coor[i, 0] if cse_coor[i, 0] <= 1 else cse_coor[i, 0] + 0.5 for i in range(len(cse_coor[:, 0]))]
    )
    pseudo_cse_sol[0, :] = cse_x(cse_tr_coor)
    # pseudo_cse_sol[0, :] = np.append(comsol_cse(start_time)[comsol.neg_ind], comsol_cse(start_time)[comsol.pos_ind])

    cs.assign(cs_1)
    cse.assign(fem.interpolate(cse_f, electrode_domain.V))
    for k, t in enumerate(time[1:], 1):
        # print('time = {}'.format(t))

        # utilities.assign_functions([comsol_j(t)], [jbar_c], domain.V, ...)
        utilities.assign_functions(
            [np.append(comsol_j(t)[comsol.neg_ind], comsol_j(t)[comsol.pos_ind])], [jbar_c], electrode_domain.V, ...
        )
        utilities.assign_functions(
            [np.append(comsol_cse(t)[comsol.neg_ind], comsol_cse(t)[comsol.pos_ind])], [cse_c], electrode_domain.V, ...
        )
        utilities.assign_functions(
            [comsol_ce(t), comsol_phis(t), comsol_phie(t)], [ce_c, phis_c, phie_c], domain.V, ...
        )

        solver.solve()
        # if 9.7 < t < 12:
        #     plt.plot(utilities.get_1d(cse_c, electrode_domain.V))
        #     plt.plot(utilities.get_1d(fem.interpolate(cse_f, electrode_domain.V), electrode_domain.V), 'r')
        #     plt.show()
        cs_1.assign(cs)
        cs_cse.assign(fem.interpolate(cs, cse_domain.V))
        cse.assign(fem.interpolate(cse_f, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs.vector().get_local()  # used to prove that cs computed correctly
        print("t={time:.3f}: error = {error:.4e}".format(time=t, error=np.abs(cs_sol[k, :] - comsol_cs(t)).max()))

    if return_comsol:
        return (
            utilities.interp_time(time, cs_sol),
            utilities.interp_time(time, pseudo_cse_sol),
            utilities.interp_time(time, cse_sol),
            comsol,
        )
    else:
        return cs_sol, pseudo_cse_sol, cse_sol


def main(start_time=None, dt=None, stop_time=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    if start_time is None:
        start_time = 0
    if stop_time is None:
        stop_time = 50
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = np.arange(start_time, stop_time, dt)

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(start_time, dt, stop_time, return_comsol=True)

    xcoor, cse, neg_ind, pos_ind = utilities.find_cse_from_cs(comsol)
    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    cse = utilities.interp_time(comsol.time_mesh, cse)

    if not get_test_stats:
        print(
            "cs total normalized RMSE%: {}".format(
                utilities.norm_rmse(cs_sol(comsol.time_mesh), comsol_cs(comsol.time_mesh))
            )
        )

        utilities.report(
            xcoor[neg_ind],
            plot_time,
            pseudo_cse_sol(plot_time)[:, neg_ind],
            cse(plot_time)[:, neg_ind],
            "pseudo $c_{s,e}^{neg}$",
        )
        utilities.save_plot(__file__, "plots/compare_pseudo_cse_neg_euler.png")

        utilities.report(
            xcoor[pos_ind],
            plot_time,
            pseudo_cse_sol(plot_time)[:, pos_ind],
            cse(plot_time)[:, pos_ind],
            "pseudo $c_{s,e}^{pos}$",
        )
        utilities.save_plot(__file__, "plots/compare_pseudo_cse_pos_euler.png")

        utilities.report(
            comsol.mesh[comsol.neg_ind],
            plot_time,
            cse_sol(plot_time)[:, comsol.neg_ind],
            comsol_cse(plot_time)[:, comsol.neg_ind],
            "$c_{s,e}$",
        )
        utilities.save_plot(__file__, "plots/compare_cse_neg_euler.png")
        plt.show()

        utilities.report(
            comsol.mesh[comsol.pos_ind],
            plot_time,
            cse_sol(plot_time)[:, comsol.pos_ind],
            comsol_cse(plot_time)[:, comsol.pos_ind],
            "$c_{s,e}$",
        )
        utilities.save_plot(__file__, "plots/compare_cse_pos_euler.png")
        plt.show()
    else:
        data = utilities.generate_test_stats(plot_time, comsol, cs_sol, comsol_cs)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data


if __name__ == "__main__":
    main()
