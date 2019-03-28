import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import common, utilities
from mtnlion.newman import equations


def run(start_time, dt, stop_time, return_comsol=False):
    time = np.arange(start_time, stop_time + dt, dt)
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    ce_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c, phie_c, ce_c_, ce_c_1, cse_c = utilities.create_functions(domain.V, 5)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    neumann = (
        de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_c_("-")), n("-")) * v("-") * dS(2)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_c_("+")), n("+")) * v("+") * dS(2)
        + de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_c_("-")), n("-")) * v("-") * dS(3)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_c_("+")), n("+")) * v("+") * dS(3)
    )

    # Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(
        cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos, cse_c, cmn.fenics_params.csmax, utilities
    )
    j = equations.j(
        ce_c_,
        cse_c,
        phie_c,
        phis_c,
        Uocp,
        **cmn.fenics_params,
        **cmn.fenics_consts,
        dm=domain.domain_markers,
        V=domain.V,
    )

    euler = equations.euler(ce_c_, ce_c_1, dtc)
    lhs, rhs1, rhs2 = equations.ce(j, ce_c_, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * euler - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) + neumann
    J = fem.derivative(F, ce_c_, ce_u)
    problem = fem.NonlinearVariationalProblem(F, ce_c_, J=J)
    solver = fem.NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1e-8
    prm["newton_solver"]["relative_tolerance"] = 1e-7
    prm["newton_solver"]["maximum_iterations"] = 5000
    prm["newton_solver"]["relaxation_parameter"] = 0.18

    if start_time < dt:
        ce_c_1.assign(cmn.fenics_consts.ce0)
    else:
        utilities.assign_functions([comsol_ce(start_time)], [ce_c_1], domain.V, ...)

    ce_sol[0, :] = utilities.get_1d(ce_c_1, domain.V)
    ce_c_.assign(ce_c_1)

    for k, t in enumerate(time[1:], 1):
        utilities.assign_functions(
            [comsol_cse(t), comsol_phis(t), comsol_phie(t)], [cse_c, phis_c, phie_c], domain.V, ...
        )

        iterations, converged = solver.solve()

        ce_c_1.assign(ce_c_)

        ce_sol[k, :] = utilities.get_1d(ce_c_, domain.V)
        print(
            "t={time:.3f}: num iterations: {iter}, error = {error:.4e}".format(
                time=t, iter=iterations, error=np.abs(ce_sol[k, :] - comsol_ce(t)).max()
            )
        )

    if return_comsol:
        return utilities.interp_time(time, ce_sol), comsol
    else:
        return utilities.interp_time(time, ce_sol)


def main(start_time=None, dt=None, stop_time=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if start_time is None:
        start_time = 0
    if stop_time is None:
        stop_time = 50
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = np.arange(start_time, stop_time, (stop_time - start_time) / 5)

    ce_sol, comsol = run(start_time, dt, stop_time, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    if not get_test_stats:
        utilities.report(comsol.mesh, plot_time, ce_sol(plot_time), comsol_ce(plot_time), "$c_e$")
        utilities.save_plot(__file__, "plots/compare_ce_euler.png")

        plt.show()
    else:
        data = utilities.generate_test_stats(plot_time, comsol, ce_sol, comsol_ce)

        return data


if __name__ == "__main__":
    main()
