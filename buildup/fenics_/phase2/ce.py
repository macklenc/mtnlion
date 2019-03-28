import dolfin as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    ce_sol, j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 2)
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c, phie_c, ce_c_, ce_c_1, cse_c_1, j_c_1 = utilities.create_functions(domain.V, 6)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(2) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(2) + \
              de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(3) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(3)

    # Uocp = equations.Uocp(cse_c_1, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos,
                                 cse_c_1, cmn.fenics_params.csmax, utilities)
    j = equations.j(ce_c_1, cse_c_1, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts,
                    dm=domain.domain_markers, V=domain.V)

    euler = equations.euler(ce_c_, ce_c_1, dtc)
    lhs, rhs1, rhs2 = equations.ce(j, ce_c_1, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * euler - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) + neumann

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_ce(t - dt), comsol_cse(t - dt), comsol_phis(t - dt), comsol_phie(t - dt)],
                                   [ce_c_1, cse_c_1, phis_c, phie_c], domain.V, ...)
        utilities.assign_functions([comsol_j(t - dt)], [j_c_1], domain.V, ...)
        ce_c_.assign(ce_c_1)
        bc = fem.DirichletBC(domain.V, comsol_ce(t)[0], domain.boundary_markers, 1)

        J = fem.derivative(F, ce_c_, ce_u)

        # utilities.newton_solver(F, phie_c_, bc, J, domain.V, relaxation=0.1)
        problem = fem.NonlinearVariationalProblem(F, ce_c_, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-7
        prm['newton_solver']['maximum_iterations'] = 5000
        prm['newton_solver']['relaxation_parameter'] = 0.18
        solver.solve()

        # solver(fem.lhs(F) == fem.rhs(F), phie, phie_c_, bc)
        ce_sol[k, :] = utilities.get_1d(ce_c_, domain.V)
        # j_sol[k, :] = utilities.get_1d(fem.interpolate(j, domain.V), domain.V)

    if return_comsol:
        return utilities.interp_time(time, ce_sol), comsol
    else:
        return utilities.interp_time(time, ce_sol)


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    if time is None:
        time = [0, 5, 10, 15, 20]
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    ce_sol, comsol = run(time, dt, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    if not get_test_stats:
        utilities.report(comsol.mesh, time, ce_sol(plot_time), comsol_ce(plot_time), '$\c_e$')
        utilities.save_plot(__file__, 'plots/compare_ce.png')
        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, ce_sol, comsol_ce)

        return data


if __name__ == '__main__':
    main()
