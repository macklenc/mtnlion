import dolfin as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, dt, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    phis_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phis_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c_, phie_c, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    Iapp = fem.Constant(0)

    Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    j = equations.j(ce_c, cse_c, phie_c, phis_c_, Uocp, **cmn.fenics_params, **cmn.fenics_consts)

    neumann = Iapp / cmn.fenics_consts.Acell * v * domain.ds(4)

    lhs, rhs = equations.phis(j, phis_c_, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs - rhs) * domain.dx((0, 2)) + fem.dot(phis_c_, v) * domain.dx(1) - neumann

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_phis(t - dt)], [phis_c_], domain.V, ...)
        utilities.assign_functions([comsol_phie(t), comsol_ce(t), comsol_cse(t)],
                                   [phie_c, ce_c, cse_c], domain.V, ...)
        Iapp.assign(float(cmn.Iapp(t)))
        bc[1] = fem.DirichletBC(domain.V, comsol_phis(t)[comsol.pos_ind][0], domain.boundary_markers, 3)

        J = fem.derivative(F, phis_c_, phis_u)
        problem = fem.NonlinearVariationalProblem(F, phis_c_, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()

        # solver(a == L, phis, phis_c_, bc)
        phis_sol[k, :] = utilities.get_1d(phis_c_, domain.V)

    if return_comsol:
        return utilities.interp_time(time, phis_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol)


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if time is None:
        time = [0, 5, 10, 15, 20]
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    phis_sol, comsol = run(time, dt, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)

    if not get_test_stats:
        utilities.report(comsol.neg, time, phis_sol(plot_time)[:, comsol.neg_ind],
                         comsol_phis(plot_time)[:, comsol.neg_ind], '$\Phi_s^{neg}$')
        utilities.save_plot(__file__, 'plots/compare_phis_neg_newton.png')
        plt.show()
        utilities.report(comsol.pos, time, phis_sol(plot_time)[:, comsol.pos_ind],
                         comsol_phis(plot_time)[:, comsol.pos_ind], '$\Phi_s^{pos}$')
        utilities.save_plot(__file__, 'plots/compare_phis_pos_newton.png')
        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, phis_sol, comsol_phis)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data


if __name__ == '__main__':
    main()
