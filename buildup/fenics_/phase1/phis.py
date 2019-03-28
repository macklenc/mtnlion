import dolfin as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)

    phis_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phis_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c, phis = utilities.create_functions(domain.V, 2)
    Iapp = fem.Constant(0.0)

    neumann = Iapp / cmn.fenics_consts.Acell * v * domain.ds(4)

    lhs, rhs = equations.phis(jbar_c, phis_u, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs - rhs) * domain.dx((0, 2)) + fem.dot(phis_u, v) * domain.dx(1) - neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_j(t)], [jbar_c], domain.V, ...)
        Iapp.assign(float(cmn.Iapp(t)))
        bc[1] = fem.DirichletBC(domain.V, comsol_phis(t)[comsol.pos_ind][0], domain.boundary_markers, 3)

        fem.solve(a == L, phis, bc)
        phis_sol[k, :] = utilities.get_1d(phis, domain.V)

    if return_comsol:
        return utilities.interp_time(time, phis_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol)


def main(time=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if time is None:
        time = [0, 5, 10, 15, 20]
    if plot_time is None:
        plot_time = time

    phis_sol, comsol = run(time, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)

    if not get_test_stats:
        utilities.report(comsol.neg, time, phis_sol(plot_time)[:, comsol.neg_ind],
                         comsol_phis(plot_time)[:, comsol.neg_ind], '$\Phi_s^{neg}$')
        utilities.save_plot(__file__, 'plots/compare_phis_neg.png')
        plt.show()
        utilities.report(comsol.pos, time, phis_sol(plot_time)[:, comsol.pos_ind],
                         comsol_phis(plot_time)[:, comsol.pos_ind], '$\Phi_s^{pos}$')
        utilities.save_plot(__file__, 'plots/compare_phis_pos.png')
        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, phis_sol, comsol_phis)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data


if __name__ == '__main__':
    main()
