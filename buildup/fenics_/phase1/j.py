import sys

import dolfin as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False, form='equation'):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    phis_c, phie_c, cse_c, ce_c, sol = utilities.create_functions(domain.V, 5)

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    if form is 'equation':
        Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    elif form is 'interp':
        Uocp = equations.Uocp_interp(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos,
                                     cse_c, cmn.fenics_params.csmax, utilities)
    else:
        return

    jbar = equations.j(ce_c, cse_c, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts)

    a = fem.dot(u, v) * domain.dx((0, 2))
    L = jbar * v * domain.dx((0, 2))

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_phis(t), comsol_phie(t), comsol_cse(t), comsol_ce(t)],
                                   [phis_c, phie_c, cse_c, ce_c], domain.V, ...)

        fem.solve(a == L, sol)
        j_sol[k, :] = utilities.get_1d(fem.interpolate(sol, domain.V), domain.V)

    if return_comsol:
        return utilities.interp_time(time, j_sol), comsol
    else:
        return utilities.interp_time(time, j_sol)


def main(time=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if time is None:
        time = [0, 5, 10, 15, 20]
    if plot_time is None:
        plot_time = time

    j_sol, comsol = run(time, return_comsol=True, form='interp')
    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)

    if not get_test_stats:
        utilities.report(comsol.mesh[comsol.neg_ind], time, j_sol(plot_time)[:, comsol.neg_ind],
                         comsol_j(plot_time)[:, comsol.neg_ind], '$j_{neg}$')
        utilities.save_plot(__file__, 'plots/compare_j_neg.png')
        plt.show()
        utilities.report(comsol.mesh[comsol.pos_ind], time, j_sol(plot_time)[:, comsol.pos_ind],
                         comsol_j(plot_time)[:, comsol.pos_ind], '$j_{pos}$')
        utilities.save_plot(__file__, 'plots/comsol_compare_j_pos.png')

        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, j_sol, comsol_j)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data

    return 0


if __name__ == '__main__':
    sys.exit(main())
