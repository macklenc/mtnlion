import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(start_time, dt, stop_time, return_comsol=False):
    time = np.arange(start_time, stop_time + dt, dt)
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    ce_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c, ce_c, ce_c_1, ce = utilities.create_functions(domain.V, 4)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS
    neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_u('-')), n('-')) * v('-') * dS(2) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_u('+')), n('+')) * v('+') * dS(2) + \
              de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_u('-')), n('-')) * v('-') * dS(3) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_u('+')), n('+')) * v('+') * dS(3)

    euler = equations.euler(ce_u, ce_c_1, dtc)
    lhs, rhs1, rhs2 = equations.ce(jbar_c, ce_u, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * euler - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) + neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    if start_time < dt:
        ce_c_1.assign(cmn.fenics_consts.ce0)
    else:
        utilities.assign_functions([comsol_ce(start_time)], [ce_c_1], domain.V, ...)

    ce_sol[0, :] = utilities.get_1d(ce_c_1, domain.V)

    for k, t in enumerate(time[1:], 1):
        utilities.assign_functions([comsol_j(t)], [jbar_c], domain.V, ...)

        fem.solve(a == L, ce)
        ce_sol[k, :] = utilities.get_1d(ce, domain.V)
        print('t={time}: error = {error}'.format(time=t, error=np.abs(ce_sol[k, :] - comsol_ce(t)).max()))

        ce_c_1.assign(ce)

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
        plot_time = np.arange(start_time, stop_time, (stop_time-start_time)/5)

    ce_sol, comsol = run(start_time, dt, stop_time, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    if not get_test_stats:
        utilities.report(comsol.mesh, plot_time, ce_sol(plot_time), comsol_ce(plot_time), '$c_e$')
        utilities.save_plot(__file__, 'plots/compare_ce_euler.png')
        plt.show()
    else:
        data = utilities.generate_test_stats(plot_time, comsol, ce_sol, comsol_ce)

        return data


if __name__ == '__main__':
    main()
