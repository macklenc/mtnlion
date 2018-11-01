import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as interp

from buildup import (common, utilities)
from mtnlion.newman import equations


def interp_time(data, time):
    y = interp.interp1d(time, data, axis=0, fill_value='extrapolate')
    return y


# TODO: fix...
def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    comsol_j = interp_time(comsol.data.j, time)
    comsol_ce = interp_time(comsol.data.ce, time)

    ce_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c_1, ce_c, ce_c_1, ce = utilities.create_functions(domain.V, 4)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS
    neumann = dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(2) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(2) + \
              dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(3) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(3)

    F = equations.ce_explicit_euler(jbar_c_1, ce_c_1, ce_u, v, domain.dx, dt,
                                    **cmn.fenics_params, **cmn.fenics_consts)
    F -= neumann

    ce_c_1.assign(cmn.fenics_consts.ce0)
    utilities.assign_functions([comsol_j(0)], [jbar_c_1], domain.V, ...)
    ce_sol[0, :] = cmn.consts.ce0

    k = 1
    c_time = dt
    stop_time = 50
    while c_time < stop_time:
        fem.solve(fem.lhs(F) == fem.rhs(F), ce)
        ce_sol[k, :] = utilities.get_1d(ce, domain.V)
        print('t={time}: error = {error}'.format(time=c_time,
                                                 error=np.abs(ce_sol[k, :] - comsol_ce(c_time)).max()))

        utilities.assign_functions([comsol_j(c_time)], [jbar_c_1], domain.V, ...)
        c_time += dt
        k += 1

    if return_comsol:
        return ce_sol, comsol
    else:
        return ce_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    # time_in = [0.1, 5, 10, 15, 20]
    time_in = np.arange(0, 50, 0.1)
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    ce_sol, comsol = run(time_in, dt, return_comsol=True)
    utilities.report(comsol.mesh, time_in, ce_sol, ce_sol, '$\c_e$')
    # utilities.save_plot(__file__, 'plots/compare_ce.png')
    plt.show()


if __name__ == '__main__':
    main()
