import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
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

    jbar = fem.Function(domain.V)
    sol = fem.Function(domain.V)
    ce_fem = fem.Function(domain.V)

    neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_fem('-')), n('-')) * v('-') * dS(2) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_fem('+')), n('+')) * v('+') * dS(2) + \
              de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_fem('-')), n('-')) * v('-') * dS(3) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_fem('+')), n('+')) * v('+') * dS(3)
    ce_eq = equations.ce2(jbar, ce_fem, v, domain.dx, **cmn.fenics_params, **cmn.fenics_consts) - neumann

    def fun(t, ce):
        utilities.assign_functions([comsol_j(t), ce], [jbar, ce_fem], domain.V, ...)
        fem.solve((cmn.fenics_params.eps_e * cmn.fenics_params.L) * ce_u * v * domain.dx == ce_eq, sol)
        return utilities.get_1d(sol, domain.V)

    ce_c.assign(cmn.fenics_consts.ce0)

    tester = integrate.BDF(fun, 0, utilities.get_1d(ce_c, domain.V), 50, atol=1e-9, rtol=1e-6)
    i = 0
    asdf = np.array([utilities.get_1d(ce_c, domain.V)])
    timevec = np.array([0])
    while (tester.status == 'running'):
        print('time step: {:.4e}, time: {:.4f}, order: {}, step: {}'.format(tester.h_abs, tester.t, tester.order, i))
        tester.step()
        timevec = np.append(timevec, tester.t)
        asdf = np.append(asdf, [tester.dense_output()(tester.t)], axis=0)
        i += 1

    plttimes = np.arange(0, 50, 5)
    plotter = interp.interp1d(timevec, asdf, axis=0, kind='cubic')

    if return_comsol:
        return plotter(plttimes), comsol
    else:
        return plotter(plttimes)


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
    # plt.plot(ce_sol.T)
    utilities.report(comsol.mesh, time_in[0::50], ce_sol, comsol.data.ce[0::50], '$c_e$')
    utilities.save_plot(__file__, 'plots/compare_ce_time.png')
    plt.show()


if __name__ == '__main__':
    main()
