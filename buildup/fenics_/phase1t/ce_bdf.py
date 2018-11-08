import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import interpolate as interp

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(comsol_time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup(comsol_time)
    comsol_j = utilities.interp_time(comsol_time, comsol.data.j)

    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar, sol, ce_fem = utilities.create_functions(domain.V, 3)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS
    ce0 = np.empty(domain.mesh.coordinates().shape).flatten()
    ce0.fill(cmn.consts.ce0)

    neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_fem('-')), n('-')) * v('-') * dS(2) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_fem('+')), n('+')) * v('+') * dS(2) + \
              de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_fem('-')), n('-')) * v('-') * dS(3) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_fem('+')), n('+')) * v('+') * dS(3)

    lhs, rhs = equations.ce(jbar, ce_fem, v, **cmn.fenics_params, **cmn.fenics_consts)
    ce_eq = rhs * domain.dx - neumann

    def fun(t, ce):
        utilities.assign_functions([comsol_j(t), ce], [jbar, ce_fem], domain.V, ...)
        fem.solve(lhs * ce_u * domain.dx == ce_eq, sol)
        return utilities.get_1d(sol, domain.V)

    ce_bdf = integrate.BDF(fun, 0, ce0, 50, atol=1e-6, rtol=1e-5)
    # using standard lists for dynamic growth
    ce_sol = list()
    ce_sol.append(ce0)
    time_vec = list()
    time_vec.append(0)

    # integrate._ivp.bdf.NEWTON_MAXITER = 50
    i = 1
    while ce_bdf.status == 'running':
        print('comsol_time step: {:.4e}, comsol_time: {:.4f}, order: {}, step: {}'.format(ce_bdf.h_abs, ce_bdf.t,
                                                                                          ce_bdf.order, i))
        ce_bdf.step()
        time_vec.append(ce_bdf.t)
        ce_sol.append(ce_bdf.dense_output()(ce_bdf.t))
        i += 1

    ce_sol = np.array(ce_sol)
    time_vec = np.array(time_vec)

    if return_comsol:
        return utilities.interp_time(time_vec, ce_sol), comsol
    else:
        return utilities.interp_time(time_vec, ce_sol)


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    # time_in = [0.1, 5, 10, 15, 20]
    time_in = np.arange(0, 50, 0.1)
    plot_times = np.arange(0, 50, 5)

    ce_sol, comsol = run(time_in, return_comsol=True)
    # plt.plot(ce_sol.T)
    utilities.report(comsol.mesh, plot_times, ce_sol(plot_times),
                     utilities.interp_time(time_in, comsol.data.ce)(plot_times), '$c_e$')
    utilities.save_plot(__file__, 'plots/compare_ce_time.png')
    plt.show()


if __name__ == '__main__':
    main()
