import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


# TODO: fix...
def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    ce_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.mesh), 1)[0]
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c_1, ce_c, ce_c_1, ce = utilities.create_functions(domain.V, 4)

    de_eff = cmn.params.De_eff
    Lc = cmn.params.L
    n = domain.n
    dS = domain.dS
    neumann = dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(2) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(2) + \
              dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(3) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(3)

    a, L = equations.ce_explicit_euler(jbar_c_1, ce_c_1, ce_u, v, domain.dx, dt,
                                       **cmn.params, **cmn.const, nonlin=False)
    L += neumann

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step

        utilities.assign_functions([comsol.data.j, comsol.data.ce], [jbar_c_1, ce_c_1], domain.V, i_1)
        bc = [fem.DirichletBC(domain.V, comsol.data.ce[i, 0], domain.boundary_markers, 1),
              fem.DirichletBC(domain.V, comsol.data.ce[i, -1], domain.boundary_markers, 4)]

        fem.solve(a == L, ce, bc)
        ce_sol[k, :] = utilities.get_1d(ce, domain.V)
        print('t={time}: error = {error}'.format(time=time[i],
                                                 error=np.abs(ce_sol[k, :] - comsol.data.ce[i, :]).max()))
        k += 1

    if return_comsol:
        return ce_sol, comsol
    else:
        return ce_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]
    # time_in = np.arange(0.1, 50, 0.1)
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    ce_sol, comsol = run(time, dt, return_comsol=True)
    utilities.report(comsol.mesh, time_in, ce_sol, comsol.data.ce[1::2], '$\c_e$')
    utilities.save_plot(__file__, 'plots/compare_ce.png')
    plt.show()


if __name__ == '__main__':
    main()
