import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


# TODO: fix...
def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    ce_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c_1, ce_c, ce_c_1, ce = utilities.create_functions(domain.V, 4)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L

    neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), domain.n('-')) * v('-') * domain.dS(2) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), domain.n('+')) * v('+') * domain.dS(2) + \
              de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), domain.n('-')) * v('-') * domain.dS(3) + \
              de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), domain.n('+')) * v('+') * domain.dS(3)

    # explicit euler
    euler = equations.euler(ce_u, ce_c_1, dtc)
    lhs, rhs = equations.ce(jbar_c_1, ce_c_1, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = lhs * euler * domain.dx - rhs * domain.dx - neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_j(t - dt), comsol_ce(t - dt)], [jbar_c_1, ce_c_1], domain.V, ...)
        bc = [fem.DirichletBC(domain.V, comsol_ce(t)[0], domain.boundary_markers, 1),
              fem.DirichletBC(domain.V, comsol_ce(t)[-1], domain.boundary_markers, 4)]

        fem.solve(a == L, ce, bc)
        ce_sol[k, :] = utilities.get_1d(ce, domain.V)
        print('t={time}: error = {error}'.format(time=t, error=np.abs(ce_sol[k, :] - comsol_ce(t)).max()))

    if return_comsol:
        return utilities.interp_time(time, ce_sol), comsol
    else:
        return utilities.interp_time(time, ce_sol)


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]
    plot_times = time_in
    dt = 0.1

    ce_sol, comsol = run(time_in, dt, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    utilities.report(comsol.mesh, time_in, ce_sol(plot_times), comsol_ce(plot_times), '$\c_e$')
    utilities.save_plot(__file__, 'plots/compare_ce.png')
    plt.show()


if __name__ == '__main__':
    main()
