import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import common, utilities
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

    neumann = (
        de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_c_1("-")), domain.n("-")) * v("-") * domain.dS(2)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_c_1("+")), domain.n("+")) * v("+") * domain.dS(2)
        + de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_c_1("-")), domain.n("-")) * v("-") * domain.dS(3)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_c_1("+")), domain.n("+")) * v("+") * domain.dS(3)
    )

    # explicit euler
    euler = equations.euler(ce_u, ce_c_1, dtc)
    lhs, rhs1, rhs2 = equations.ce(jbar_c_1, ce_c_1, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * euler - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) - neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_j(t - dt), comsol_ce(t - dt)], [jbar_c_1, ce_c_1], domain.V, ...)
        bc = [
            fem.DirichletBC(domain.V, comsol_ce(t)[0], domain.boundary_markers, 1),
            fem.DirichletBC(domain.V, comsol_ce(t)[-1], domain.boundary_markers, 4),
        ]

        fem.solve(a == L, ce, bc)
        ce_sol[k, :] = utilities.get_1d(ce, domain.V)
        print("t={time}: error = {error}".format(time=t, error=np.abs(ce_sol[k, :] - comsol_ce(t)).max()))

    if return_comsol:
        return utilities.interp_time(time, ce_sol), comsol
    else:
        return utilities.interp_time(time, ce_sol)


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if time is None:
        time = np.arange(0, 50, 5)
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    ce_sol, comsol = run(time, dt, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    if not get_test_stats:
        utilities.report(comsol.mesh, time, ce_sol(plot_time), comsol_ce(plot_time), "$\c_e$")
        utilities.save_plot(__file__, "plots/compare_ce.png")
        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, ce_sol, comsol_ce)

        return data


if __name__ == "__main__":
    main()
