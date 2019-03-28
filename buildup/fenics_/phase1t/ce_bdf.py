import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from buildup import common, utilities
from mtnlion.newman import equations


def run(start_time, dt, stop_time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar, sol, ce_fem = utilities.create_functions(domain.V, 3)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    if start_time < dt:
        ce0 = np.empty(domain.mesh.coordinates().shape).flatten()
        ce0.fill(cmn.consts.ce0)
    else:
        ce0 = comsol_ce(start_time)

    neumann = (
        de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_fem("-")), n("-")) * v("-") * dS(2)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_fem("+")), n("+")) * v("+") * dS(2)
        + de_eff("-") / Lc("-") * fem.inner(fem.grad(ce_fem("-")), n("-")) * v("-") * dS(3)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce_fem("+")), n("+")) * v("+") * dS(3)
    )

    lhs, rhs1, rhs2 = equations.ce(jbar, ce_fem, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * ce_u - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) + neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    def fun(t, ce):
        utilities.assign_functions([comsol_j(t), ce], [jbar, ce_fem], domain.V, ...)
        fem.solve(a == L, sol)
        return utilities.get_1d(sol, domain.V)

    ce_bdf = integrate.BDF(fun, 0, ce0, 50, atol=1e-6, rtol=1e-5)

    # using standard lists for dynamic growth
    ce_sol = list()
    ce_sol.append(ce0)
    time_vec = list()
    time_vec.append(0)

    # integrate._ivp.bdf.NEWTON_MAXITER = 50
    i = 1
    while ce_bdf.status == "running":
        print(
            "comsol_time step: {:.4e}, comsol_time: {:.4f}, order: {}, step: {}".format(
                ce_bdf.h_abs, ce_bdf.t, ce_bdf.order, i
            )
        )
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
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    [sim_start_time, sim_dt, sim_stop_time] = [0, 0.1, 50]
    plot_times = np.arange(0, 50, 5)

    ce_sol, comsol = run(sim_start_time, sim_dt, sim_stop_time, return_comsol=True)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    utilities.report(comsol.mesh, plot_times, ce_sol(plot_times), comsol_ce(plot_times), "$c_e$")
    utilities.save_plot(__file__, "plots/compare_ce_bdf.png")
    plt.show()


if __name__ == "__main__":
    main()
