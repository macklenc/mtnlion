import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import common, utilities
from mtnlion.newman import equations


def run(time, dt, solver, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    phis_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phis_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c_, phie_c, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    phis = utilities.create_functions(domain.V, 1)[0]
    Iapp = fem.Constant(0)

    # Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(
        cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos, cse_c, cmn.fenics_params.csmax, utilities
    )
    j = equations.j(ce_c, cse_c, phie_c, phis_c_, Uocp, **cmn.fenics_params, **cmn.fenics_consts)

    neumann = Iapp / cmn.fenics_consts.Acell * v * domain.ds(4)

    lhs, rhs = equations.phis(j, phis_u, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs - rhs) * domain.dx((0, 2)) + fem.dot(phis_u, v) * domain.dx(1) - neumann

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_phis(t - dt)], [phis_c_], domain.V, ...)
        utilities.assign_functions([comsol_phie(t), comsol_ce(t), comsol_cse(t)], [phie_c, ce_c, cse_c], domain.V, ...)
        Iapp.assign(float(cmn.Iapp(t)))
        bc[1] = fem.DirichletBC(domain.V, comsol_phis(t)[comsol.pos_ind][0], domain.boundary_markers, 3)

        solver(fem.lhs(F) == fem.rhs(F), phis, phis_c_, bc)
        phis_sol[k, :] = utilities.get_1d(phis, domain.V)

    if return_comsol:
        return utilities.interp_time(time, phis_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol)


def main():
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    time = np.arange(0, 50, 5)
    sim_dt = 0.1
    plot_time = time

    phis_sol, comsol = run(time, sim_dt, utilities.picard_solver, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)

    utilities.report(
        comsol.neg,
        time,
        phis_sol(plot_time)[:, comsol.neg_ind],
        comsol_phis(plot_time)[:, comsol.neg_ind],
        "$\Phi_s^{neg}$",
    )
    utilities.save_plot(__file__, "plots/compare_phis_neg_picard.png")
    plt.show()
    utilities.report(
        comsol.pos,
        time,
        phis_sol(plot_time)[:, comsol.pos_ind],
        comsol_phis(plot_time)[:, comsol.pos_ind],
        "$\Phi_s^{pos}$",
    )
    utilities.save_plot(__file__, "plots/compare_phis_pos_picard.png")
    plt.show()


if __name__ == "__main__":
    main()
