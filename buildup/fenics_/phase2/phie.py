import dolfin as fem
import matplotlib.pyplot as plt

from buildup import common, utilities
from mtnlion.newman import equations


def run(time, dt, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    phie_sol, j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 2)
    phie_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c, phie_c_, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    kappa_eff, kappa_Deff = common.kappa_Deff(ce_c, **cmn.fenics_params, **cmn.fenics_consts)

    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    newmann_a = (
        kappa_eff("-") / Lc("-") * fem.inner(fem.grad(phie_c_("-")), n("-")) * v("-")
        + kappa_eff("+") / Lc("+") * fem.inner(fem.grad(phie_c_("+")), n("+")) * v("+")
    ) * (dS(2) + dS(3))

    newmann_L = -(
        kappa_Deff("-") / Lc("-") * fem.inner(fem.grad(fem.ln(ce_c("-"))), n("-")) * v("-")
        + kappa_Deff("+") / Lc("+") * fem.inner(fem.grad(fem.ln(ce_c("+"))), n("+")) * v("+")
    ) * (dS(2) + dS(3))

    # Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(
        cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos, cse_c, cmn.fenics_params.csmax, utilities
    )
    j = equations.j(
        ce_c,
        cse_c,
        phie_c_,
        phis_c,
        Uocp,
        **cmn.fenics_params,
        **cmn.fenics_consts,
        dm=domain.domain_markers,
        V=domain.V,
    )

    lhs, rhs1, rhs2 = equations.phie(
        j, ce_c, phie_c_, v, kappa_eff, kappa_Deff, **cmn.fenics_params, **cmn.fenics_consts
    )
    F = (lhs - rhs1) * domain.dx - rhs2 * domain.dx((0, 2)) + newmann_a - newmann_L

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_phie(t - dt)], [phie_c_], domain.V, ...)
        utilities.assign_functions([comsol_phis(t), comsol_ce(t), comsol_cse(t)], [phis_c, ce_c, cse_c], domain.V, ...)
        bc = fem.DirichletBC(domain.V, comsol_phie(t)[0], domain.boundary_markers, 1)

        J = fem.derivative(F, phie_c_, phie_u)

        # utilities.newton_solver(F, phie_c_, bc, J, domain.V, relaxation=0.1)
        problem = fem.NonlinearVariationalProblem(F, phie_c_, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = 1e-8
        prm["newton_solver"]["relative_tolerance"] = 1e-7
        prm["newton_solver"]["maximum_iterations"] = 5000
        prm["newton_solver"]["relaxation_parameter"] = 0.17
        solver.solve()

        # solver(fem.lhs(F) == fem.rhs(F), phie, phie_c_, bc)
        phie_sol[k, :] = utilities.get_1d(phie_c_, domain.V)
        # j_sol[k, :] = utilities.get_1d(fem.interpolate(j, domain.V), domain.V)

    if return_comsol:
        return utilities.interp_time(time, phie_sol), comsol
    else:
        return utilities.interp_time(time, phie_sol)


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.ERROR)
    import numpy as np

    # Times at which to run solver
    if time is None:
        time = np.arange(0.1, 50, 1)
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    phie_sol, comsol = run(time, dt, return_comsol=True)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    if not get_test_stats:
        utilities.report(comsol.mesh, time, phie_sol(plot_time), comsol_phie(plot_time), "$\Phi_e$")
        utilities.save_plot(__file__, "plots/compare_phie.png")
        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, phie_sol, comsol_phie)

        return data


if __name__ == "__main__":
    main()
