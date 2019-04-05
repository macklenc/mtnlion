import dolfin as fem
import matplotlib.pyplot as plt

from buildup import common, utilities
from mtnlion.newman import equations

# DOESN'T WORK


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)

    # pseudo_domain = cmn.pseudo_domain
    electrode_domain = cmn.electrode_domain

    phis_elem = fem.FiniteElement("CG", electrode_domain.mesh.ufl_cell(), 1)
    phie_elem = fem.FiniteElement("CG", domain.mesh.ufl_cell(), 1)
    ce_elem = fem.FiniteElement("CG", domain.mesh.ufl_cell(), 1)
    # cs_elem = fem.FiniteElement('CG', pseudo_domain.mesh.ufl_cell(), 1)

    W_elem = fem.MixedElement([phis_elem, phie_elem, ce_elem])
    W = fem.FunctionSpace(domain.mesh, W_elem)
    u = fem.Function(W)
    phis, phie, ce = fem.split(u)
    ce_1 = fem.Function(domain.V)
    du = fem.TrialFunction(W)
    v_phis, v_phie, v_ce = fem.TestFunctions(W)

    phis_sol = utilities.create_solution_matrices(len(time), len(electrode_domain.mesh.coordinates()), 1)[0]
    phie_sol, ce_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 2)
    bc = [fem.DirichletBC(W.sub(0), 0.0, electrode_domain.boundary_markers, 1), 0]

    phis_c_, phie_c, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    Iapp = fem.Constant(0)

    Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    j = equations.j(u.sub(2), cse_c, u.sub(1), u.sub(0), Uocp, **cmn.fenics_params, **cmn.fenics_consts)
    kappa_eff, kappa_Deff = common.kappa_Deff(u.sub(2), **cmn.fenics_params, **cmn.fenics_consts)

    Lc = cmn.fenics_params.L
    de_eff = cmn.fenics_params.De_eff
    phis_neumann = Iapp / cmn.fenics_consts.Acell * v_phis * electrode_domain.ds(4)
    phie_newmann_a = (
        kappa_eff("-") / Lc("-") * fem.inner(fem.grad(phie("-")), domain.n("-")) * v_phie("-")
        + kappa_eff("+") / Lc("+") * fem.inner(fem.grad(phie("+")), domain.n("+")) * v_phie("+")
    ) * (domain.dS(2) + domain.dS(3))

    phie_newmann_L = -(
        kappa_Deff("-") / Lc("-") * fem.inner(fem.grad(fem.ln(ce_c("-"))), domain.n("-")) * v_phie("-")
        + kappa_Deff("+") / Lc("+") * fem.inner(fem.grad(fem.ln(ce_c("+"))), domain.n("+")) * v_phie("+")
    ) * (domain.dS(2) + domain.dS(3))

    ce_neumann = (
        de_eff("-") / Lc("-") * fem.inner(fem.grad(ce("-")), domain.n("-")) * v_ce("-") * domain.dS(2)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce("+")), domain.n("+")) * v_ce("+") * domain.dS(2)
        + de_eff("-") / Lc("-") * fem.inner(fem.grad(ce("-")), domain.n("-")) * v_ce("-") * domain.dS(3)
        + de_eff("+") / Lc("+") * fem.inner(fem.grad(ce("+")), domain.n("+")) * v_ce("+") * domain.dS(3)
    )

    euler = equations.euler(ce, ce_1, dtc)
    phis_lhs, phis_rhs = equations.phis(j, phis, v_phis, **cmn.fenics_params, **cmn.fenics_consts)
    phie_lhs, phie_rhs1, phie_rhs2 = equations.phie(
        j, ce_c, phie, v_phie, kappa_eff, kappa_Deff, **cmn.fenics_params, **cmn.fenics_consts
    )
    ce_lhs, ce_rhs1, ce_rhs2 = equations.ce(j, ce, v_ce, **cmn.fenics_params, **cmn.fenics_consts)

    F = (phis_lhs - phis_rhs) * domain.dx((0, 2)) + fem.dot(phis, v_phis) * domain.dx(1) - phis_neumann
    F += (phie_lhs - phie_rhs1) * domain.dx - phie_rhs2 * domain.dx((0, 2)) + phie_newmann_a - phie_newmann_L
    F += (ce_lhs * euler - ce_rhs1) * domain.dx - ce_rhs2 * domain.dx((0, 2)) + ce_neumann

    for k, t in enumerate(time):
        utilities.assign_functions(
            [comsol_phis(t), comsol_phie(t), comsol_ce(t), comsol_cse(t)],
            [phis_c_, phie_c, ce_c, cse_c],
            domain.V,
            Ellipsis,
        )
        # utilities.assign_functions([comsol_ce(t - dt)],
        #                            [ce_1], domain.V, ...)
        Iapp.assign(float(cmn.Iapp(t)))
        bc[1] = fem.DirichletBC(W.sub(0), comsol_phis(t)[comsol.pos_ind][0], electrode_domain.boundary_markers, 3)

        # u.sub(2).assign(ce_c)
        J = fem.derivative(F, u, du)
        problem = fem.NonlinearVariationalProblem(F, u, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm["newton_solver"]["absolute_tolerance"] = 1e-8
        prm["newton_solver"]["relative_tolerance"] = 1e-7
        prm["newton_solver"]["maximum_iterations"] = 100
        prm["newton_solver"]["relaxation_parameter"] = 1.0
        solver.solve()

        ce_1.assign(u.sub(2))
        # solver(a == L, phis, phis_c_, bc)
        # phis_sol[k, :] = utilities.get_1d(phis_c_, domain.V)

    if return_comsol:
        return utilities.interp_time(time, phis_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol)


def main():
    fem.set_log_level(fem.LogLevel.INFO)

    # Times at which to run solver
    time = [0.1, 5, 10, 15, 20]
    sim_dt = 0.1
    plot_time = time

    phis_sol, comsol = run(time, sim_dt, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)

    utilities.report(
        comsol.neg,
        time,
        phis_sol(plot_time)[:, comsol.neg_ind],
        comsol_phis(plot_time)[:, comsol.neg_ind],
        "$\Phi_s^{neg}$",
    )
    utilities.save_plot(__file__, "plots/compare_phis_neg_newton.png")
    plt.show()
    utilities.report(
        comsol.pos,
        time,
        phis_sol(plot_time)[:, comsol.pos_ind],
        comsol_phis(plot_time)[:, comsol.pos_ind],
        "$\Phi_s^{pos}$",
    )
    utilities.save_plot(__file__, "plots/compare_phis_pos_newton.png")
    plt.show()


if __name__ == "__main__":
    main()
