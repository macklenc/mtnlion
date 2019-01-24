import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, dt, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)

    phis_elem = fem.FiniteElement('CG', domain.mesh.ufl_cell(), 1)
    phie_elem = fem.FiniteElement('CG', domain.mesh.ufl_cell(), 1)

    W_elem = fem.MixedElement([phis_elem, phie_elem])
    W = fem.FunctionSpace(domain.mesh, W_elem)
    du = fem.TrialFunction(W)
    u = fem.Function(W)
    phis, phie = fem.split(u)

    v_phis, v_phie = fem.TestFunctions(W)

    phis_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    phie_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    j_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    bc = [fem.DirichletBC(W.sub(0), 0.0, domain.boundary_markers, 1), 0, 0]

    phis_c_, phie_c_, ce_c, cse_c, j_c = utilities.create_functions(domain.V, 5)
    Iapp = fem.Constant(0)

    # Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos,
                                 cse_c, cmn.fenics_params.csmax, utilities)
    j = equations.j(ce_c, cse_c, u.sub(1), u.sub(0), Uocp, **cmn.fenics_params, **cmn.fenics_consts)
    kappa_eff, kappa_Deff = common.kappa_Deff(ce_c, **cmn.fenics_params, **cmn.fenics_consts)
    # j = j_c

    Lc = cmn.fenics_params.L

    phis_neumann = Iapp / cmn.fenics_consts.Acell * v_phis * domain.ds(4)
    phie_newmann_a = (kappa_eff('-') / Lc('-') * fem.inner(fem.grad(phie('-')), domain.n('-')) * v_phie('-') +
                      kappa_eff('+') / Lc('+') * fem.inner(fem.grad(phie('+')), domain.n('+')) * v_phie('+')) * \
                     (domain.dS(2) + domain.dS(3))

    phie_newmann_L = -(kappa_Deff('-') / Lc('-') * fem.inner(fem.grad(fem.ln(ce_c('-'))), domain.n('-')) * v_phie('-') +
                       kappa_Deff('+') / Lc('+') * fem.inner(fem.grad(fem.ln(ce_c('+'))), domain.n('+')) * v_phie(
            '+')) * \
                     (domain.dS(2) + domain.dS(3))

    phis_lhs, phis_rhs = equations.phis(j, phis, v_phis, **cmn.fenics_params, **cmn.fenics_consts)
    phie_lhs, phie_rhs1, phie_rhs2 = equations.phie(j, ce_c, phie, v_phie, kappa_eff, kappa_Deff,
                                                    **cmn.fenics_params, **cmn.fenics_consts)

    F = (phis_lhs - phis_rhs) * domain.dx((0, 2)) + fem.dot(phis, v_phis) * domain.dx(1) - phis_neumann
    F += (phie_lhs - phie_rhs1) * domain.dx - phie_rhs2 * domain.dx((0, 2)) + phie_newmann_a - phie_newmann_L
    J = fem.derivative(F, u, du)

    # for evaluating j
    u_j = fem.TrialFunction(domain.V)
    v_j = fem.TestFunction(domain.V)
    a_j = fem.dot(u_j, v_j) * domain.dx((0, 2))
    L_j = j * v_j * domain.dx((0, 2))

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_phis(t - dt), comsol_phie(t - dt)], [phis_c_, phie_c_], domain.V, ...)
        utilities.assign_functions([comsol_ce(t), comsol_cse(t), comsol_j(t)],
                                   [ce_c, cse_c, j_c], domain.V, ...)
        fem.assign(u.sub(0), phis_c_)
        fem.assign(u.sub(1), phie_c_)
        Iapp.assign(float(cmn.Iapp(t)))
        bc[1] = fem.DirichletBC(W.sub(0), comsol_phis(t)[comsol.pos_ind][0], domain.boundary_markers, 3)
        bc[2] = fem.DirichletBC(W.sub(1), comsol_phie(t)[0], domain.boundary_markers, 1)

        problem = fem.NonlinearVariationalProblem(F, u, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-8
        prm['newton_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['relaxation_parameter'] = 0.18
        solver.solve()

        fem.solve(a_j == L_j, j_c)

        phis_c_.assign(u.sub(0, True))
        phie_c_.assign(u.sub(1, True))

        # solver(a == L, phis, phis_c_, bc)
        phis_sol[k, :] = utilities.get_1d(phis_c_, domain.V)
        phie_sol[k, :] = utilities.get_1d(phie_c_, domain.V)
        j_sol[k, :] = utilities.get_1d(fem.interpolate(j_c, domain.V), domain.V)

    if return_comsol:
        return utilities.interp_time(time, phis_sol), utilities.interp_time(time, phie_sol), \
               utilities.interp_time(time, j_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol), utilities.interp_time(time, phie_sol), \
               utilities.interp_time(time, j_sol)


def main():
    fem.set_log_level(fem.INFO)

    # Times at which to run solver
    time = [1, 5, 10, 15, 20]
    sim_dt = 0.1
    plot_time = time

    phis_sol, phie_sol, j_sol, comsol = run(time, sim_dt, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)

    utilities.report(comsol.neg, time, phis_sol(plot_time)[:, comsol.neg_ind],
                     comsol_phis(plot_time)[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    utilities.save_plot(__file__, 'plots/coupled_phie_phis_j/compare_phis_neg_newton.png')
    plt.show()
    utilities.report(comsol.pos, time, phis_sol(plot_time)[:, comsol.pos_ind],
                     comsol_phis(plot_time)[:, comsol.pos_ind], '$\Phi_s^{pos}$')
    utilities.save_plot(__file__, 'plots/coupled_phie_phis_j/compare_phis_pos_newton.png')
    plt.show()

    utilities.report(comsol.mesh, time, phie_sol(plot_time), comsol_phie(plot_time), '$\Phi_e$')
    utilities.save_plot(__file__, 'plots/coupled_phie_phis_j/compare_phie.png')
    plt.show()

    utilities.report(comsol.neg, time, j_sol(plot_time)[:, comsol.neg_ind],
                     comsol_j(plot_time)[:, comsol.neg_ind], '$j^{neg}$')
    utilities.save_plot(__file__, 'plots/coupled_phie_phis_j/compare_j_neg_newton.png')
    plt.show()
    utilities.report(comsol.pos, time, j_sol(plot_time)[:, comsol.pos_ind],
                     comsol_j(plot_time)[:, comsol.pos_ind], '$j^{pos}$')
    utilities.save_plot(__file__, 'plots/coupled_phie_phis_j/compare_j_pos_newton.png')
    plt.show()


if __name__ == '__main__':
    main()
