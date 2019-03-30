import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(start_time, dt, stop_time, return_comsol=False):
    time = np.arange(start_time, stop_time + dt, dt)
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)

    phis_elem = fem.FiniteElement('CG', domain.mesh.ufl_cell(), 1)
    phie_elem = fem.FiniteElement('CG', domain.mesh.ufl_cell(), 1)
    ce_elem = fem.FiniteElement('CG', domain.mesh.ufl_cell(), 1)

    W_elem = fem.MixedElement([phis_elem, phie_elem, ce_elem])
    W = fem.FunctionSpace(domain.mesh, W_elem)
    du = fem.TrialFunction(W)
    u = fem.Function(W)
    phis, phie, ce = fem.split(u)

    v_phis, v_phie, v_ce = fem.TestFunctions(W)

    phis_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    phie_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    ce_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    j_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]
    bc = [fem.DirichletBC(W.sub(0), 0.0, domain.boundary_markers, 1), 0, 0]

    phis_c_, phie_c_, ce_c_1, cse_c, j_c = utilities.create_functions(domain.V, 5)
    Iapp = fem.Constant(0)

    # Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    Uocp = equations.Uocp_interp(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos,
                                 cse_c, cmn.fenics_params.csmax, utilities)
    j = equations.j(u.sub(2), cse_c, u.sub(1), u.sub(0), Uocp, **cmn.fenics_params, **cmn.fenics_consts)
    kappa_eff, kappa_Deff = common.kappa_Deff(u.sub(2), **cmn.fenics_params, **cmn.fenics_consts)
    # j = j_c

    Lc = cmn.fenics_params.L
    de_eff = cmn.fenics_params.De_eff

    phis_neumann = Iapp / cmn.fenics_consts.Acell * v_phis * domain.ds(4)
    phie_newmann_a = (kappa_eff('-') / Lc('-') * fem.inner(fem.grad(phie('-')), domain.n('-')) * v_phie('-') +
                      kappa_eff('+') / Lc('+') * fem.inner(fem.grad(phie('+')), domain.n('+')) * v_phie('+')) * \
                     (domain.dS(2) + domain.dS(3))

    phie_newmann_L = -(kappa_Deff('-') / Lc('-') * fem.inner(fem.grad(fem.ln(ce('-'))), domain.n('-')) * v_phie('-') +
                       kappa_Deff('+') / Lc('+') * fem.inner(fem.grad(fem.ln(ce('+'))), domain.n('+')) * v_phie(
            '+')) * \
                     (domain.dS(2) + domain.dS(3))
    ce_neumann = de_eff('-') / Lc('-') * fem.inner(fem.grad(ce('-')), domain.n('-')) * v_ce('-') * domain.dS(2) + \
                 de_eff('+') / Lc('+') * fem.inner(fem.grad(ce('+')), domain.n('+')) * v_ce('+') * domain.dS(2) + \
                 de_eff('-') / Lc('-') * fem.inner(fem.grad(ce('-')), domain.n('-')) * v_ce('-') * domain.dS(3) + \
                 de_eff('+') / Lc('+') * fem.inner(fem.grad(ce('+')), domain.n('+')) * v_ce('+') * domain.dS(3)

    euler = equations.euler(ce, ce_c_1, dtc)

    phis_lhs, phis_rhs = equations.phis(j, phis, v_phis, **cmn.fenics_params, **cmn.fenics_consts)
    phie_lhs, phie_rhs1, phie_rhs2 = equations.phie(j, ce, phie, v_phie, kappa_eff, kappa_Deff,
                                                    **cmn.fenics_params, **cmn.fenics_consts)
    ce_lhs, ce_rhs1, ce_rhs2 = equations.ce(j, ce, v_ce, **cmn.fenics_params, **cmn.fenics_consts)

    F = (phis_lhs - phis_rhs) * domain.dx((0, 2)) + fem.dot(phis, v_phis) * domain.dx(1) - phis_neumann
    F += (phie_lhs - phie_rhs1) * domain.dx - phie_rhs2 * domain.dx((0, 2)) + phie_newmann_a - phie_newmann_L
    F += (ce_lhs * euler - ce_rhs1) * domain.dx - ce_rhs2 * domain.dx((0, 2)) + ce_neumann

    # for evaluating j
    u_j = fem.TrialFunction(domain.V)
    v_j = fem.TestFunction(domain.V)
    a_j = fem.dot(u_j, v_j) * domain.dx((0, 2))
    L_j = j * v_j * domain.dx((0, 2))

    J = fem.derivative(F, u, du)

    # TODO: fix initial conditions
    if start_time < dt:
        ce_c_1.assign(cmn.fenics_consts.ce0)
        utilities.assign_functions([comsol_phis(0), comsol_phie(0),
                                    comsol_j(0)], [phis_c_, phie_c_, j_c], domain.V, ...)
    else:
        utilities.assign_functions([comsol_ce(start_time)], [ce_c_1], domain.V, ...)
        utilities.assign_functions([comsol_phis(start_time - dt), comsol_phie(start_time - dt),
                                    comsol_j(start_time - dt)], [phis_c_, phie_c_, j_c], domain.V, ...)

    phis_sol[0, :] = utilities.get_1d(phis_c_, domain.V)
    phie_sol[0, :] = utilities.get_1d(phie_c_, domain.V)
    ce_sol[0, :] = utilities.get_1d(ce_c_1, domain.V)
    j_sol[0, :] = utilities.get_1d(j_c, domain.V)

    fem.assign(u.sub(0), phis_c_)
    fem.assign(u.sub(1), phie_c_)
    fem.assign(u.sub(2), ce_c_1)

    for k, t in enumerate(time[1:], 1):
        utilities.assign_functions([comsol_cse(t), comsol_j(t)], [cse_c, j_c], domain.V, ...)
        Iapp.assign(float(cmn.Iapp(t)))

        bc[1] = fem.DirichletBC(W.sub(0), comsol_phis(t)[comsol.pos_ind][0], domain.boundary_markers, 3)
        bc[2] = fem.DirichletBC(W.sub(1), comsol_phie(t)[0], domain.boundary_markers, 1)

        problem = fem.NonlinearVariationalProblem(F, u, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-7
        prm['newton_solver']['relative_tolerance'] = 1e-6
        prm['newton_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['relaxation_parameter'] = 0.18
        iterations, converged = solver.solve()

        fem.solve(a_j == L_j, j_c)
        ce_c_1.assign(u.sub(2, True))

        phis_sol[k, :] = utilities.get_1d(u.sub(0, True), domain.V)
        phie_sol[k, :] = utilities.get_1d(u.sub(1, True), domain.V)
        ce_sol[k, :] = utilities.get_1d(u.sub(2, True), domain.V)
        j_sol[k, :] = utilities.get_1d(fem.interpolate(j_c, domain.V), domain.V)

        print('t={time:.3f}: num iterations: {iter}, error = {error:.4e}'.format(
            time=t, iter=iterations, error=np.abs(ce_sol[k, :] - comsol_ce(t)).max()))

    if return_comsol:
        return utilities.interp_time(time, phis_sol), utilities.interp_time(time, phie_sol), \
               utilities.interp_time(time, j_sol), utilities.interp_time(time, ce_sol), comsol
    else:
        return utilities.interp_time(time, phis_sol), utilities.interp_time(time, phie_sol), \
               utilities.interp_time(time, j_sol), utilities.interp_time(time, ce_sol)


def main():
    # fem.set_log_level(fem.INFO)
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    [sim_start_time, sim_dt, sim_stop_time] = [0, 0.1, 50]
    plot_time = np.arange(0, 50, 5)

    phis_sol, phie_sol, j_sol, ce_sol, comsol = run(sim_start_time, sim_dt, sim_stop_time, return_comsol=True)
    comsol_phis = utilities.interp_time(comsol.time_mesh, comsol.data.phis)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)
    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)

    utilities.report(comsol.neg, plot_time, phis_sol(plot_time)[:, comsol.neg_ind],
                     comsol_phis(plot_time)[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_phis_neg_newton.png')
    plt.show()
    utilities.report(comsol.pos, plot_time, phis_sol(plot_time)[:, comsol.pos_ind],
                     comsol_phis(plot_time)[:, comsol.pos_ind], '$\Phi_s^{pos}$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_phis_pos_newton.png')
    plt.show()

    utilities.report(comsol.mesh, plot_time, phie_sol(plot_time), comsol_phie(plot_time), '$\Phi_e$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_phie.png')
    plt.show()

    utilities.report(comsol.mesh, plot_time, ce_sol(plot_time), comsol_ce(plot_time), '$c_e$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_ce_euler.png')
    plt.show()

    utilities.report(comsol.neg, plot_time, j_sol(plot_time)[:, comsol.neg_ind],
                     comsol_j(plot_time)[:, comsol.neg_ind], '$j^{neg}$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_j_neg_newton.png')
    plt.show()
    utilities.report(comsol.pos, plot_time, j_sol(plot_time)[:, comsol.pos_ind],
                     comsol_j(plot_time)[:, comsol.pos_ind], '$j^{pos}$')
    utilities.save_plot(__file__, 'plots/coupled_physical/compare_j_pos_newton.png')
    plt.show()


if __name__ == '__main__':
    main()
