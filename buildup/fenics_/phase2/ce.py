import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    ce_sol, j_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.mesh), 2)
    ce_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c, phie_c, ce_c_, ce_c_1, cse_c_1, j_c_1 = utilities.create_functions(domain.V, 6)

    de_eff = cmn.fenics_params.De_eff
    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    neumann = dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(2) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(2) + \
              dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_c_1('-')), n('-')) * v('-') * dS(3) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_c_1('+')), n('+')) * v('+') * dS(3)

    Uocp = equations.Uocp(cse_c_1, **cmn.fenics_params)
    j = equations.j(ce_c_1, cse_c_1, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts,
                    dm=domain.domain_markers, V=domain.V)

    F = equations.ce_explicit_euler(j, ce_c_1, ce_c_, v, domain.dx((0, 2)), dt,
                                    **cmn.fenics_params, **cmn.fenics_consts)
    F += equations.ce_explicit_euler(fem.Constant(0), ce_c_1, ce_c_, v, domain.dx(1), dt,
                                     **cmn.fenics_params, **cmn.fenics_consts)

    F += neumann

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step
        utilities.assign_functions([comsol.data.ce, comsol.data.cse, comsol.data.phis, comsol.data.phie],
                                   [ce_c_1, cse_c_1, phis_c, phie_c], domain.V, i_1)
        utilities.assign_functions([comsol.data.j], [j_c_1], domain.V, i_1)
        ce_c_.assign(ce_c_1)
        bc = fem.DirichletBC(domain.V, comsol.data.ce[i, 0], domain.boundary_markers, 1)

        J = fem.derivative(F, ce_c_, ce_u)

        # utilities.newton_solver(F, phie_c_, bc, J, domain.V, relaxation=0.1)
        problem = fem.NonlinearVariationalProblem(F, ce_c_, bc, J)
        solver = fem.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-8
        prm['newton_solver']['relative_tolerance'] = 1e-7
        prm['newton_solver']['maximum_iterations'] = 5000
        prm['newton_solver']['relaxation_parameter'] = 0.18
        solver.solve()

        # solver(fem.lhs(F) == fem.rhs(F), phie, phie_c_, bc)
        ce_sol[k, :] = utilities.get_1d(ce_c_, domain.V)
        j_sol[k, :] = utilities.get_1d(fem.interpolate(j, domain.V), domain.V)
        k += 1

    if return_comsol:
        return ce_sol, comsol, j_sol
    else:
        return ce_sol


def main():
    # Quiet
    fem.set_log_level(fem.INFO)

    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]
    # time_in = np.arange(0.1, 50, 0.1)
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    ce_sol, comsol, j_sol = run(time, dt, return_comsol=True)
    utilities.report(comsol.mesh, time_in, ce_sol, comsol.data.ce[1::2], '$\c_e$')
    utilities.save_plot(__file__, 'plots/compare_ce.png')
    plt.show()

    utilities.report(comsol.mesh[comsol.neg_ind], time_in, j_sol[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind][1::2], '$j_{neg}$')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time_in, j_sol[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind][1::2], '$j_{pos}$')
    plt.show()


if __name__ == '__main__':
    main()
