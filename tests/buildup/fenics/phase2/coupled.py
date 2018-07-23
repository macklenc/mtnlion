import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import j as jeq
import phis as phiseq
import utilities


def lambify(sym_j):
    csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
    f, r, Tref = sym.symbols('F, R, Tref')

    return sym.lambdify((csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, f, r, Tref), sym_j)


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    k_norm_ref, csmax, alpha, ce0 = cmn.k_norm_ref, cmn.csmax, cmn.alpha, cmn.ce0
    F, R, Tref = cmn.F, cmn.R, cmn.Tref
    V = domain.V
    v = fem.TestFunction(V)

    Acell, sigma_eff, L, a_s, F = cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, F

    j_e = jeq.J(cmn.params.neg.Uocp[0], cmn.params.pos.Uocp[0], domain.V, degree=1)
    cse_f = fem.Function(V)
    ce_f = fem.Function(V)
    phis_f = fem.Function(V)
    phie_f = fem.Function(V)
    j_f = fem.Function(V)

    dx = (domain.dx(1) + domain.dx(3))
    u = fem.TrialFunction(V)
    phis_e = phiseq.Phis(Acell, sigma_eff, L, a_s, F, u, v, dx, domain.ds(4))

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(comsol.mesh)))
    u_array2 = np.empty((len(time), len(comsol.mesh)))

    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    bc = [fem.DirichletBC(V, 0.0, domain.boundary_markers, 1), 0]
    comsol.data.cse[np.isnan(comsol.data.cse)] = 0
    comsol.data.phis[np.isnan(comsol.data.phis)] = 0
    for i, (cse_t, ce_t, phis_t, phie_t, j_t) in enumerate(
        zip(comsol.data.cse, comsol.data.ce, comsol.data.phis, comsol.data.phie, comsol.data.j)):
        cse_f.vector()[:] = cse_t[fem.dof_to_vertex_map(V)].astype('double')
        ce_f.vector()[:] = ce_t[fem.dof_to_vertex_map(V)].astype('double')
        phie_f.vector()[:] = phie_t[fem.dof_to_vertex_map(V)].astype('double')
        phis_f.vector()[:] = phis_t[fem.dof_to_vertex_map(V)].astype('double')
        j_f.vector()[:] = j_t[fem.dof_to_vertex_map(V)].astype('double')

        bc[1] = fem.DirichletBC(V, phis_t[-1], domain.boundary_markers, 4)
        du = fem.TrialFunction(V)

        phis = fem.Function(V)
        u_ = phis_f

        j = j_e.get(csmax, cse_f, ce_f, ce0, alpha, k_norm_ref, phie_f, u_, F, R, Tref)
        Feq = phis_e.get(j, fem.Constant(Iapp[i]) / Acell)

        a = fem.lhs(Feq)
        lin = fem.rhs(Feq)

        eps = 1.0
        tol = 1e-5
        iter = 0
        maxiter = 25
        while eps > tol and iter < maxiter:
            iter += 1
            fem.solve(a == lin, phis, bc)
            phis.vector()[np.isnan(phis.vector().get_local())] = 0
            diff = phis.vector().get_local() - u_.vector().get_local()
            eps = np.linalg.norm(diff, ord=np.Inf)
            print('iter={}, norm={}'.format(iter, eps))
            u_.assign(phis)

        # J = fem.derivative(Feq, u_, du)
        #
        # problem = fem.NonlinearVariationalProblem(Feq, u_, bc, J)
        # solver = fem.NonlinearVariationalSolver(problem)
        # prm = solver.parameters
        # prm['newton_solver']['absolute_tolerance'] = 1E-8
        # prm['newton_solver']['relative_tolerance'] = 1E-7
        # prm['newton_solver']['maximum_iterations'] = 25
        # prm['newton_solver']['relaxation_parameter'] = 1.0
        # fem.set_log_level(fem.PROGRESS)
        #
        # solver.solve()
        u_array[i, :] = phis.vector().get_local()[fem.vertex_to_dof_map(domain.V)]
        u_array2[i, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(domain.V)]

    utilities.report(comsol.neg, time, u_array[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind],
                     '$\Phi_s^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind],
                     '$\Phi_s^{pos}$')
    plt.show()

    utilities.report(comsol.neg, time, u_array2[:, comsol.neg_ind], comsol.data.j[:, comsol.neg_ind],
                     '$j^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array2[:, comsol.pos_ind], comsol.data.j[:, comsol.pos_ind],
                     '$j^{pos}$')
    plt.show()

    # utilities.report(comsol.mesh[comsol.neg_ind][:-1], time, fenics[:, comsol.neg_ind][:, :-1],
    #                  comsol.data.j[:, comsol.neg_ind][:, :-1], '$j_{neg}$')
    # plt.savefig('comsol_compare_j_neg.png')
    # plt.show()
    # utilities.report(comsol.mesh[comsol.pos_ind], time, fenics[:, comsol.pos_ind],
    #                  comsol.data.j[:, comsol.pos_ind], '$j_{pos}$')
    # plt.savefig('comsol_compare_j_pos.png')
    # plt.show()

    return


if __name__ == '__main__':
    sys.exit(main())
