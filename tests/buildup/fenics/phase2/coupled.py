import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import j as jeq
import phis as phiseq
import utilities


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

    Acell, sigma_eff, L, a_s, F = cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, F

    j_e = jeq.J(cmn.params.neg.Uocp[0], cmn.params.pos.Uocp[0], domain.V, degree=1)
    phis_e = phiseq.Phis(domain, Acell, sigma_eff, L, a_s, F)
    cse_f = fem.Function(V)
    ce_f = fem.Function(V)
    phis_f = fem.Function(V)
    phie_f = fem.Function(V)
    j_f = fem.Function(V)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(comsol.mesh)))

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
        neumann = fem.Constant(Iapp[i]) / Acell

        phis = fem.Function(V)

        j = j_e.get(csmax, cse_f, ce_f, ce0, alpha, k_norm_ref, phie_f, phis, F, R, Tref)
        # asdf = fem.interpolate(j, V)
        # u_array[i, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]
        # plt.plot(u_array[i])
        # plt.show()
        # asdf.vector()[:] = j_t[fem.dof_to_vertex_map(V)].astype('double')

        # tmp = asdf.vector().get_local()
        # tmp[np.isnan(tmp)] = 0
        # asdf.vector()[:] = tmp
        a, lin = phis_e.get(j, neumann)

        fem.solve(a == lin, phis, bc)
        u_array[i, :] = phis.vector().get_local()[fem.vertex_to_dof_map(domain.V)]

    utilities.report(comsol.neg, time, u_array[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind],
                     '$\Phi_s^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind],
                     '$\Phi_s^{pos}$')
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
