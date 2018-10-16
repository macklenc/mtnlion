import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, solver, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    phis_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.mesh), 1)[0]
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phis_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c_, phie_c, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    phis = utilities.create_functions(domain.V, 1)[0]
    Iapp = fem.Constant(0)

    # TODO: Fix j
    j = equations.j(ce_c, cse_c, phie_c, phis_c_, **cmn.fenics_params, **cmn.fenics_consts)
    # j = equations.j_new(ce_c, cse_c, phie_c, phis_c_, **cmn.fenics_params, **cmn.fenics_consts,
    #                     dm=domain.domain_markers, V=domain.V)

    a, L = equations.phis(j, phis_u, v, domain.dx((0, 2)), **cmn.fenics_params, **cmn.fenics_consts,
                          neumann=Iapp / cmn.fenics_consts.Acell, ds=domain.ds(4), nonlin=False)
    a += fem.dot(phis_u, v) * domain.dx(1)

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step
        utilities.assign_functions([comsol.data.phis], [phis_c_], domain.V, i_1)
        utilities.assign_functions([comsol.data.phie, comsol.data.ce, comsol.data.cse],
                                   [phie_c, ce_c, cse_c], domain.V, i)
        Iapp.assign(cmn.Iapp[i])
        bc[0] = fem.DirichletBC(domain.V, comsol.data.phis[i, 0], domain.boundary_markers, 1)
        bc[1] = fem.DirichletBC(domain.V, comsol.data.phis[i, -1], domain.boundary_markers, 4)

        solver(a == L, phis, phis_c_, bc)
        phis_sol[k, :] = utilities.get_1d(phis, domain.V)
        k += 1

    if return_comsol:
        return phis_sol, comsol
    else:
        return phis_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]
    # time_in = np.arange(0.1, 50, 0.1)
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    phis_sol, comsol = run(time, utilities.picard_solver, return_comsol=True)
    utilities.report(comsol.neg, time_in, phis_sol[:, comsol.neg_ind],
                     comsol.data.phis[:, comsol.neg_ind][1::2], '$\Phi_s^{neg}$')
    utilities.save_plot(__file__, 'plots/compare_phis_neg.png')
    plt.show()
    utilities.report(comsol.pos, time_in, phis_sol[:, comsol.pos_ind],
                     comsol.data.phis[:, comsol.pos_ind][1::2], '$\Phi_s^{pos}$')
    utilities.save_plot(__file__, 'plots/compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    main()
