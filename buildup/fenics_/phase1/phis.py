import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    phis_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phis_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c, phis = utilities.create_functions(domain.V, 2)
    Iapp = fem.Constant(0)

    F = equations.phis(jbar_c, phis_u, v, domain.dx((0, 2)), **cmn.fenics_params, **cmn.fenics_consts,
                       neumann=Iapp / cmn.fenics_consts.Acell, ds=domain.ds(4))
    F += fem.dot(phis_u, v) * domain.dx(1)
    # same as
    # a2, L2 = equations.phis(fem.Constant(0), phis_u, v, domain.dx(1), **cmn.params, **cmn.const, nonlin=False)
    # a += a2
    # L += L2

    for i in range(len(time)):
        utilities.assign_functions([comsol.data.j], [jbar_c], domain.V, i)
        Iapp.assign(cmn.Iapp[i])
        bc[1] = fem.DirichletBC(domain.V, comsol.data.phis[i, comsol.pos_ind][0], domain.boundary_markers, 3)

        fem.solve(fem.lhs(F) == fem.rhs(F), phis, bc)
        phis_sol[i, :] = utilities.get_1d(phis, domain.V)

    if return_comsol:
        return phis_sol, comsol
    else:
        return phis_sol


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    phis_sol, comsol = run(time, return_comsol=True)
    utilities.report(comsol.neg, time, phis_sol[:, comsol.neg_ind],
                     comsol.data.phis[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    utilities.save_plot(__file__, 'plots/compare_phis_neg.png')
    plt.show()
    utilities.report(comsol.pos, time, phis_sol[:, comsol.pos_ind],
                     comsol.data.phis[:, comsol.pos_ind], '$\Phi_s^{pos}$')
    utilities.save_plot(__file__, 'plots/compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    main()
