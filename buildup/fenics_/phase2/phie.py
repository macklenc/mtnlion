import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, solver, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    phie_sol, j_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.mesh), 2)
    phie_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    phis_c, phie_c_, ce_c, cse_c = utilities.create_functions(domain.V, 4)
    phie = utilities.create_functions(domain.V, 1)[0]
    kappa_eff, kappa_Deff = common.kappa_Deff(ce_c, **cmn.fenics_params, **cmn.fenics_consts)

    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    newmann_a = (kappa_eff('-') / Lc('-') * fem.inner(fem.grad(phie_u('-')), n('-')) * v('-') +
                 kappa_eff('+') / Lc('+') * fem.inner(fem.grad(phie_u('+')), n('+')) * v('+')) * (dS(2) + dS(3))

    newmann_L = -(kappa_Deff('-') / Lc('-') * fem.inner(fem.grad(fem.ln(ce_c('-'))), n('-')) * v('-') +
                  kappa_Deff('+') / Lc('+') * fem.inner(fem.grad(fem.ln(ce_c('+'))), n('+')) * v('+')) * (dS(2) + dS(3))

    cse_0 = utilities.piecewise(cmn.mesh, cmn.dm, domain.V, 0.325*cmn.params.csmax[0], 0, 0.422*cmn.params.csmax[2])
    j = equations.j(fem.Constant(2000), cse_0, phie_c_, phis_c, **cmn.fenics_params, **cmn.fenics_consts,
                        dm=domain.domain_markers, V=domain.V)

    F = equations.phie(j, ce_c, phie_u, v, domain.dx((0, 2)), kappa_eff, kappa_Deff,
                       **cmn.fenics_params, **cmn.fenics_consts, nonlin=True)
    F += equations.phie(fem.Constant(0), ce_c, phie_u, v, domain.dx(1), kappa_eff, kappa_Deff,
                        **cmn.fenics_params, **cmn.fenics_consts, nonlin=True)

    F += newmann_a - newmann_L

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step
        utilities.assign_functions([comsol.data.phie], [phie_c_], domain.V, i_1)
        utilities.assign_functions([comsol.data.phis, comsol.data.ce, comsol.data.cse],
                                   [phis_c, ce_c, cse_c], domain.V, i)
        bc = fem.DirichletBC(domain.V, comsol.data.phie[i, 0], domain.boundary_markers, 1)

        solver(fem.lhs(F) == fem.rhs(F), phie, phie_c_, bc)
        phie_sol[k, :] = utilities.get_1d(phie, domain.V)
        j_sol[k, :] = utilities.get_1d(fem.interpolate(j, domain.V), domain.V)
        k += 1

    if return_comsol:
        return phie_sol, comsol, j_sol
    else:
        return phie_sol


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

    phie_sol, comsol, j_sol = run(time, utilities.picard_solver, return_comsol=True)
    utilities.report(comsol.mesh, time_in, phie_sol, comsol.data.phie[1::2], '$\Phi_e$')
    utilities.save_plot(__file__, 'plots/compare_phie.png')
    plt.show()

    utilities.report(comsol.mesh[comsol.neg_ind], time_in, j_sol[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind][1::2], '$j_{neg}$')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time_in, j_sol[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind][1::2], '$j_{pos}$')
    plt.show()


if __name__ == '__main__':
    main()
