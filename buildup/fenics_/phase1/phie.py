import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup()

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_ce = utilities.interp_time(comsol.time_mesh, comsol.data.ce)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)

    phie_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    phie_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c, ce_c, phie = utilities.create_functions(domain.V, 3)
    kappa_eff, kappa_Deff = common.kappa_Deff(ce_c, **cmn.fenics_params, **cmn.fenics_consts)

    Lc = cmn.fenics_params.L
    n = domain.n
    dS = domain.dS

    newmann_a = (kappa_eff('-') / Lc('-') * fem.inner(fem.grad(phie_u('-')), n('-')) * v('-') +
                 kappa_eff('+') / Lc('+') * fem.inner(fem.grad(phie_u('+')), n('+')) * v('+')) * (dS(2) + dS(3))

    newmann_L = -(kappa_Deff('-') / Lc('-') * fem.inner(fem.grad(fem.ln(ce_c('-'))), n('-')) * v('-') +
                  kappa_Deff('+') / Lc('+') * fem.inner(fem.grad(fem.ln(ce_c('+'))), n('+')) * v('+')) * (dS(2) + dS(3))

    lhs, rhs = equations.phie(jbar_c, ce_c, phie_u, v, kappa_eff, kappa_Deff, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs - rhs) * domain.dx + newmann_a - newmann_L

    a = fem.lhs(F)
    L = fem.rhs(F)

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_j(t), comsol_ce(t)], [jbar_c, ce_c], domain.V, ...)
        bc = fem.DirichletBC(domain.V, comsol_phie(t)[0], domain.boundary_markers, 1)

        fem.solve(a == L, phie, bc)
        phie_sol[k, :] = utilities.get_1d(phie, domain.V)

    if return_comsol:
        return utilities.interp_time(time, phie_sol), comsol
    else:
        return utilities.interp_time(time, phie_sol)


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]
    plot_time = time

    phie_sol, comsol = run(time, return_comsol=True)
    comsol_phie = utilities.interp_time(comsol.time_mesh, comsol.data.phie)

    utilities.report(comsol.mesh, time, phie_sol(plot_time), comsol_phie(plot_time), '$\Phi_e$')
    utilities.save_plot(__file__, 'plots/compare_phie.png')
    plt.show()


if __name__ == '__main__':
    main()
