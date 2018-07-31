import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    phie_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]
    phie_u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    jbar_c, ce_c, phie = utilities.create_functions(domain.V, 3)
    kappa_eff, kappa_Deff = common.kappa_Deff(ce_c, **cmn.params, **cmn.const)

    # TODO: add internal neumann conditions
    a, L = equations.phie(jbar_c, ce_c, phie_u, v, domain.dx, kappa_eff, kappa_Deff,
                          **cmn.params, **cmn.const, nonlin=False)

    for i in range(len(time)):
        utilities.assign_functions([comsol.data.j, comsol.data.ce], [jbar_c, ce_c], domain.V, i)
        bc = fem.DirichletBC(domain.V, comsol.data.phie[i, 0], domain.boundary_markers, 1)

        fem.solve(a == L, phie, bc)
        phie_sol[i, :] = utilities.get_1d(phie, domain.V)

    if return_comsol:
        return phie_sol, comsol
    else:
        return phie_sol


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    phie_sol, comsol = run(time, return_comsol=True)
    utilities.report(comsol.mesh, time, phie_sol, comsol.data.phie, '$\Phi_e$')
    plt.savefig('comsol_compare_phie.png')
    plt.show()


if __name__ == '__main__':
    main()
