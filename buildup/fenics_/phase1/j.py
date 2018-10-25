import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def eref_pos(cmn):
    x_values = cmn.Uocp_spline.Uocp_pos[:, 0]
    y_values = cmn.Uocp_spline.Uocp_pos[:, 1]

    mesh = fem.IntervalMesh(len(x_values) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([x_values]).transpose()

    V1 = fem.FunctionSpace(mesh, 'Lagrange', 1)
    eref = fem.Function(V1)
    eref.vector()[:] = y_values[fem.vertex_to_dof_map(V1)]
    # ret = fem.conditional(x >= 2, eref, 0)
    # eref.set_allow_extrapolation(True)
    # print(eref(0.5))

    return eref

def eref_neg(cmn):
    x_values = cmn.Uocp_spline.Uocp_neg[:, 0]
    y_values = cmn.Uocp_spline.Uocp_neg[:, 1]

    mesh = fem.IntervalMesh(len(x_values) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([x_values]).transpose()

    V1 = fem.FunctionSpace(mesh, 'Lagrange', 1)
    eref = fem.Function(V1)
    eref.vector()[:] = y_values[fem.vertex_to_dof_map(V1)]
    # ret = fem.conditional(x >= 2, eref, 0)
    # eref.set_allow_extrapolation(True)
    # print(eref(0.5))

    return eref

def run(time, return_comsol=False, engine='comsol', form='equation'):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]

    phis_c, phie_c, cse_c, ce_c, sol = utilities.create_functions(domain.V, 5)

    u = fem.TrialFunction(domain.V)
    v = fem.TestFunction(domain.V)

    # cmn.fenics_params.Uocp_pos = eref_pos(cmn)
    # cmn.fenics_params.Uocp_neg = eref_neg(cmn)
    # cmn.fenics_params.materials = cmn.domain.domain_markers
    #
    # h = fem.interpolate(fem.Expression('x[0] <= 1.0 ? f : (x[0] >= 2.0 ? g : 0)', f=cmn.params.csmax[0], g=cmn.params.csmax[2], degree=1), domain.V)
    # soc = fem.Expression('cse/csmax', cse=cse_c, csmax=cmn.fenics_params.csmax, degree=1)
    # Uocp, neg, pos = common.create_uocp_fenics(cmn.Uocp_spline.Uocp_neg, cmn.Uocp_spline.Uocp_pos, domain.domain_markers, soc)
    # a = fem.dot(u, v) * domain.dx((0, 2))
    # L = Uocp * v * domain.dx((0, 2))


    # TODO: add forms to j. I.e. equation, interpolation
    Uocp = equations.Uocp(cse_c, **cmn.fenics_params)
    jbar = equations.j(ce_c, cse_c, phie_c, phis_c, Uocp, **cmn.fenics_params, **cmn.fenics_consts, form=form, Uocp2=Uocp)
    #
    a = fem.dot(u, v) * domain.dx((0, 2))
    L = jbar * v * domain.dx((0, 2))

    for i in range(len(time)):
        utilities.assign_functions([comsol.data.phis, comsol.data.phie, comsol.data.cse, comsol.data.ce],
                                   [phis_c, phie_c, cse_c, ce_c], domain.V, i)

        fem.solve(a == L, sol)
        j_sol[i, :] = utilities.get_1d(fem.interpolate(sol, domain.V), domain.V)

    if return_comsol:
        return j_sol, comsol
    else:
        return j_sol


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    j_sol, comsol = run(time, return_comsol=True)

    utilities.report(comsol.mesh[comsol.neg_ind], time, j_sol[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind], '$j_{neg}$')
    utilities.save_plot(__file__, 'plots/compare_j_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time, j_sol[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind], '$j_{pos}$')
    utilities.save_plot(__file__, 'plots/comsol_compare_j_pos.png')

    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
