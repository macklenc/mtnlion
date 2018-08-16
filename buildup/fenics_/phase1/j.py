import sys

import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations

import numpy as np


def eref_pos():
    x_values = np.array([0.44, 0.467804, 0.495607, 0.523411, 0.551214, 0.579018, 0.606821, 0.634625, 0.662428, 0.690232,
                0.718035, 0.745839, 0.773642, 0.801446, 0.829249, 0.857053, 0.884856, 0.91266, 0.940463, 0.968267,
                0.99607, 1])

    y_values = np.array([4.24783, 4.19155, 4.16236, 4.14308, 4.12451, 4.09837, 4.069, 4.04214, 4.01823, 3.99499,
                         3.97035, 3.94095, 3.90082, 3.8528, 3.8116, 3.78152, 3.75736, 3.735, 3.71117, 3.68231, 3.64192,
                         3.62824])

    mesh = fem.IntervalMesh(len(x_values) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([x_values]).transpose()

    V1 = fem.FunctionSpace(mesh, 'Lagrange', 1)
    eref = fem.Function(V1)
    eref.vector()[:] = y_values[fem.vertex_to_dof_map(V1)]
    eref.set_allow_extrapolation(True)
    # print(eref(0.5))

    return eref

def run(time, return_comsol=False, engine='comsol', form='equation'):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]

    phis_c, phie_c, cse_c, ce_c = utilities.create_functions(domain.V, 4)
    cmn.fenics_params.Uocp_pos = eref_pos()

    # TODO: add forms to j. I.e. equation, interpolation
    jbar = equations.j(ce_c, cse_c, phie_c, phis_c, **cmn.fenics_params, **cmn.fenics_consts, form=form)

    for i in range(len(time)):
        utilities.assign_functions([comsol.data.phis, comsol.data.phie, comsol.data.cse, comsol.data.ce],
                                   [phis_c, phie_c, cse_c, ce_c], domain.V, i)

        j_sol[i, :] = utilities.get_1d(fem.interpolate(jbar, domain.V), domain.V)

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


if __name__ == '__main__':
    sys.exit(main())
