import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def eref_pos():
    x_values = np.array([0.175, 0.195, 0.215, 0.235, 0.255, 0.275, 0.295, 0.315, 0.335, 0.355, 0.375, 0.395, 0.415,
                         0.435, 0.455, 0.475, 0.495, 0.515, 0.535, 0.555, 0.575, 0.595, 0.615, 0.635, 0.655, 0.675,
                         0.695, 0.715, 0.735, 0.755, 0.775, 0.795, 0.815, 0.835, 0.855, 0.875, 0.895, 0.915, 0.935,
                         0.955, 0.975, 0.995])

    y_values = np.array([4.2763, 4.1898, 4.1507, 4.133, 4.1248, 4.1209, 4.119, 4.1179, 4.1171, 4.1165, 4.116,
                         4.1153, 4.1145, 4.1135, 4.1121, 4.1099, 4.1066, 4.1014, 4.0934, 4.082, 4.067, 4.05,
                         4.0333, 4.0192, 4.0087, 4.0012, 3.996, 3.9923, 3.9893, 3.9867, 3.9841, 3.9813, 3.9783,
                         3.9747, 3.9705, 3.9652, 3.9585, 3.9493, 3.9361, 3.9144, 3.869, 3.5944])

    mesh = fem.IntervalMesh(len(x_values) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([x_values]).transpose()

    V1 = fem.FunctionSpace(mesh, 'Lagrange', 1)
    eref = fem.Function(V1)
    eref.vector()[:] = y_values[fem.vertex_to_dof_map(V1)]
    # ret = fem.conditional(x >= 2, eref, 0)
    # eref.set_allow_extrapolation(True)
    # print(eref(0.5))

    return eref

def eref_neg():
    x_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])

    y_values = np.array([0.9761, 0.8179, 0.6817, 0.5644, 0.4635, 0.3767, 0.3019, 0.2376, 0.1822, 0.1345, 0.0935,
                         0.0582, 0.0278, 0.0016])

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

    phis_c, phie_c, cse_c, ce_c = utilities.create_functions(domain.V, 4)
    cmn.fenics_params.Uocp_pos = eref_pos()
    cmn.fenics_params.Uocp_neg = eref_neg()
    cmn.fenics_params.materials = cmn.domain.domain_markers

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
