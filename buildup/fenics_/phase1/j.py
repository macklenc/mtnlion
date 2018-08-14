import sys

import fenics as fem
import matplotlib.pyplot as plt

from buildup import (common, utilities)
from mtnlion.newman import equations


def run(time, return_comsol=False, engine='comsol', form='equation'):
    cmn, domain, comsol = common.prepare_comsol_buildup(time)

    j_sol = utilities.create_solution_matrices(len(time), len(comsol.mesh), 1)[0]

    phis_c, phie_c, cse_c, ce_c = utilities.create_functions(domain.V, 4)

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
