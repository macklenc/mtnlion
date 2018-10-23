import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

from buildup import (common, utilities)
from mtnlion.newman import equations


def find_cse_from_cs(comsol):
    data = np.append(comsol.pseudo_mesh, comsol.data.cs[1::2].T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    return xcoor, cse, neg_ind, pos_ind


# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    xbar = fem.Expression(cppcode=utilities.expressions.xbar, markers=dest_markers,
                          neg=dest_x_neg, sep=dest_x_sep, pos=dest_x_pos, degree=1)
    return fem.Expression(cppcode=utilities.expressions.composition, inner=xbar, outer=func, degree=1)


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)
    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain

    cs_sol = utilities.create_solution_matrices(int(len(time) / 2), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = \
        utilities.create_solution_matrices(int(len(time) / 2), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(int(len(time) / 2), len(domain.mesh.coordinates()), 1)[0]

    cs_u = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs = utilities.create_functions(pseudo_domain.V, 2)
    jbar_c = utilities.create_functions(domain.V, 1)[0]
    cse = utilities.create_functions(electrode_domain.V, 1)[0]
    cs_jbar, cs_cse = utilities.create_functions(cse_domain.V, 2)

    cs_jbar.set_allow_extrapolation(True)
    cse.set_allow_extrapolation(True)

    jbar_to_pseudo = cross_domain(jbar_c, cse_domain.domain_markers,
                                  fem.Expression('x[0]', degree=1),
                                  fem.Expression('2*x[0]-1', degree=1),
                                  fem.Expression('x[0] + 0.5', degree=1))

    cs_cse_to_cse = cross_domain(cs, electrode_domain.domain_markers,
                                 fem.Expression(('x[0]', '1.0'), degree=1),
                                 fem.Expression(('0.5*(x[0]+1)', '1.0'), degree=1),
                                 fem.Expression(('x[0] - 0.5', '1.0'), degree=1))

    ds = pseudo_domain.ds
    dx = pseudo_domain.dx

    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    jbar = fem.Expression('j', j=cs_jbar, degree=1)  # HACK! TODO: figure out a way to make fenics happy with cs_jbar
    neumann = dtc * rbar2 * jbar * v * ds(5)

    F = equations.cs(cs_1, cs_u, v, dx, dtc, **cmn.fenics_params, **cmn.fenics_consts)
    F += neumann

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2
        i = i*2 + 1

        utilities.assign_functions([comsol.data.j], [jbar_c], domain.V, i_1)
        # TODO: make assignable with utilities.assign_functions
        cs_1.vector()[:] = comsol.data.cs[i_1].astype('double')
        cs_jbar.assign(fem.interpolate(jbar_to_pseudo, cse_domain.V))

        fem.solve(fem.lhs(F) == fem.rhs(F), cs)
        cs_cse.assign(fem.interpolate(cs, cse_domain.V))
        cse.assign(fem.interpolate(cs_cse_to_cse, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs.vector().get_local()  # used to prove that cs computed correctly
        k += 1

    if return_comsol:
        return cs_sol, pseudo_cse_sol, cse_sol, comsol
    else:
        return cs_sol, pseudo_cse_sol, cse_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = [15, 25, 35, 45]
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(time, dt, return_comsol=True)

    print('cs total normalized RMSE%: {}'.format(utilities.norm_rmse(cs_sol, comsol.data.cs[1::2])))

    xcoor, cse, neg_ind, pos_ind = find_cse_from_cs(comsol)
    utilities.report(xcoor[neg_ind], time_in, pseudo_cse_sol[:, neg_ind],
                     cse.T[:, neg_ind], 'pseudo $c_{s,e}^{neg}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_neg.png')
    utilities.report(xcoor[pos_ind], time_in, pseudo_cse_sol[:, pos_ind],
                     cse.T[:, pos_ind], 'pseudo $c_{s,e}^{pos}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_pos.png')

    utilities.report(comsol.mesh[comsol.neg_ind], time_in, cse_sol[:, comsol.neg_ind],
                     comsol.data.cse[1::2, comsol.neg_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time_in, cse_sol[:, comsol.pos_ind],
                     comsol.data.cse[1::2, comsol.pos_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_pos.png')

    plt.show()


if __name__ == '__main__':
    main()
