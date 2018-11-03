import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as interp

from buildup import (common, utilities)
from mtnlion.newman import equations


def interp_time(data, time):
    y = interp.interp1d(time, data, axis=0, fill_value='extrapolate')
    return y


def find_cse_from_cs(comsol):
    data = np.append(comsol.pseudo_mesh, comsol.data.cs.T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    return xcoor, cse.T, neg_ind, pos_ind


# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    xbar = fem.Expression(cppcode=utilities.expressions.xbar, markers=dest_markers,
                          neg=dest_x_neg, sep=dest_x_sep, pos=dest_x_pos, degree=1)
    return fem.Expression(cppcode=utilities.expressions.composition, inner=xbar, outer=func, degree=1)


def run(comsol_time, dt, stop_time, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(comsol_time)
    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain
    sim_time = np.arange(0, stop_time, dt)

    comsol_j = interp_time(comsol.data.j, comsol_time)

    cs_sol = utilities.create_solution_matrices(len(sim_time), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = \
        utilities.create_solution_matrices(len(sim_time), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(len(sim_time), len(domain.mesh.coordinates()), 1)[0]

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

    a = fem.lhs(F)
    L = fem.rhs(F)
    cs_1.assign(cmn.fenics_params.cs_0)
    k = 0
    for i in sim_time:
        print('time = {}'.format(i))
        utilities.assign_functions([comsol_j(i - dt)], [jbar_c], domain.V, ...)
        # TODO: make assignable with utilities.assign_functions
        # cs_1.vector()[:] = comsol.data.cs[i_1].astype('double')
        cs_jbar.assign(fem.interpolate(jbar_to_pseudo, cse_domain.V))

        fem.solve(a == L, cs)
        cs_1.assign(cs)
        cs_cse.assign(fem.interpolate(cs, cse_domain.V))
        cse.assign(fem.interpolate(cs_cse_to_cse, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs.vector().get_local()  # used to prove that cs computed correctly
        k += 1

    if return_comsol:
        return interp_time(cs_sol, sim_time), interp_time(pseudo_cse_sol, sim_time), interp_time(cse_sol,
                                                                                                 sim_time), comsol
    else:
        return cs_sol, pseudo_cse_sol, cse_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = np.arange(0, 50, 0.1)
    plot_times = np.arange(0, 50, 5)
    dt = 0.1

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(time_in, 0.1, 50, return_comsol=True)

    comsol_cs = interp_time(comsol.data.cs, time_in)
    print('cs total normalized RMSE%: {}'.format(utilities.norm_rmse(cs_sol(time_in), comsol_cs(time_in))))

    xcoor, cse, neg_ind, pos_ind = find_cse_from_cs(comsol)
    cse = interp_time(cse, time_in)
    comsol_cse = interp_time(comsol.data.cse, time_in)

    utilities.report(xcoor[neg_ind], plot_times, pseudo_cse_sol(plot_times)[:, neg_ind],
                     cse(plot_times)[:, neg_ind], 'pseudo $c_{s,e}^{neg}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_neg.png')
    plt.show()
    utilities.report(xcoor[pos_ind], plot_times, pseudo_cse_sol(plot_times)[:, pos_ind],
                     cse(plot_times)[:, pos_ind], 'pseudo $c_{s,e}^{pos}$')
    utilities.save_plot(__file__, 'plots/compare_pseudo_cse_pos.png')
    plt.show()

    utilities.report(comsol.mesh[comsol.neg_ind], plot_times, cse_sol(plot_times)[:, comsol.neg_ind],
                     comsol_cse(plot_times)[:, comsol.neg_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], plot_times, cse_sol(plot_times)[:, comsol.pos_ind],
                     comsol_cse(plot_times)[:, comsol.pos_ind], '$c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_pos.png')

    plt.show()


if __name__ == '__main__':
    main()
