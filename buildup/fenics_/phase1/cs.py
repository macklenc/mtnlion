import dolfin as fem
import matplotlib.pyplot as plt

from buildup import common, utilities
from mtnlion.newman import equations

#TODO: CHECK ERROR AFTER FENICS 2018 UPDATE

# essentially dest_x_*** is a converstion from the destination x to the source x, we'll call the source xbar
# then this method returns func(xbar)
def cross_domain(func, dest_markers, dest_x_neg, dest_x_sep, dest_x_pos):
    # NOTE: .cpp_object() will not be required later as per
    # https://bitbucket.org/fenics-project/dolfin/issues/1041/compiledexpression-cant-be-initialized
    # TODO: Use python wrappers
    xbar = fem.CompiledExpression(fem.compile_cpp_code(utilities.expressions.xbar).XBar(),
                                  markers=dest_markers,
                                  neg=dest_x_neg.cpp_object(),
                                  sep=dest_x_sep.cpp_object(),
                                  pos=dest_x_pos.cpp_object(),
                                  degree=1,
    )
    return fem.CompiledExpression(fem.compile_cpp_code(utilities.expressions.composition).Composition(),
                                  inner=xbar.cpp_object(),
                                  outer=func.cpp_object(),
                                  degree=1)


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup()
    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain
    electrode_domain = cmn.electrode_domain

    comsol_j = utilities.interp_time(comsol.time_mesh, comsol.data.j)
    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)

    cs_sol = utilities.create_solution_matrices(len(time), len(pseudo_domain.mesh.coordinates()), 1)[0]
    pseudo_cse_sol = utilities.create_solution_matrices(len(time), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(len(time), len(domain.mesh.coordinates()), 1)[0]

    cs_u = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs = utilities.create_functions(pseudo_domain.V, 2)
    jbar_c = utilities.create_functions(domain.V, 1)[0]
    cse = utilities.create_functions(electrode_domain.V, 1)[0]
    cs_cse = utilities.create_functions(cse_domain.V, 1)[0]

    cse.set_allow_extrapolation(True)

    jhat = cross_domain(
        jbar_c,
        pseudo_domain.domain_markers,
        fem.Expression("x[0]", degree=1),
        fem.Expression("2*x[0]-1", degree=1),
        fem.Expression("x[0] + 0.5", degree=1),
    )

    cse_f = cross_domain(
        cs,
        electrode_domain.domain_markers,
        fem.Expression(("x[0]", "1.0"), degree=1),
        fem.Expression(("0.5*(x[0]+1)", "1.0"), degree=1),
        fem.Expression(("x[0] - 0.5", "1.0"), degree=1),
    )

    neumann = jhat * v * pseudo_domain.ds(5)

    euler = equations.euler(cs_u, cs_1, dtc)
    lhs, rhs = equations.cs(cs_1, v, **cmn.fenics_params, **cmn.fenics_consts)
    F = (lhs * euler - rhs) * pseudo_domain.dx + neumann

    a = fem.lhs(F)
    L = fem.rhs(F)

    for k, t in enumerate(time):
        utilities.assign_functions([comsol_j(t - dt)], [jbar_c], domain.V, ...)
        # TODO: make assignable with utilities.assign_functions
        cs_1.vector()[:] = comsol_cs(t - dt).astype("double")

        fem.solve(a == L, cs)
        cs_cse.assign(fem.interpolate(cs, cse_domain.V))
        cse.assign(fem.interpolate(cse_f, electrode_domain.V))

        pseudo_cse_sol[k, :] = cs_cse.vector().get_local()  # used to show that cs computed correctly
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse, domain.V), domain.V)  # desired result
        # TODO: make usable with get 1d
        cs_sol[k, :] = cs.vector().get_local()  # used to prove that cs computed correctly

    if return_comsol:
        return (
            utilities.interp_time(time, cs_sol),
            utilities.interp_time(time, pseudo_cse_sol),
            utilities.interp_time(time, cse_sol),
            comsol,
        )
    else:
        return (
            utilities.interp_time(time, cs_sol),
            utilities.interp_time(time, pseudo_cse_sol),
            utilities.interp_time(time, cse_sol),
        )


def main(time=None, dt=None, plot_time=None, get_test_stats=False):
    # Quiet
    fem.set_log_level(fem.LogLevel.ERROR)

    # Times at which to run solver
    if time is None:
        time = [0, 5, 10, 15, 20]
    if dt is None:
        dt = 0.1
    if plot_time is None:
        plot_time = time

    cs_sol, pseudo_cse_sol, cse_sol, comsol = run(time, dt, return_comsol=True)
    comsol_cse = utilities.interp_time(comsol.time_mesh, comsol.data.cse)
    comsol_cs = utilities.interp_time(comsol.time_mesh, comsol.data.cs)

    if not get_test_stats:
        print("cs total normalized RMSE%: {}".format(utilities.norm_rmse(cs_sol(plot_time), comsol_cs(plot_time))))

        xcoor, cse, neg_ind, pos_ind = utilities.find_cse_from_cs(comsol)
        comsol_pseudo_cse = utilities.interp_time(comsol.time_mesh, cse)

        utilities.report(
            xcoor[neg_ind],
            plot_time,
            pseudo_cse_sol(plot_time)[:, neg_ind],
            comsol_pseudo_cse(plot_time)[:, neg_ind],
            "pseudo $c_{s,e}^{neg}$",
        )
        utilities.save_plot(__file__, "plots/compare_pseudo_cse_neg.png")
        utilities.report(
            xcoor[pos_ind],
            plot_time,
            pseudo_cse_sol(plot_time)[:, pos_ind],
            comsol_pseudo_cse(plot_time)[:, pos_ind],
            "pseudo $c_{s,e}^{pos}$",
        )
        utilities.save_plot(__file__, "plots/compare_pseudo_cse_pos.png")

        utilities.report(
            comsol.mesh[comsol.neg_ind],
            plot_time,
            cse_sol(plot_time)[:, comsol.neg_ind],
            comsol_cse(plot_time)[:, comsol.neg_ind],
            "$c_{s,e}$",
        )
        utilities.save_plot(__file__, "plots/compare_cse_neg.png")
        plt.show()
        utilities.report(
            comsol.mesh[comsol.pos_ind],
            plot_time,
            cse_sol(plot_time)[:, comsol.pos_ind],
            comsol_cse(plot_time)[:, comsol.pos_ind],
            "$c_{s,e}$",
        )
        utilities.save_plot(__file__, "plots/compare_cse_pos.png")

        plt.show()
    else:
        data = utilities.generate_test_stats(time, comsol, cs_sol, comsol_cs)

        # Separator info is garbage:
        for d in data:
            d[1, ...] = 0

        return data


if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace
    #
    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr
    #
    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
