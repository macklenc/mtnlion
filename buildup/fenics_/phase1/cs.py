import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import mtnlion.engine as engine
import utilities


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)
    pseudo_domain = cmn.pseudo_domain

    bmesh = fem.BoundaryMesh(pseudo_domain.mesh, 'exterior')
    cc = fem.MeshFunction('size_t', bmesh, bmesh.topology().dim())
    top = fem.AutoSubDomain(lambda x: (1.0 - fem.DOLFIN_EPS) <= x[1] <= (1.0 + fem.DOLFIN_EPS))
    # top = fem.CompiledSubDomain('near(x[1], b, DOLFIN_EPS)', b=1.0)
    top.mark(cc, 9)
    submesh = fem.SubMesh(bmesh, cc, 9)
    X = fem.FunctionSpace(submesh, 'Lagrange', 1)

    # cc = fem.MeshFunction('size_t', pseudo_domain.mesh, pseudo_domain.mesh.topology().dim()-1)
    # top = fem.AutoSubDomain(lambda x: (1.0 - fem.DOLFIN_EPS) <= x[1] <= (1.0 + fem.DOLFIN_EPS))
    # top.mark(cc, 9)
    # submesh = fem.SubMesh(pseudo_domain.mesh, cc, 9)
    # X = fem.FunctionSpace(submesh, 'Lagrange', 1)
    #
    # print(submesh.coordinates()[fem.dof_to_vertex_map(X)])
    # fem.plot(submesh)
    # plt.show()
    # exit(0)

    cs_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.pseudo_mesh), 1)[0]
    cse_sol = utilities.create_solution_matrices(int(len(time) / 2), len(submesh.coordinates()[:, 0]), 1)[0]

    cs_u = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs, jbar2_interp = utilities.create_functions(pseudo_domain.V, 3)
    jbar_c = utilities.create_functions(domain.V, 1)[0]
    jbar2 = utilities.create_functions(X, 1)[0]
    jbar2.set_allow_extrapolation(True)

    Ds = cmn.fenics_params.Ds_ref
    Rs = cmn.fenics_params.Rs
    ds = pseudo_domain.ds
    dx = pseudo_domain.dx

    j, y, eps = sym.symbols('j x[1] DOLFIN_EPS')
    sym_j = sym.Piecewise((j, (y + eps) >= 1), (0, True))
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    jbar = fem.Expression(sym.printing.ccode(sym_j), j=jbar2, degree=3)
    neumann = dtc * rbar2 * (-jbar) * v * ds(5)
    a = Rs * rbar2 * cs_u * v * dx

    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]

    # Collect common data
    V = pseudo_domain.V

    cs_1 = fem.Function(V)
    cs_1.vector()[:] = comsol.data.cs[-1, :].astype('double')

    # create local variables
    comsol_sol = cmn.comsol_solution

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2
        i = i*2 + 1

        tmpj = np.append(comsol_sol.data.j[i_1, comsol_sol.neg_ind], comsol_sol.data.j[i_1, comsol_sol.pos_ind])
        utilities.assign_functions([comsol_sol.data.j], [jbar_c], domain.V, i_1)
        # jbar2.vector()[:] = tmpj[fem.dof_to_vertex_map(X)].astype('double')
        jbar2.vector()[:] = tmpj.astype('double')
        # jbar2.assign(fem.Constant(0))

        # jbar2.assign(fem.Constant(0))
        # utilities.assign_functions([comsol.data.cs], [cs_1], pseudo_domain.V, i_1) # Already organized????
        cs_1.vector()[:] = comsol.data.cs[i_1].astype('double')

        # Setup equation
        L = Rs*rbar2*cs_1*v*dx - dtc*Ds*rbar2/Rs*fem.dot(cs_1.dx(1), v.dx(1))*dx + neumann

        # Solve
        fem.solve(a == L, cs)
        cs.set_allow_extrapolation(True)
        cse = fem.interpolate(cs, X)
        # asdf = utilities.get_1d(cse, X)
        # fdsa = np.concatenate(
        #     (asdf[:len(comsol_sol.neg)], np.zeros(len(comsol_sol.sep) - 2), asdf[len(comsol_sol.neg):]))
        cse_sol[k, :] = cse.vector().get_local()

        u_nodal_values = cs.vector()
        cs_sol[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(pseudo_domain.V)]
        k += 1

    # exit()
    ## find c_se from c_s
    data = np.append(comsol.pseudo_mesh, comsol.data.cs[1::2].T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    ## find c_se on the pseudo dim coordinates
    # tmpcse = np.concatenate((comsol_sol.data.cse[1::2, comsol_sol.neg_ind], comsol_sol.data.cse[1::2, comsol_sol.pos_ind]), axis=1)
    #
    # utilities.report(xcoor[neg_ind], time_in, cse.T[:, neg_ind], tmpcse[:, neg_ind], '$c_{s,e}^{neg}$')
    # plt.show()
    # utilities.report(xcoor[pos_ind], time_in, cse.T[:, pos_ind], tmpcse[:, pos_ind], '$c_{s,e}^{pos}$')
    # plt.show()

    print(engine.rmse(cse_sol, cse.T) / np.sqrt(np.average(cse.T ** 2)) * 100)

    utilities.report(xcoor[neg_ind], time_in, cse_sol[:, neg_ind],
                     cse.T[:, neg_ind], '$c_{s,e}^{neg}$')
    plt.show()
    utilities.report(xcoor[pos_ind], time_in, cse_sol[:, pos_ind],
                     cse.T[:, pos_ind], '$c_{s,e}^{pos}$')
    plt.show()
    # plt.savefig('comsol_compare_cs.png')
    # plt.show()
    # plt.plot(np.repeat([cmn.comsol_solution.mesh], len(time_in), axis=0).T, cmn.comsol_solution.data.cse[1::2].T)
    # plt.show()

    if return_comsol:
        return cse_sol, comsol
    else:
        return cse_sol


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

    cs_sol, comsol = run(time, dt, return_comsol=True)
    # utilities.report(comsol.mesh, time_in, cs_sol, comsol.data.ce[1::2], '$\c_s$')
    # utilities.save_plot(__file__, 'plots/compare_cs.png')
    # plt.show()


if __name__ == '__main__':
    main()
