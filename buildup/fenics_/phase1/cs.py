import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import mtnlion.engine as engine
import utilities


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsold = common.prepare_comsol_buildup(time)
    pseudo_domain = cmn.pseudo_domain
    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]

    # Collect common data
    V = pseudo_domain.V

    Ds = cmn.fenics_params.Ds_ref
    Rs = cmn.fenics_params.Rs

    cmn1d = common.Common(time)
    cs_data = comsold.data.cs

    t1e = fem.Expression('x[0]', degree=1)
    t1 = fem.interpolate(t1e, V)
    values = t1.vector().get_local()
    t1.vector()[:] = values

    cs_1 = fem.Function(V)
    cs_1.vector()[:] = cs_data[-1, :].astype('double')

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(pseudo_domain.mesh.coordinates()[:])))

    # create local variables
    comsol_sol = cmn.comsol_solution
    dx, ds, bm, dm = pseudo_domain.dx, pseudo_domain.ds, pseudo_domain.boundary_markers, pseudo_domain.domain_markers

    j, y, eps = sym.symbols('j x[1] DOLFIN_EPS')
    sym_j = sym.Piecewise((j, (y + eps) >= 1), (0, True))
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    # cse = fem.Expression('')

    bmesh = fem.BoundaryMesh(pseudo_domain.mesh, 'exterior')
    cc = fem.MeshFunction('size_t', bmesh, bmesh.topology().dim())
    top = fem.AutoSubDomain(lambda x: (1.0 - fem.DOLFIN_EPS) <= x[1] <= (1.0 + fem.DOLFIN_EPS))
    # top = fem.CompiledSubDomain('near(x[1], b, DOLFIN_EPS)', b=1.0)
    top.mark(cc, 9)
    submesh = fem.SubMesh(bmesh, cc, 9)
    # print(submesh.coordinates())
    # fem.plot(submesh)
    # plt.show()
    # exit(0)
    u_array2 = np.empty((len(time_in), len(submesh.coordinates()[:, 0])))

    X = fem.FunctionSpace(submesh, 'Lagrange', 1)
    # plt.plot(np.sort(X.tabulate_dof_coordinates().reshape(-1, 2)[:, 0]),
    #          fem.interpolate(cs_1, X).vector().get_local()[fem.vertex_to_dof_map(X)])
    # plt.show()


    k = 0
    for i, j in enumerate(comsol_sol.data.j[1::2]):
        i_1 = i * 2
        i = i*2 + 1

        # Define function space and basis functions
        V = fem.FunctionSpace(pseudo_domain.mesh, 'Lagrange', 1)
        W = fem.FunctionSpace(cmn1d.domain.mesh, 'Lagrange', 1)
        cs = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        X = fem.FunctionSpace(submesh, 'Lagrange', 1)

        jbar1 = fem.Function(W)
        jbar2 = fem.Function(X)
        tmpj = np.append(comsol_sol.data.j[i_1, comsol_sol.neg_ind], comsol_sol.data.j[i_1, comsol_sol.pos_ind])

        jbar1.vector()[:] = comsol_sol.data.j[i_1, fem.dof_to_vertex_map(W)].astype('double')
        jbar2.vector()[:] = tmpj[fem.dof_to_vertex_map(X)].astype('double')

        jbar2.set_allow_extrapolation(True)
        jbar2 = fem.interpolate(jbar2, V)
        jbar = fem.Expression(sym.printing.ccode(sym_j), j=jbar2, degree=1)

        cs_1 = fem.Function(V)
        cs_1.vector()[:] = cs_data[i_1].astype('double')

        cse_eq = fem.Expression('x[1] >= 1 - DOLFIN_EPS ? cs : 0', cs=cs_1, degree=1)
        bc = [fem.DirichletBC(V, cse_eq, bm, 5)]

        # Setup Neumann BCs
        neumann = dtc*rbar2*(-jbar)*v*ds(5)

        # Setup equation
        a = Rs*rbar2*cs*v*dx

        L = Rs*rbar2*cs_1*v*dx - dtc*Ds*rbar2/Rs*fem.dot(cs_1.dx(1), v.dx(1))*dx + neumann

        # Solve
        cs = fem.Function(V)
        fem.solve(a == L, cs)
        cs.set_allow_extrapolation(True)
        cse = fem.interpolate(cs, X)

        u_array2[k, :] = cse.vector().get_local()

        u_nodal_values = cs.vector()
        u_array[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
        k += 1
    print(utilities.norm_rmse(u_array, cs_data[1::2]))
    print(engine.rmse(u_array, cs_data[1::2]) / (np.max(cs_data[1::2]) - np.min(cs_data[1::2])) * 100)
    print(
        np.average(np.subtract(u_array, cs_data[1::2]), axis=1) / (np.max(cs_data[1::2]) - np.min(cs_data[1::2])) * 100)

    data = np.append(comsold.pseudo_mesh, comsold.data.cs[0::2].T, axis=1)
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]
    data = data[indices]
    data = data[data[:, 0].argsort()]
    xcoor = data[:, 0]
    neg_ind = np.where(xcoor <= 1)[0]
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]

    print(engine.rmse(u_array2, cse.T) / np.sqrt(np.average(cse.T ** 2)) * 100)

    utilities.report(xcoor[neg_ind], time_in, u_array2[:, neg_ind],
                     cse.T[:, neg_ind], '$c_{s,e}^{neg}$')
    plt.show()
    utilities.report(xcoor[pos_ind], time_in, u_array2[:, pos_ind],
                     cse.T[:, pos_ind], '$c_{s,e}^{pos}$')

    plt.savefig('comsol_compare_cs.png')
    plt.show()
    plt.plot(np.repeat([cmn1d.comsol_solution.mesh], len(time_in), axis=0).T, cmn1d.comsol_solution.data.cse[1::2].T)
    plt.show()


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

    run(time, dt, return_comsol=True)


if __name__ == '__main__':
    main()
