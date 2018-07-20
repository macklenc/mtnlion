import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import mtnlion.loader as loader
import utilities


def c_e():
    # Times at which to run solver
    time_in = [0.1, 5, 10, 15, 20]
    dt = 0.1
    dtc = fem.Constant(dt)
    time = [None]*(len(time_in)*2)
    time[::2] = [t-dt for t in time_in]
    time[1::2] = time_in

    # Collect common data
    mesh = fem.Mesh('../../../reference/comsol_solution/cs.xml')
    file_data = loader.collect_files(['../../../reference/comsol_solution/cs.csv.bz2'], format_key=comsol.format_name, loader=loader.load_csv_file)
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    dofs = V.tabulate_dof_coordinates().reshape(-1, 2)
    # fem.plot(mesh)
    # plt.show()
    transform = []
    for i in dofs:
        ind1 = np.where(np.abs(file_data['cs'][:, 0] - i[0]) <= 1e-5)
        ind2 = np.where(np.abs(file_data['cs'][:, 1] - i[1]) <= 1e-5)

        if len(ind1[0]) > 0 and len(ind2[0]) > 0:
            transform.append(np.intersect1d(ind1, ind2)[0])
            if len(np.intersect1d(ind1, ind2)) > 1:
                raise ValueError('Too many matching indices')
        else:
            raise ValueError('Missing indices, check tolerances')

    cs_data1 = file_data['cs'][transform]
    cs_data = cs_data1[:, 2:]

    # print(fem.vertex_to_dof_map(V))
    # print(mesh.coordinates()[:])
    cmn1d = common.Common(time)
    cmn = common.Common2(time, mesh)

    cs_data1 = np.array(cs_data1[:, np.append([0, 1], cmn.time_ind + 2)])
    cs_data = np.array(cs_data[:, cmn.time_ind]).T

    t1e = fem.Expression('x[0]', degree=1)
    t1 = fem.interpolate(t1e, V)
    values = t1.vector().get_local()
    t1.vector()[:] = values
    # fem.plot(t1)
    # plt.show()
    # exit(0)

    # mgx, mgy = np.mgrid(cs_data1[:, 0], cs_data1[:, 1])
    # plt.pcolormesh(mgx, mgy, cs_data[-1])

    cs_1 = fem.Function(V)
    cs_1.vector()[:] = cs_data[-1, :].astype('double')
    # fem.plot(cs_1)
    # plt.show()
    # exit()

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(mesh.coordinates()[:])))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm

    Ds = cmn.Ds
    Rs = cmn.Rs

    j, y, eps = sym.symbols('j x[1] DOLFIN_EPS')
    sym_j = sym.Piecewise((j, (y + eps) >= 1), (0, True))
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    # cse = fem.Expression('')

    bmesh = fem.BoundaryMesh(mesh, 'exterior')
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

    # ofile = fem.File('u.pvd')

    k = 0
    for i, j in enumerate(comsol_sol.data.j[1::2]):
        i_1 = i * 2
        i = i*2 + 1

        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        W = fem.FunctionSpace(cmn1d.mesh, 'Lagrange', 1)
        cs = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        X = fem.FunctionSpace(submesh, 'Lagrange', 1)

        jbar1 = fem.Function(W)
        jbar2 = fem.Function(X)
        tmpj = np.append(comsol_sol.data.j[i_1, comsol_sol.neg_ind], comsol_sol.data.j[i_1, comsol_sol.pos_ind])
        # plt.plot(tmpj)
        # plt.show()
        # plt.plot(comsol_sol.data.j[i_1])
        # plt.show()
        jbar1.vector()[:] = comsol_sol.data.j[i_1, fem.dof_to_vertex_map(W)].astype('double')
        jbar2.vector()[:] = tmpj[fem.dof_to_vertex_map(X)].astype('double')

        # gdim = cmn1d.mesh.geometry().dim()
        # from_dof_coords = W.tabulate_dof_coordinates().reshape(-1, gdim)
        # to_dof_coords = X.tabulate_dof_coordinates().reshape(-1, 2)[:, 0]
        # bnd_to_glob_map = {}
        # z = 0
        #
        # msh = fem.vertex_to_dof_map(W)
        # bnd = fem.vertex_to_dof_map(X)
        #
        # z = 0
        # for i in msh:
        #     coord = W.tabulate_dof_coordinates().reshape(-1, gdim)[i]
        #     if coord <= 1 or coord >= 2:
        #         bnd_to_glob_map[bnd[z]] = i
        #         z += 1
        #
        # for bnd_dof, mesh_dof in bnd_to_glob_map.items():
        #     jbar2.vector()[bnd_dof] = jbar1.vector().get_local()[mesh_dof]
        #     print('from {}, {}'.format(from_dof_coords[mesh_dof], jbar1.vector().get_local()[mesh_dof]))
        #     print('to {}, {}'.format(to_dof_coords[bnd_dof], jbar2.vector().get_local()[bnd_dof]))
        # jbar2 = fem.interpolate(jbar1, V)
        # plt.plot(comsol_sol.mesh, comsol_sol.data.j[i_1])
        # plt.title('comsol')
        # plt.show()
        # plt.plot(np.sort(W.tabulate_dof_coordinates()), jbar1.vector().get_local()[fem.vertex_to_dof_map(W)])
        # plt.title('mesh')
        # plt.show()
        # plt.plot(np.sort(X.tabulate_dof_coordinates().reshape(-1, 2)[:, 0]),
        #          jbar2.vector().get_local()[fem.vertex_to_dof_map(X)])
        # plt.title('pseudo')
        # plt.show()
        jbar2.set_allow_extrapolation(True)
        jbar2 = fem.interpolate(jbar2, V)
        # print(jbar2.vector().get_local())
        jbar = fem.Expression(sym.printing.ccode(sym_j), j=jbar2, degree=1)
        # fem.plot(fem.interpolate(jbar, V))
        # plt.show()

        # fem.plot(jbar)
        # plt.show()
        cs_1 = fem.Function(V)
        cs_1.vector()[:] = cs_data[i_1].astype('double')
        # cs_d = fem.Function(V)
        # cs_d.vector()[:] = cs_data[i, fem.dof_to_vertex_map(V)].astype('double')
        # Initialize Dirichlet BCs
        # cs_bc = fem.Function(V)
        # cs_bc.vector()[:] = cs_data[i, fem.dof_to_vertex_map(V)]
        # cs_bcr = fem.Function(X)
        # cs_bcr.vector()[:] = fem.interpolate(cse_eq, X)
        cse_eq = fem.Expression('x[1] >= 1 - DOLFIN_EPS ? cs : 0', cs=cs_1, degree=1)
        # cse_eq2 = fem.Expression('x[1] >= 1 - DOLFIN_EPS ? cs : 0', cs=cs_d, degree=1)
        bc = [fem.DirichletBC(V, cse_eq, bm, 5)]
        # fem.plot(cs_1)
        # plt.title('COMSOL')
        # plt.show()

        # Setup Neumann BCs
        neumann = dtc*rbar2*(-jbar)*v*ds(5)

        # Setup equation
        a = Rs*rbar2*cs*v*dx

        L = Rs*rbar2*cs_1*v*dx - dtc*Ds*rbar2/Rs*fem.dot(cs_1.dx(1), v.dx(1))*dx + neumann

        # Solve
        cs = fem.Function(V)
        fem.solve(a == L, cs)
        cs.set_allow_extrapolation(True)
        # fem.solve(a == L, cs, bc)
        # u_array2[k, :] = fem.interpolate(cs, X).vector().get_local()[fem.vertex_to_dof_map(X)]
        # fem.plot(cs)
        # plt.title('FEniCS')
        # plt.show()
        cse = fem.interpolate(cs, X)

        # xx = np.linspace(0, 2.5, num=500)
        # uu = np.zeros_like(xx)
        #
        # for i in np.arange(0, xx.size):
        #     uu[i] = cs(xx[i], 1)
        #
        # plt.plot(uu)
        # plt.show()
        # plt.plot(submesh.coordinates()[fem.vertex_to_dof_map(X)], cse.vector().get_local())
        # plt.plot(cse.vector().get_local())
        # plt.show()

        # gdim = submesh.geometry().dim()
        # bmsh_dof_coords = X.tabulate_dof_coordinates().reshape(-1, gdim)
        # gmsh_dof_coords = V.tabulate_dof_coordinates().reshape(-1, gdim)
        # mesh_dof_coords = W.tabulate_dof_coordinates().reshape(-1, gdim-1)
        #
        # bnd_to_glob_map = {}
        # z = 0
        # for i, mesh_coords in enumerate(mesh_dof_coords):
        #     if mesh_coords <= 1 or mesh_coords >= 2:
        #         bnd_to_glob_map[z] = i
        #         z += 1
        # # bnd_to_glob_map = {i: j for i in range(len(bmsh_dof_coords)) if mesh_dof_coords[j] <= 1 or mesh_dof_coords[j] >= 2}
        #
        # V_to_cse = fem.Function(W)
        # for bnd_dof, mesh_dof in bnd_to_glob_map.items():
        #     V_to_cse.vector()[mesh_dof] = cse.vector().get_local()[bnd_dof]
        cset = fem.interpolate(cse_eq, X)
        # fem.plot(cset)
        # plt.plot(cset.vector().get_local())

        # plt.show()
        u_array2[k, :] = cse.vector().get_local()
            # u_array2[k, mesh_dof] = cse.vector().get_local()[fem.vertex_to_dof_map(X)][bnd_dof]

        u_nodal_values = cs.vector()
        u_array[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
        # ofile << (cs, float(time[i]))
        k += 1
    # plt.savefig('test.png', dpi=300)
    # plt.grid()
    # plt.show()

    print(utilities.norm_rmse(u_array, cs_data[1::2]))
    # print(engine.rmse(u_array, cs_data[1::2])/np.sqrt(np.average(cs_data[1::2]**2))*100)
    # print(np.average(np.subtract(u_array, cs_data[1::2]), axis=1))

    data = np.append(cs_data1[:, 0:2], cs_data1[:, 3::2], axis=1)
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]
    data = data[indices]
    data = data[data[:, 0].argsort()]
    xcoor = data[:, 0]
    cse = data[:, 2:]

    print(engine.rmse(u_array2, cse.T) / np.sqrt(np.average(cse.T ** 2)) * 100)

    # plt.plot(np.repeat([np.sort(X.tabulate_dof_coordinates().reshape(-1, 2)[:, 0])], len(time_in), axis=0).T,
    #          u_array2.T)
    # plt.show()
    # plt.plot(np.repeat([xcoor], len(time_in), axis=0).T, cse)
    # plt.show()

    # utilities.overlay_plt(xcoor, time_in, '$c_s$', u_array2, cse.T)

    utilities.report(xcoor, time_in, u_array2, cse.T, '$c_s$')

    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    # utilities.overlay_plt(comsol_sol.mesh, time_in, '$c_s$', u_array2, comsol_sol.data.cse[1::2])
    # utilities.overlay_plt(comsol_sol.mesh, time_in, '$c_s$', u_array2, comsol_sol.data.cse[1::2])
    # plt.plot(np.sort(X.tabulate_dof_coordinates().reshape(-1, 2)[:, 0]), u_array2.T, marker='o')

    plt.savefig('comsol_compare_cs.png')
    plt.show()
    exit()
    plt.plot(np.repeat([cmn1d.comsol_solution.mesh], len(time_in), axis=0).T, cmn1d.comsol_solution.data.cse[1::2].T)
    plt.show()

if __name__ == '__main__':
    c_e()


#
#
#
# Rs = cmn.Rs
#
# fem.plot(cmn.bm)
# plt.show()
#
# exit(0)
# mesh = fin.UnitSquareMesh(10, 10)
# # V = fin.VectorElement("Lagrange", mesh.ufl_cell(), 1)
# V = fin.FunctionSpace(mesh, 'Lagrange', 1)
#
# bmesh = fin.BoundaryMesh(mesh, 'exterior')
# cc = fin.MeshFunction('size_t', bmesh, bmesh.topology().dim())
# top = fin.AutoSubDomain(lambda x: x[1] > 1-1e-8)
# top.mark(cc, 1)
# submesh = fin.SubMesh(bmesh, cc, 1)
# W = fin.FunctionSpace(submesh, 'Lagrange', 1)
#
#
# # ME = fin.FunctionSpace(mesh, V*V*V)
# # W = fin.FunctionSpace(mesh, V)
# e = fin.Expression('x[0]', degree=1)
# g = fin.Expression('near(x[1], 1, DOLFIN_EPS)', degree=1)
# f = fin.interpolate(e, V)
#
#
# # fin.plot(f)
# # plt.show()
#
# plt.plot(submesh.coordinates()[:][:, 0], fin.interpolate(f, W).vector().get_local()[fin.vertex_to_dof_map(W)])
# plt.show()
#
# exit(0)
#
# mesh = fin.UnitIntervalMesh(20)
#
# V = fin.FiniteElement('Lagrange', mesh.ufl_cell(), 1)
# # V = fin.FunctionSpace(mesh, 'CG', 1)
# M = fin.FunctionSpace(mesh, V*V)
# V1 = fin.FunctionSpace(mesh, V)
#
# # Some function in M to illustrate that components
# # will change by assign
# m = fin.interpolate(fin.Expression(('x[0]', '1-x[0]'), degree=1), M)
# m0, m1 = fin.split(m)
#
# # Plot for comparison
# fin.plot(m0, title='m0 before')
# plt.show()
# fin.plot(m1, title='m1 before')
# plt.show()
#
# # Functions for components
# v0 = fin.interpolate(fin.Expression('cos(pi*x[0])', degree=1), V1)
# v1 = fin.interpolate(fin.Expression('sin(pi*x[0])', degree=1), V1)
#
# # Assign the components
# fin.assign(m.sub(0), v0)
# fin.assign(m.sub(1), v1)
#
# # See if it worked
# fin.plot(m0, title='m0 after')
# plt.show()
# fin.plot(m1, title='m1 after')
# plt.show()

