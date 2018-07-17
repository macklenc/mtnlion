import fenics as fem
import common
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import mtnlion.engine as engine
import mtnlion.loader as loader
import mtnlion.comsol as comsol
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
    mesh = fem.Mesh('../reference/comsol_solution/cs.xml')
    file_data = loader.collect_files(['../reference/comsol_solution/cs.csv.bz2'], format_key=comsol.format_name, loader=loader.load_csv_file)
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)

    transform = []
    for i in file_data['cs']:
        ind1 = np.where(np.abs(mesh.coordinates()[:][:, 0] - i[0]) <= 1e-5)
        ind2 = np.where(np.abs(mesh.coordinates()[:][:, 1] - i[1]) <= 1e-5)

        if len(ind1[0]) > 0 and len(ind2[0]) > 0:
            transform.append(np.intersect1d(ind1, ind2)[0])
            if len(np.intersect1d(ind1, ind2)) > 1:
                raise ValueError('Too many matching indices')
        else:
            raise ValueError('Missing indices, check tolerances')

    cs_data = file_data['cs'][transform, 2:]

    # print(fem.vertex_to_dof_map(V))
    # print(mesh.coordinates()[:])
    cmn1d = common.Common(time)
    cmn = common.Common2(time, mesh)

    cs_data = np.array(cs_data[:, cmn.time_ind]).T
    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(mesh.coordinates()[:])))
    u_array2 = np.empty((len(time_in), 42))
    u_array2[:] = np.nan

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm

    Ds = cmn.Ds
    Rs = cmn.Rs

    j, y, eps = sym.symbols('j x[1] DOLFIN_EPS')
    sym_j = sym.Piecewise((j, (y-eps) >= 1), (0, True))
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    # cse = fem.Expression('')

    bmesh = fem.BoundaryMesh(mesh, 'exterior')
    cc = fem.MeshFunction('size_t', bmesh, bmesh.topology().dim())
    top = fem.AutoSubDomain(lambda x: x[1] > 1 - fem.DOLFIN_EPS)
    top.mark(cc, 1)
    submesh = fem.SubMesh(bmesh, cc, 1)


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
        cse_eq = fem.Expression('x[1] - 1 <= DOLFIN_EPS ? cs : 0', cs=cs, degree=1)

        # Initialize Dirichlet BCs
        # cs_bc = fem.Function(V)
        # cs_bc.vector()[:] = cs_data[i, fem.dof_to_vertex_map(V)]
        # cs_bcr = fem.Function(X)
        # cs_bcr.vector()[:] = fem.interpolate(cse_eq, X)

        bc = [fem.DirichletBC(V, cse_eq, bm, 5)]
        jbar1 = fem.Function(W)
        jbar1.vector()[:] = comsol_sol.data.j[i_1, fem.dof_to_vertex_map(W)].astype('double')
        jbar2 = fem.interpolate(jbar1, V)
        jbar = fem.Expression(sym.printing.ccode(sym_j), j=jbar2, degree=1)

        # fem.plot(jbar)
        # plt.show()
        cs_1 = fem.Function(V)
        cs_1.vector()[:] = cs_data[i_1, fem.dof_to_vertex_map(V)].astype('double')
        fem.plot(cs_1)
        plt.title('1')
        plt.show()

        # Setup Neumann BCs
        neumann = dtc*rbar2*(-jbar)*v*ds(5)

        # Setup equation
        a = Rs*rbar2*cs*v*dx

        L = Rs*rbar2*cs_1*v*dx - dtc*Ds*rbar2/Rs*fem.dot(cs_1.dx(1), v.dx(1))*dx + neumann

        # Solve
        cs = fem.Function(V)
        fem.solve(a == L, cs)
        # fem.solve(a == L, cs, bc)
        # u_array2[k, :] = fem.interpolate(cs, X).vector().get_local()[fem.vertex_to_dof_map(X)]
        fem.plot(cs)
        plt.title('2')
        plt.show()
        cse = fem.interpolate(cs, X)
        # plt.plot(submesh.coordinates()[fem.vertex_to_dof_map(X)], cse.vector().get_local())
        # plt.show()
        # exit(0)

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
        cset = fem.interpolate(cse_eq, V)
        fem.plot(cset)
        plt.show()
        u_array2[k, :] = cse.vector().get_local()[fem.vertex_to_dof_map(X)]
            # u_array2[k, mesh_dof] = cse.vector().get_local()[fem.vertex_to_dof_map(X)][bnd_dof]

        u_nodal_values = cs.vector()
        u_array[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
        k += 1
    # plt.savefig('test.png', dpi=300)
    # plt.grid()
    # plt.show()
    print(engine.rmse(u_array, cs_data[1::2]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    # utilities.overlay_plt(comsol_sol.mesh, time_in, '$c_s$', u_array2, comsol_sol.data.cse[1::2])
    plt.plot(submesh.coordinates()[:, 0][fem.vertex_to_dof_map(X)], u_array2.T, marker='o')

    plt.savefig('comsol_compare_cs.png')
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

