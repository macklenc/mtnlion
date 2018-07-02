import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import mtnlion.engine as engine
import utilities


def phis():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm
    Iapp, Acell, sigma_eff, Lc, a_s, F = cmn.Iapp, cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, cmn.F

    for i, j in enumerate(cmn.comsol_solution.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        phi = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, 0.0, bm, 1), fem.DirichletBC(V, comsol_sol.data.phis[i, -1], bm, 4)]
        jbar = fem.Function(V)
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')

        # Setup Neumann BCs
        neumann = 0*v*ds(1) + 0*v*ds(2) + 0*v*ds(3) + Iapp[i]/Acell*v*ds(4)

        # Setup equation
        a1 = -sigma_eff/Lc * fem.dot(fem.grad(phi), fem.grad(v))
        a = a1*dx(1) + 0*v*dx(2) + a1*dx(3)

        L1 = Lc*a_s*F*jbar*v
        L = L1*dx(1) + 0*v*dx(2) + L1*dx(3) + neumann

        # Solve
        phi = fem.Function(V)
        fem.solve(a == L, phi, bc)
        fem.plot(phi)

        u_nodal_values = phi.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]

    print(engine.rmse(u_array[:, comsol_sol.neg_ind], comsol_sol.data.phis[:, comsol_sol.neg_ind]))
    print(engine.rmse(u_array[:, comsol_sol.pos_ind], comsol_sol.data.phis[:, comsol_sol.pos_ind]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    utilities.overlay_plt(comsol_sol.neg, time, '$\Phi_s^{neg}$',
                          u_array[:, comsol_sol.neg_ind], comsol_sol.data.phis[:, comsol_sol.neg_ind])
    plt.savefig('comsol_compare_phis_neg.png')
    plt.show()

    utilities.overlay_plt(comsol_sol.pos, time, '$\Phi_s^{pos}$',
                          u_array[:, comsol_sol.pos_ind], comsol_sol.data.phis[:, comsol_sol.pos_ind])
    plt.savefig('comsol_compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    phis()
