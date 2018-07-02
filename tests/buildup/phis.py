import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


def phis():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect required data
    # TODO: make sure refactored comsol works here
    comsol_data, params = utilities.gather_data()
    time_ind = engine.find_ind(comsol_data.time_mesh, time)
    data = comsol.get_standardized(comsol_data.filter_time(time_ind))

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(data.mesh)))
    mesh, dx, ds, bm, dm = domain2.generate_domain(data.mesh)

    # Initialize parameters
    F = fem.Constant(96487)
    I_1C = fem.Constant(20.5)
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else fem.Constant(0) for i in time]
    Acell = fem.Constant(params.const.Acell)

    Lc = utilities.mkparam(dm, params.neg.L, params.sep.L, params.pos.L)
    sigma_eff = utilities.mkparam(dm, params.neg.sigma_ref * params.neg.eps_s ** params.neg.brug_sigma, 0,
                        params.pos.sigma_ref * params.pos.eps_s ** params.pos.brug_sigma)
    a_s = utilities.mkparam(dm, 3 * params.neg.eps_s / params.neg.Rs, 0, 3 * params.pos.eps_s / 8e-6)

    for i, j in enumerate(data.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        phi = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, 0.0, bm, 1), fem.DirichletBC(V, data.data.phis[i, -1], bm, 4)]
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

    print(engine.rmse(u_array[:, data.neg_ind], data.data.phis[:, data.neg_ind]))
    print(engine.rmse(u_array[:, data.pos_ind], data.data.phis[:, data.pos_ind]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    utilities.overlay_plt(data.neg, time, '$\Phi_s^{neg}$', u_array[:, data.neg_ind], data.data.phis[:, data.neg_ind])
    plt.savefig('comsol_compare_phis_neg.png')
    plt.show()

    utilities.overlay_plt(data.pos, time, '$\Phi_s^{pos}$', u_array[:, data.pos_ind], data.data.phis[:, data.pos_ind])
    plt.savefig('comsol_compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    phis()
