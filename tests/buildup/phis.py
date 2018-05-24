import domain2
import fenics as fem
import matplotlib.pyplot as plt
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import numpy as np


def gather_data():
    # Load required cell data
    resources = '../reference/'
    params = engine.fetch_params(resources + 'GuAndWang_parameter_list.xlsx')
    d_comsol = comsol.load(resources + 'guwang.npz')
    return d_comsol, params


def mkparam(markers, k_1 = 0, k_2 = 0, k_3 = 0, k_4 = 0):
    cppcode = """
    class K : public Expression
    {
        public:
            void eval(Array<double>& values,
                      const Array<double>& x,
                      const ufc::cell& cell) const
            {
                switch((*markers)[cell.index]){
                case 1:
                    values[0] = k_1;
                    break;
                case 2:
                    values[0] = k_2;
                    break;
                case 3:
                    values[0] = k_3;
                    break;
                case 4:
                    values[0] = k_4;
                    break;
                default:
                    values[0] = 0;
                }
            }

        std::shared_ptr<MeshFunction<std::size_t>> markers;
        double k_1, k_2, k_3, k_4;
    };
    """

    var = fem.Expression(cppcode=cppcode, degree=0)
    var.markers = markers
    var.k_1, var.k_2, var.k_3, var.k_4 = k_1, k_2, k_3, k_4

    return var


def phis():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect required data
    # TODO: make sure refactored comsol works here
    comsol_data, params = gather_data()
    time_ind = engine.find_ind(comsol_data.time_mesh, time)
    data = comsol.get_standardized(comsol_data.filter_time(time_ind))

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(data.mesh)))
    mesh, dx, ds, bm, dm = domain2.generate_domain(data.mesh)

    # Initialize parameters
    F = fem.Constant(96487)
    Iapp = fem.Constant(0)
    Acell = fem.Comstant(params.const.Acell)

    Lc = mkparam(dm, params.neg.L, params.sep.L, params.pos.L)
    sigma_eff = mkparam(dm, params.neg.sigma_ref * params.neg.eps_s ** params.neg.brug_sigma, 0,
                        params.pos.sigma_ref * params.pos.eps_s ** params.pos.brug_sigma)
    a_s = mkparam(dm, 3 * params.neg.eps_s / params.neg.Rs, 0, 3 * params.pos.eps_s / params.pos.Rs)

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
        neumann = 0*v*ds(1) + 0*v*ds(2) + 0*v*ds(3) + Iapp/Acell*v*ds(4)

        # Setup equation
        a1 = -sigma_eff/Lc * fem.dot(fem.grad(phi), fem.grad(v))
        a = a1*dx(1) + 0*v*dx(2) + a1*dx(3)

        L1 = Lc*a_s*F*jbar*v
        L = L1*dx(1) + 0*v*dx(2) + L2*dx(3) + neumann

        # Solve
        phi = fem.Function(V)
        fem.solve(a == L, phi, bc)
        fem.plot(phi)

        u_nodal_values = phi.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
    plt.savefig('test.png', dpi=300)
    plt.grid()
    plt.show()

    print(engine.rmse(u_array[:, data.neg_ind], data.data.phis[:, data.neg_ind]))
    print(engine.rmse(u_array[:, data.pos_ind], data.data.phis[:, data.pos_ind]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    plt.figure(1, figsize=(15, 9))

    plt.subplot(221)
    plt.plot(np.repeat([data.neg], len(time), axis=0).T, u_array[:, data.neg_ind].T)
    plt.grid(), plt.title('FEniCS Negative Electrode')
    plt.subplot(222)
    plt.plot(np.repeat([data.neg], len(time), axis=0).T, data.data.phis[:, data.neg_ind].T)
    plt.grid(), plt.title('COMSOL Negative Electrode')
    plt.subplot(223)
    plt.plot(np.repeat([data.pos], len(time), axis=0).T, u_array[:, data.pos_ind].T)
    plt.grid(), plt.title('FEniCS Positive Electrode')
    plt.subplot(224)
    plt.plot(np.repeat([data.pos], len(time), axis=0).T, data.data.phis[:, data.pos_ind].T)
    plt.grid(), plt.title('COMSOL Positive Electrode')

    plt.show()

if __name__ == '__main__':
    phis()
