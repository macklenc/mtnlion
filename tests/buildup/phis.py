import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import comsol
import domain
import engine


def gather_data():
    # Load required cell data
    resources = '../reference/'
    params = engine.fetch_params(resources + 'GuAndWang_parameter_list.xlsx')
    # c_data = comsol.ComsolData(resources + 'guwang.npz')
    data_file = comsol.IOHandler(resources + 'guwang.npz')
    d_comsol = comsol.Formatter.set_data(data_file.data)
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
    sim_data = comsol.Formatter.get_fenics_friendly(comsol_data)
    data = sim_data.get_solution_near_time(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(data.mesh.mesh)))
    mesh, dx, ds, bm, dm = domain.generate_domain(data.mesh.mesh)

    # Initialize parameters
    F = 96487
    Iapp = 0

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
        bc = [fem.DirichletBC(V, 0.0, bm, 1), fem.DirichletBC(V, 3.8, bm, 4)]
        f = fem.Function(V)
        f.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double') * fem.Constant(F)

        a = fem.Constant(1) / (Lc * Lc) * sigma_eff * fem.dot(fem.grad(phi), fem.grad(v)) * dx(1) + fem.Constant(1) / (
                Lc * Lc) * sigma_eff * fem.dot(fem.grad(phi), fem.grad(v)) * dx(3)
        L = -a_s * f * v * dx(1) - a_s * f * v * dx(3) - fem.Constant(Iapp / params.const.Acell) * v * ds(4)

        phi = fem.Function(V)
        fem.solve(a == L, phi, bc)
        fem.plot(phi)

        u_nodal_values = phi.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
    plt.savefig('test.png', dpi=300)
    plt.grid()
    plt.show()

    print(engine.rmse(u_array[:, data.mesh.neg_ind], data.data.phis[:, data.mesh.neg_ind]))
    print(engine.rmse(u_array[:, data.mesh.pos_ind], data.data.phis[:, data.mesh.pos_ind]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    print(type(u_array))

    pass


if __name__ == '__main__':
    phis()
