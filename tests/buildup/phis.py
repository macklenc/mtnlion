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
    c_data = comsol.ComsolData(resources + 'guwang.npz')

    return c_data, params


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
    time = [0, 5, 10, 15, 20]
    # Collect required data
    comsol_data, params = gather_data()
    data = comsol_data.data.get_solution_near_time(time)
    u_array = np.empty((len(time), len(data.mesh.mesh) - 2))
    mesh, dx, ds, bm, dm = domain.generate_domain(comsol_data)

    # Initialize parameters
    F = 96487
    Iapp = 0

    # remove repeated boundary
    fun = comsol_data.remove_dup_boundary(data.j)

    Lc = mkparam(dm, params.neg.L, params.sep.L, params.pos.L)
    sigma_eff = mkparam(dm, params.neg.sigma_ref * params.neg.eps_s ** params.neg.brug_sigma, 0,
                        params.pos.sigma_ref * params.pos.eps_s ** params.pos.brug_sigma)
    a_s = mkparam(dm, 3 * params.neg.eps_s / params.neg.Rs, 0, 3 * params.pos.eps_s / params.pos.Rs)

    for i, j in enumerate(fun):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        phi = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, 0.0, bm, 1), fem.DirichletBC(V, 4.2, bm, 4)]
        f = fem.Function(V)
        f.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double') * fem.Constant(F)

        a = fem.Constant(1) / (Lc * Lc) * sigma_eff * fem.dot(fem.grad(phi), fem.grad(v)) * dx(1) + fem.Constant(1) / (
                Lc * Lc) * sigma_eff * fem.dot(fem.grad(phi), fem.grad(v)) * dx(3)
        L = -a_s * f * v * dx(1) - a_s * f * v * dx(3) - fem.Constant(Iapp / params.const.Acell) * v * ds(4)

        phi = fem.Function(V)
        fem.solve(a == L, phi, bc)
        fem.plot(phi)

        u_nodal_values = phi.vector()
        u_array[i, :] = u_nodal_values.get_local()
    plt.savefig('test.png', dpi=300)
    plt.grid()
    plt.show()


    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    print(type(u_array))

    pass


if __name__ == '__main__':
    phis()
