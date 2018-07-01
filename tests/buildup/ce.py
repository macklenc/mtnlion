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
    time_in = [0.1, 5, 10, 15, 20]
    dt = 0.1
    dtc = fem.Constant(dt)
    time = [None]*(len(time_in)*2)
    time[::2] = [t-dt for t in time_in]
    time[1::2] = time_in

    # Collect required data
    # TODO: make sure refactored comsol works here
    comsol_data, params = gather_data()
    time_ind = engine.find_ind_near(comsol_data.time_mesh, time)
    data = comsol.get_standardized(comsol_data.filter_time(time_ind))

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(data.mesh)))
    mesh, dx, ds, bm, dm = domain2.generate_domain(data.mesh)

    # Initialize parameters
    eps_e = mkparam(dm, params.neg.eps_e, params.sep.eps_e, params.pos.eps_e)
    de_eff = mkparam(dm, params.const.De_ref*params.neg.eps_e**params.neg.brug_De,
                     params.const.De_ref * params.sep.eps_e ** params.sep.brug_De,
                     params.const.De_ref * params.pos.eps_e ** params.pos.brug_De)
    t_plus = fem.Constant(params.const.t_plus)
    Lc = mkparam(dm, params.neg.L, params.sep.L, params.pos.L)
    a_s = mkparam(dm, 3 * params.neg.eps_s / params.neg.Rs, 0, 3 * params.pos.eps_s / params.pos.Rs)

    # plt.plot(data.mesh, data.data.ce[1])
    # plt.show()
    k = 0
    for i, j in enumerate(data.data.j[1::2]):
        i_1 = i * 2
        i = i*2 + 1

        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        ce = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, data.data.ce[i, 0], bm, 1), fem.DirichletBC(V, data.data.ce[i, -1], bm, 4)]
        jbar = fem.Function(V)
        data.data.j[i_1, 21] = data.data.j[i_1, 20]
        data.data.j[i_1, 49] = data.data.j[i_1, 50]
        jbar.vector()[:] = data.data.j[i_1, fem.dof_to_vertex_map(V)].astype('double')
        fem.plot(jbar)
        plt.show()
        ce_1 = fem.Function(V)
        ce_1.vector()[:] = data.data.ce[i_1, fem.dof_to_vertex_map(V)].astype('double')

        boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
        b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
        b1.mark(boundary_markers, 2)
        b2.mark(boundary_markers, 3)

        n = fem.FacetNormal(mesh)

        # Setup measures
        dS = fem.Measure('dS', domain=mesh, subdomain_data=boundary_markers)

        # Setup Neumann BCs
        neumann = dtc*de_eff('-')/Lc('-')*fem.inner(fem.grad(ce_1('-')), n('-'))*v('-')*dS(2) + \
                  dtc*de_eff('+')/Lc('+')*fem.inner(fem.grad(ce_1('+')), n('+'))*v('+')*dS(2) + \
                  dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce_1('-')), n('-')) * v('-') * dS(3) + \
                  dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce_1('+')), n('+')) * v('+') * dS(3)

        # Setup equation
        a = Lc*eps_e*ce*v*dx

        L = Lc*eps_e*ce_1*v*dx - dtc*de_eff/Lc*fem.dot(fem.grad(ce_1), fem.grad(v))*dx + dtc*Lc*a_s*\
            (fem.Constant(1) - t_plus)*jbar*v*dx + neumann

        # Solve
        ce = fem.Function(V)
        fem.solve(a == L, ce, bc)
        # fem.plot(ce)

        u_nodal_values = ce.vector()
        u_array[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
        k += 1
    # plt.savefig('test.png', dpi=300)
    # plt.grid()
    # plt.show()

    print(engine.rmse(u_array, data.data.ce[1::2]))

    coor = mesh.coordinates()
    # for i in range(len(u_array)):
    #     print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    plt.figure(1, figsize=(15, 9))

    plt.subplot(211)
    plt.plot(np.repeat([data.mesh], len(time_in), axis=0).T, u_array.T)
    plt.grid(), plt.title('FEniCS'), plt.legend(['t = {}'.format(t) for t in time_in])
    plt.subplot(212)
    plt.plot(np.repeat([data.mesh], len(time_in), axis=0).T, data.data.ce[1::2].T)
    plt.grid(), plt.title('COMSOL'), plt.legend(['t = {}'.format(t) for t in time_in])

    plt.savefig('comsol_compare_ce.png')
    plt.show()

if __name__ == '__main__':
    phis()
