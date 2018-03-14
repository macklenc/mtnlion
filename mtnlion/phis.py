import fenics as fem
import numpy as np
import matplotlib.pyplot as plt
import ldp
import munch
import sympy as sym
import engine

def gather_data():
    # Load required cell data
    comsol = ldp.load('../tests/reference/guwang.npz')
    sheet = ldp.read_excel(
        '../tests/reference/GuAndWang_parameter_list.xlsx', 0)

    (ncol, pcol) = (2, 3)
    params = dict()
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    structured_params = munch.DefaultMunch.fromDict(params)

    return comsol, structured_params


def generate_domain(comsol):
    boundaries = range(4)

    # Create 1D mesh
    mesh = fem.IntervalMesh(len(comsol['mesh'])-1, 0, 3)
    mesh.coordinates()[:] = np.array([comsol['mesh']]).transpose()

    # Setup subdomain markers
    neg_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[0], b2=boundaries[1])
    sep_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[1], b2=boundaries[2])
    pos_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[2], b2=boundaries[3])

    # Setup boundary markers
    b0 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)', b=boundaries[0])
    b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=boundaries[1])
    b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=boundaries[2])
    b3 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)', b=boundaries[3])

    # Mark the subdomains
    domain_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim())
    domain_markers.set_all(0)
    neg_domain.mark(domain_markers, 1)
    sep_domain.mark(domain_markers, 2)
    pos_domain.mark(domain_markers, 3)

    # Mark the boundaries
    boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    b0.mark(boundary_markers, 1)
    b1.mark(boundary_markers, 2)
    b2.mark(boundary_markers, 3)
    b3.mark(boundary_markers, 4)

    # Setup measures
    dx = fem.Measure('dx', domain=mesh, subdomain_data=domain_markers)
    ds = fem.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # print(domain_markers.array())
    # print(boundary_markers.array())
    # fem.plot(markers)
    # plt.show()

    return mesh, dx, ds, boundary_markers, domain_markers


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
    # Collect required data
    comsol_data, params = gather_data()
    mesh, dx, ds, bm, dm = generate_domain(comsol_data)

    # Define function space and basis functions
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    phi = fem.TrialFunction(V)
    v = fem.TestFunction(V)

    # Initialize Dirichlet BCs
    bc = [fem.DirichletBC(V, 0.0, bm, 1), fem.DirichletBC(V, 4.2, bm, 4)]

    # Initialize parameters
    F = 96487
    Iapp = 0

    f = fem.Function(V)
    # f.vector()[:] = comsol_data['j'][fem.dof_to_vertex_map(V),1].astype('double')*fem.Constant(F)
    data = engine.fetch_comsol_solutions('../tests/reference/guwang.npz', [5])
    f.vector()[:] = data.j[0, fem.dof_to_vertex_map(V)].astype('double')*fem.Constant(F)

    Lc = mkparam(dm, params.neg.L, params.sep.L, params.pos.L)

    sigma_eff = mkparam(dm, params.neg.sigma_ref*params.neg.eps_s**params.neg.brug_sigma, 0,
                        params.pos.sigma_ref*params.pos.eps_s**params.pos.brug_sigma)
    a_s = mkparam(dm, 3*params.neg.eps_s/params.neg.Rs, 0, 3*params.pos.eps_s/params.pos.Rs)

    a = fem.Constant(1)/(Lc*Lc)*sigma_eff*fem.dot(fem.grad(phi), fem.grad(v))*dx(1) + fem.Constant(1)/(Lc*Lc)*sigma_eff*fem.dot(fem.grad(phi), fem.grad(v))*dx(3)
    L = -a_s*f*v*dx(1) - a_s*f*v*dx(3) - fem.Constant(Iapp/params.const.Acell)*v*ds(4)

    phi = fem.Function(V)
    fem.solve(a == L, phi, bc)
    fem.plot(phi)
    plt.savefig('test.png', dpi=300)
    plt.grid()
    plt.show()

    u_nodal_values = phi.vector()
    u_array = u_nodal_values.get_local()
    coor = mesh.coordinates()
    for i in range(len(u_array)):
        print('u(%8g) = %g' % (coor[i], u_array[len(u_array)-1-i]))

    print(type(u_array))

    pass


if __name__ == '__main__':
    phis()
