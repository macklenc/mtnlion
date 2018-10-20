import fenics as fem
import numpy as np


class Domain():
    def __init__(self, mesh, V, dx, ds, dS, n, boundary_markers, domain_markers):
        self.mesh = mesh
        self.V = V
        self.dx = dx
        self.ds = ds
        self.dS = dS
        self.n = n
        self.neg_marker, self.sep_marker, self.pos_marker = (0, 1, 2)
        self.boundary_markers = boundary_markers
        self.domain_markers = domain_markers


# TODO: refactor
def generate_domain(raw_mesh, pseudo_mesh):
    boundaries = np.arange(4)
    pseudo_boundaries = np.array([0, 1, 1.5, 2.5])

    # Create 1D mesh
    mesh = fem.IntervalMesh(len(raw_mesh) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([raw_mesh]).transpose()

    main_V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    pseudo_V = fem.FunctionSpace(pseudo_mesh, 'Lagrange', 1)

    # Setup subdomain markers
    neg_domain = fem.CompiledSubDomain('(x[0] >= (b1 - DOLFIN_EPS)) && (x[0] <= (b2 + DOLFIN_EPS))',
                                       b1=boundaries[0].astype(np.double), b2=boundaries[1].astype(np.double))
    sep_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[1].astype(np.double), b2=boundaries[2].astype(np.double))
    pos_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[2].astype(np.double), b2=boundaries[3].astype(np.double))

    # Setup boundary markers
    b0 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)', b=boundaries[0].astype(np.double))
    b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=boundaries[1].astype(np.double))
    b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=boundaries[2].astype(np.double))
    b3 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)', b=boundaries[3].astype(np.double))

    # Setup subdomain markers
    pseudo_neg_domain = fem.CompiledSubDomain('(x[0] >= (b1 - DOLFIN_EPS)) && (x[0] <= (b2 + DOLFIN_EPS))',
                                              b1=pseudo_boundaries[0].astype(np.double),
                                              b2=pseudo_boundaries[1].astype(np.double))
    pseudo_sep_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                              b1=pseudo_boundaries[1].astype(np.double),
                                              b2=pseudo_boundaries[2].astype(np.double))
    pseudo_pos_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                              b1=pseudo_boundaries[2].astype(np.double),
                                              b2=pseudo_boundaries[3].astype(np.double))

    # Setup boundary markers
    pseudo_b0 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)',
                                      b=pseudo_boundaries[0].astype(np.double))
    pseudo_b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=pseudo_boundaries[1].astype(np.double))
    pseudo_b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=pseudo_boundaries[2].astype(np.double))
    pseudo_b3 = fem.CompiledSubDomain('on_boundary && near(x[0], b, DOLFIN_EPS)',
                                      b=pseudo_boundaries[3].astype(np.double))
    cse = fem.CompiledSubDomain('on_boundary && near(x[1], b, DOLFIN_EPS)', b=1)  # pseudo dim only

    # Mark the subdomains, main dim
    main_domain_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim())
    main_domain_markers.set_all(99)
    sep_domain.mark(main_domain_markers, 1)
    neg_domain.mark(main_domain_markers, 0)
    pos_domain.mark(main_domain_markers, 2)

    # Mark the boundaries, main dim
    main_boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    main_boundary_markers.set_all(0)
    b0.mark(main_boundary_markers, 1)
    b1.mark(main_boundary_markers, 2)
    b2.mark(main_boundary_markers, 3)
    b3.mark(main_boundary_markers, 4)

    # Setup measures, main dim
    main_dx = fem.Measure('dx', domain=mesh, subdomain_data=main_domain_markers)
    main_ds = fem.Measure('ds', domain=mesh, subdomain_data=main_boundary_markers)
    main_dS = fem.Measure('dS', domain=mesh, subdomain_data=main_boundary_markers)

    # normal vector, main dim
    main_n = fem.FacetNormal(mesh)

    # Mark the subdomains, pseudo dim
    pseudo_domain_markers = fem.MeshFunction('size_t', pseudo_mesh, pseudo_mesh.topology().dim())
    pseudo_domain_markers.set_all(0)
    pseudo_sep_domain.mark(pseudo_domain_markers, 2)
    pseudo_neg_domain.mark(pseudo_domain_markers, 1)
    pseudo_pos_domain.mark(pseudo_domain_markers, 3)

    # Mark the boundaries, pseudo dim
    pseudo_boundary_markers = fem.MeshFunction('size_t', pseudo_mesh, pseudo_mesh.topology().dim() - 1)
    pseudo_boundary_markers.set_all(0)
    pseudo_b0.mark(pseudo_boundary_markers, 1)
    pseudo_b1.mark(pseudo_boundary_markers, 2)
    pseudo_b2.mark(pseudo_boundary_markers, 3)
    pseudo_b3.mark(pseudo_boundary_markers, 4)
    cse.mark(pseudo_boundary_markers, 5)

    # Setup measures, pseudo dim
    pseudo_dx = fem.Measure('dx', domain=pseudo_mesh, subdomain_data=pseudo_domain_markers)
    pseudo_ds = fem.Measure('ds', domain=pseudo_mesh, subdomain_data=pseudo_boundary_markers)
    pseudo_dS = fem.Measure('dS', domain=pseudo_mesh, subdomain_data=pseudo_boundary_markers)

    # normal vector, pseudo dim
    pseudo_n = fem.FacetNormal(pseudo_mesh)

    # print(main_domain_markers.array())
    # print(main_boundary_markers.array())
    # fem.plot(markers)
    # plt.show()

    return Domain(mesh, main_V, main_dx, main_ds, main_dS, main_n, main_boundary_markers, main_domain_markers), \
           Domain(pseudo_mesh, pseudo_V, pseudo_dx, pseudo_ds, pseudo_dS, pseudo_n, pseudo_boundary_markers,
                  pseudo_domain_markers)


def generate_domain2(mesh):
    boundaries = [0, 1, 1.5, 2.5]

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

    cse = fem.CompiledSubDomain('on_boundary && near(x[1], b, DOLFIN_EPS)', b=1)

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
    cse.mark(boundary_markers, 5)

    # Setup measures
    dx = fem.Measure('dx', domain=mesh, subdomain_data=domain_markers)
    ds = fem.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # print(domain_markers.array())
    # print(boundary_markers.array())
    # fem.plot(markers)
    # plt.show()

    return mesh, dx, ds, boundary_markers, domain_markers
