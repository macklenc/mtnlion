import fenics as fem
import numpy as np


def generate_domain(comsol):
    boundaries = range(4)

    # Create 1D mesh
    mesh = fem.IntervalMesh(len(np.unique(comsol.data.mesh.mesh)) - 1, 0, 3)
    mesh.coordinates()[:] = np.array([np.unique(comsol.data.mesh.mesh)]).transpose()

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
