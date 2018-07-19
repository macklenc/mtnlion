import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import mtnlion.engine as engine
import utilities


class Domain():
    def __init__(self, V, dx, ds, boundary_markers, domain_markers):
        self.V = V
        self.dx = dx
        self.ds = ds
        self.boundary_markers = boundary_markers
        self.domain_markers = domain_markers


class Phis():
    def __init__(self, domain, Acell, sigma_eff, L, a_s, F, dirichlet, neumann):
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.phi = fem.Function(domain.V)

        phi = fem.TrialFunction(domain.V)
        v = fem.TestFunction(domain.V)

        a = -sigma_eff / L * fem.dot(fem.grad(phi), fem.grad(v)) * domain.dx
        self.lin = L * a_s * F * phi * v * domain.dx

        self.A = fem.assemble(a)
        self.M = fem.assemble(L * a_s * F * phi * v * domain.dx)

        self.neumann_vec = fem.assemble(fem.Constant(1) / Acell * v * domain.ds(4))

    def solve(self, jbar, Iapp=1):
        neumann = Iapp * self.neumann_vec
        b = self.M * jbar.vector() + neumann

        for k in self.dirichlet:
            k.apply(self.A, b)

        fem.solve(self.A, self.phi.vector(), b)
        return self.phi


def phis(time):
    # Collect common data
    cmn = common.Common(time)
    true_sol = cmn.comsol_solution

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(true_sol.mesh)))

    mesh = cmn.mesh
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    domain = Domain(V, cmn.dx, cmn.ds, cmn.bm, cmn.dm)
    bc = [fem.DirichletBC(V, 0.0, cmn.bm, 1), 0]
    nc = fem.Constant(1) / cmn.Acell * cmn.ds(4)
    phi_s = Phis(domain, cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, cmn.F, bc, nc)

    # create local variables
    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]
    jbar = fem.Function(V)

    for i, j in enumerate(true_sol.data.j):
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')

        # Solve
        phi_s.dirichlet[1] = fem.DirichletBC(V, true_sol.data.phis[i, -1], cmn.bm, 4)
        phi = phi_s.solve(jbar, Iapp[i])
        u_array[i, :] = phi.vector().get_local()[fem.vertex_to_dof_map(V)]

    return u_array, true_sol


if __name__ == '__main__':
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]
    fenics, comsol = phis(time)

    print(engine.rmse(fenics[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind]))
    print(engine.rmse(fenics[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind]))

    utilities.overlay_plt(comsol.neg, time, '$\Phi_s^{neg}$',
                          fenics[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind])
    plt.savefig('comsol_compare_phis_neg.png')
    plt.show()

    utilities.overlay_plt(comsol.pos, time, '$\Phi_s^{pos}$',
                          fenics[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind])
    plt.savefig('comsol_compare_phis_pos.png')
    plt.show()

