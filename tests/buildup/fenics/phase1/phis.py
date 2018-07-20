import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import utilities


class Phis():
    def __init__(self, domain, Acell, sigma_eff, L, a_s, F, dirichlet, neumann):
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.phi = fem.Function(domain.V)

        phi = fem.TrialFunction(domain.V)
        v = fem.TestFunction(domain.V)

        a = -sigma_eff / L * fem.dot(fem.grad(phi), fem.grad(v)) * domain.dx
        lin = L * a_s * F * phi * v * domain.dx

        self.A = fem.assemble(a)
        self.M = fem.assemble(lin)

        self.neumann_vec = fem.assemble(fem.Constant(1) / Acell * v * domain.ds(4))

    def get(self, jbar, Iapp=1):
        neumann = Iapp * self.neumann_vec
        b = self.M * jbar.vector() + neumann

        for k in self.dirichlet:
            k.apply(self.A, b)

        return self.A, self.phi, b


def solve(time, domain, Acell, sigma_eff, L, a_s, F, Iapp, true_sol):
    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(true_sol.mesh)))

    nc = fem.Constant(1) / Acell * domain.ds(4)
    bc = [fem.DirichletBC(domain.V, 0.0, domain.boundary_markers, 1), 0]
    phi_s = Phis(domain, Acell, sigma_eff, L, a_s, F, bc, nc)

    jbar = fem.Function(domain.V)

    for i, j in enumerate(true_sol.data.j):
        jbar.vector()[:] = j[fem.dof_to_vertex_map(domain.V)].astype('double')

        # Solve
        phi_s.dirichlet[1] = fem.DirichletBC(domain.V, true_sol.data.phis[i, -1], domain.boundary_markers, 4)
        A, phi, b = phi_s.get(jbar, Iapp[i])
        fem.solve(A, phi.vector(), b)
        u_array[i, :] = phi.vector().get_local()[fem.vertex_to_dof_map(domain.V)]

    return u_array, true_sol


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]
    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    cmn = common.Common(time)
    domain = cmn.domain

    with utilities.Timer() as t:
        fenics, comsol = solve(time, domain, cmn.Acell, cmn.sigma_eff, cmn.Lc,
                               cmn.a_s, cmn.F, Iapp, cmn.comsol_solution)

    print('Runtime: {}s'.format(t.interval))
    utilities.report(comsol.neg, time, fenics[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    plt.savefig('comsol_compare_phis_neg.png')
    plt.show()

    utilities.report(comsol.pos, time, fenics[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind], '$\Phi_s^{neg}$')
    plt.savefig('comsol_compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)
