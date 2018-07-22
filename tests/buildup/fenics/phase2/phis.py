import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import utilities


class Phis():
    def __init__(self, domain, Acell, sigma_eff, L, a_s, F):
        self.phi = fem.TrialFunction(domain.V)
        self.v = fem.TestFunction(domain.V)
        self.domain = domain
        self.L, self.a_s, self.F = L, a_s, F
        self.Acell, self.sigma_eff = Acell, sigma_eff

        self.a = -self.sigma_eff / self.L * fem.dot(fem.grad(self.phi), fem.grad(self.v)) * self.domain.dx

    def get(self, jbar, neumann):
        neumann = neumann * self.v * self.domain.ds(4)

        lin = self.L * self.a_s * self.F * jbar * self.v * self.domain.dx + neumann
        return self.a, lin


def solve(time, domain, Acell, sigma_eff, L, a_s, F, Iapp, true_sol):
    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(true_sol.mesh)))
    phi_s = Phis(domain, Acell, sigma_eff, L, a_s, F)
    bm = domain.boundary_markers
    V = domain.V

    jbar = fem.Function(domain.V)
    phis = fem.Function(domain.V)

    bc = [fem.DirichletBC(V, 0.0, bm, 1), 0]
    for i, j in enumerate(true_sol.data.j):
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')
        bc[1] = fem.DirichletBC(V, true_sol.data.phis[i, -1], bm, 4)

        a, lin = phi_s.get(jbar, fem.Constant(Iapp[i]) / Acell)

        # Solve
        fem.solve(a == lin, phis, bc)
        u_array[i, :] = phis.vector().get_local()[fem.vertex_to_dof_map(domain.V)]

    return u_array, true_sol


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]
    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    cmn = common.Common(time)
    domain = cmn.domain

    fenics, comsol = solve(time, domain, cmn.Acell, cmn.sigma_eff, cmn.Lc, cmn.a_s, cmn.F, Iapp, cmn.comsol_solution)
    utilities.report(comsol.neg, time, fenics[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    plt.savefig('comsol_compare_phis_neg.png')
    plt.show()
    utilities.report(comsol.pos, time, fenics[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind], '$\Phi_s^{neg}$')
    plt.savefig('comsol_compare_phis_pos.png')
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)
