import sys

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import utilities


class J():
    def __init__(self, Uocp_neg, Uocp_pos, V, degree=1):
        self.sym_j = self.mksym(Uocp_neg, Uocp_pos)
        init1 = fem.Constant(0)
        init2 = fem.Function(V)
        self.j = fem.Expression(sym.printing.ccode(self.sym_j), csmax=init1, cse=init2, ce=init2, ce0=init1,
                                alpha=init1,
                                k_norm_ref=init1, phie=init2, phis=init2, R=init1, F=init1, Tref=init1, degree=degree)

    def get(self, csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, F, R, T):
        self.j.csmax, self.j.cse, self.j.ce, self.j.ce0 = csmax, cse, ce, ce0
        self.j.alpha, self.j.k_norm_ref, self.j.phie = alpha, k_norm_ref, phie
        self.j.phis, self.j.R, self.j.F, self.j.Tref = phis, R, F, T

        return self.j

    def mksym(self, Uocp_neg, Uocp_pos):
        number = sym.Symbol('n')
        csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
        x, f, r, Tref = sym.symbols('x[0], F, R, Tref')

        nabs = ((sym.sign(number) + 1) / 2) * sym.Abs(number)
        s1 = nabs.subs(number, ((csmax - cse) / csmax) * (ce / ce0)) ** (1 - alpha)
        s2 = nabs.subs(number, cse / csmax) ** alpha
        sym_flux = k_norm_ref * s1 * s2
        soc = cse / csmax

        tmpx = sym.Symbol('x')
        # Uocp_pos = Uocp_pos * 1.00025  #########################################FIX ME!!!!!!!!!!!!!!!!!!*1.00025

        Uocp_neg = Uocp_neg.subs(tmpx, soc)
        Uocp_pos = Uocp_pos.subs(tmpx, soc)

        uocp = sym.Piecewise((Uocp_neg, x <= 1 + fem.DOLFIN_EPS + 0.1), (Uocp_pos, x >= 2 - fem.DOLFIN_EPS), (0, True))

        eta = phis - phie - uocp
        sym_j = sym_flux * (sym.exp((1 - alpha) * f * eta / (r * Tref)) - sym.exp(-alpha * f * eta / (r * Tref)))

        return sym_j


def solve(time, domain, csmax, ce0, alpha, k_norm_ref, F, R, T, Uocp_neg, Uocp_pos, comsol):
    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(comsol.mesh)))

    # Define function space and basis functions
    V = domain.V
    cse = fem.Function(V)
    ce = fem.Function(V)
    phis = fem.Function(V)
    phie = fem.Function(V)

    # Evaluate expressions
    csmax = fem.interpolate(csmax, V)
    alpha = fem.interpolate(alpha, V)

    jbar = J(Uocp_neg, Uocp_pos, V)

    for i, j in enumerate(comsol.data.j):
        # Turn COMSOL solutions into FEniCS functions
        cse.vector()[:] = comsol.data.cse[i, fem.dof_to_vertex_map(V)].astype('double')
        ce.vector()[:] = comsol.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')
        phie.vector()[:] = comsol.data.phie[i, fem.dof_to_vertex_map(V)].astype('double')
        phis.vector()[:] = comsol.data.phis[i, fem.dof_to_vertex_map(V)].astype('double')

        expr = jbar.get(csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, F, R, T)
        u_array[i, :] = fem.interpolate(expr, V).vector().get_local()[fem.vertex_to_dof_map(V)]

    return u_array

def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    k_norm_ref, csmax, alpha, ce0 = cmn.k_norm_ref, cmn.csmax, cmn.alpha, cmn.ce0
    F, R, Tref = cmn.F, cmn.R, cmn.Tref

    fenics = solve(time, domain, csmax, ce0, alpha, k_norm_ref, F, R, Tref,
                   cmn.params.neg.Uocp[0], cmn.params.pos.Uocp[0], comsol)

    utilities.report(comsol.mesh[comsol.neg_ind], time, fenics[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind], '$j_{neg}$')
    plt.savefig('comsol_compare_j_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time, fenics[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind], '$j_{pos}$')
    plt.savefig('comsol_compare_j_pos.png')
    plt.show()

    return


if __name__ == '__main__':
    sys.exit(main())
