import fenics as fem
import sympy as sym


# TODO: add internal Neumann conditions or remove boundary Neumann conditions

def phis(jbar, phis, v, dx, a_s, F, sigma_eff, L, ds=0, neumann=0, **kwargs):
    a = -sigma_eff / L * fem.dot(fem.grad(phis), fem.grad(v)) * dx
    Lin = L * a_s * F * jbar * v * dx + neumann * v * ds

    return a - Lin


def phie(jbar, ce, phie, v, dx, kappa_eff, kappa_Deff, L, a_s, F, ds=0, neumann=0, **kwargs):
    a = kappa_eff / L * fem.dot(fem.grad(phie), fem.grad(v)) * dx
    Lin = L * a_s * F * jbar * v * dx - kappa_Deff / L * \
        fem.dot(fem.grad(fem.ln(ce)), fem.grad(v)) * dx + neumann * v * ds

    return a - Lin


class K(fem.Expression):
    def __init__(self, materials, csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, R, F, Tref, j, **kwargs):
        self.materials = materials
        self.csmax = csmax
        self.cse = cse
        self.ce = ce
        self.ce0 = ce0
        self.alpha = alpha
        self.k_norm_ref = k_norm_ref
        self.phie = phie
        self.phis = phis
        self.R = R
        self.F = F
        self.Tref = Tref
        self.j = j

    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0 or self.materials[cell.index] == 2:
            values[0] = self.j(self.csmax(x), self.cse(x), self.ce(x), self.ce0(x), self.alpha(x),
                               self.k_norm_ref(x), self.phie(x), self.phis(x), x, self.F(x), self.R(x), self.Tref(x))
        else:
            values[0] = 0


# TODO: add explicit euler class?
def ce_explicit_euler(jbar, ce_1, ce, v, dx, dt, a_s, eps_e, De_eff, t_plus, L,
                      ds=0, neumann=0, **kwargs):
    a = L * eps_e * ce * v * dx
    Lin = L * eps_e * ce_1 * v * dx - dt * De_eff / L * fem.dot(fem.grad(ce_1), fem.grad(v)) * dx + \
          dt * L * a_s * (fem.Constant(1) - t_plus) * jbar * v * dx + neumann * v * ds

    return a - Lin


def cs(cs_1, cs, v, dx, dt, Rs, Ds_ref, **kwargs):
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    a = Rs * rbar2 * cs * v * dx
    Lin = Rs * rbar2 * cs_1 * v * dx - dt * Ds_ref * rbar2 / Rs * fem.dot(cs_1.dx(1), v.dx(1)) * dx

    return a - Lin


def j(ce, cse, phie, phis, csmax, ce0, alpha, k_norm_ref, F, R, Tref, Uocp, degree=1, **kwargs):
    return fem.Expression(sym.printing.ccode(_sym_j()[0]),
                          ce=ce, cse=cse, phie=phie, phis=phis, csmax=csmax,
                          ce0=ce0, alpha=alpha, k_norm_ref=k_norm_ref, F=F,
                          R=R, Tref=Tref, Uocp=Uocp, degree=degree)


def j_new(ce, cse, phie, phis, csmax, ce0, alpha, k_norm_ref, F, R, Tref, Uocp_neg, Uocp_pos, dm, V, degree=1,
          **kwargs):
    # return fem.Expression(sym.printing.ccode(_sym_j(Uocp_neg, Uocp_pos)),
    #                       ce=ce, cse=cse, phie=phie, phis=phis, csmax=csmax,
    #                       ce0=ce0, alpha=alpha, k_norm_ref=k_norm_ref, F=F,
    #                       R=R, Tref=Tref, degree=degree)
    csmax = fem.interpolate(csmax, V)
    ce0 = fem.interpolate(ce0, V)
    alpha = fem.interpolate(alpha, V)
    k_norm_ref = fem.interpolate(k_norm_ref, V)

    _, sym_jeval = _sym_j(Uocp_neg, Uocp_pos)
    return K(dm, csmax=csmax, cse=cse, ce=ce, ce0=ce0, alpha=alpha, k_norm_ref=k_norm_ref, phie=phie,
             phis=phis, R=R, F=F, Tref=Tref, j=sym_jeval, degree=1)


def eval_j(x, ce, cse, phie, phis, csmax, ce0, alpha, k_norm_ref, F, R, Tref, Uocp_neg, Uocp_pos, **kwargs):
    _, jeval = _sym_j(Uocp_neg, Uocp_pos)
    return jeval(csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, x, F, R, Tref)


def _sym_j():
    number = sym.Symbol('n')
    csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
    x, f, r, Tref = sym.symbols('x[0], F, R, Tref')

    nabs = ((sym.sign(number) + 1) / 2) * sym.Abs(number)
    s1 = nabs.subs(number, ((csmax - cse) / csmax) * (ce / ce0)) ** (1 - alpha)
    s2 = nabs.subs(number, cse / csmax) ** alpha
    sym_flux = k_norm_ref * s1 * s2
    soc = cse / csmax

    tmpx = sym.Symbol('soc')
    # Uocp_pos = Uocp_pos * 1.00025  #########################################FIX ME!!!!!!!!!!!!!!!!!!*1.00025

    uocp = sym.Symbol('Uocp')

    eta = phis - phie - uocp
    sym_j = sym_flux * (sym.exp((1 - alpha) * f * eta / (r * Tref)) - sym.exp(-alpha * f * eta / (r * Tref)))
    sym_j_domain = sym.Piecewise((sym_j, x <= 1), (sym_j, x >= 2), (0, True))

    jeval = sym.lambdify((csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, x, f, r, Tref), sym_j, 'numpy')

    return sym_j_domain, jeval
