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


def j(ce, cse, phie, phis, Uocp, csmax, ce0, alpha, k_norm_ref, F, R, Tref, degree=1, **kwargs):
    return fem.Expression(sym.printing.ccode(_sym_j()[0]),
                          ce=ce, cse=cse, phie=phie, phis=phis, csmax=csmax,
                          ce0=ce0, alpha=alpha, k_norm_ref=k_norm_ref, F=F,
                          R=R, Tref=Tref, Uocp=Uocp, degree=degree)


def Uocp(cse, csmax, uocp_str, **kwargs):
    soc = fem.Expression('cse/csmax', cse=cse, csmax=csmax, degree=1)
    return fem.Expression(sym.printing.ccode(uocp_str), soc=soc, degree=1)


# TODO: refactor
def Uocp_interp(Uocp_neg_interp, Uocp_pos_interp, cse, csmax, utilities):
    eref_neg = utilities.fenics_interpolate(Uocp_neg_interp)
    eref_pos = utilities.fenics_interpolate(Uocp_pos_interp)

    soc = fem.Expression('cse/csmax', cse=cse, csmax=csmax, degree=1)
    Uocp_neg = utilities.compose(soc, eref_neg)
    Uocp_pos = utilities.compose(soc, eref_pos)

    return fem.Expression('x[0] <= 1.0 + DOLFIN_EPS ? neg : (x[0] >= 2.0 - DOLFIN_EPS ? pos : sep)',
                          neg=Uocp_neg, sep=fem.Constant(0), pos=Uocp_pos, degree=1)


def _sym_j():
    number = sym.Symbol('n')
    uocp = sym.Symbol('Uocp')
    csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis = sym.symbols('csmax cse ce ce0 alpha k_norm_ref phie phis')
    x, f, r, Tref = sym.symbols('x[0], F, R, Tref')

    nabs = ((sym.sign(number) + 1) / 2) * sym.Abs(number)
    s1 = nabs.subs(number, ((csmax - cse) / csmax) * (ce / ce0)) ** (1 - alpha)
    s2 = nabs.subs(number, cse / csmax) ** alpha
    sym_flux = k_norm_ref * s1 * s2

    eta = phis - phie - uocp
    sym_j = sym_flux * (sym.exp((1 - alpha) * f * eta / (r * Tref)) - sym.exp(-alpha * f * eta / (r * Tref)))
    sym_j_domain = sym.Piecewise((sym_j, x <= 1), (sym_j, x >= 2), (0, True))

    jeval = sym.lambdify((csmax, cse, ce, ce0, alpha, k_norm_ref, phie, phis, x, f, r, Tref), sym_j, 'numpy')

    return sym_j_domain, jeval
