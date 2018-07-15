import fenics as fem

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


class Common:
    def __init__(self, time):
        self.time = time

        # Collect required data
        self.comsol_data, self.params = utilities.gather_data()
        self.time_ind = engine.find_ind_near(self.comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(self.comsol_data.filter_time(self.time_ind))

        self.mesh, self.dx, self.ds, self.bm, self.dm = domain2.generate_domain(self.comsol_solution.mesh)

        # Initialize parameters
        self.F = fem.Constant(96487)
        self.R = fem.Constant(8.314)  # universal gas constant
        self.T = fem.Constant(298.15)
        self.I_1C = fem.Constant(20.5)
        self.Iapp = [self.I_1C if 10 <= i <= 20 else -self.I_1C if 30 <= i <= 40 else fem.Constant(0) for i in time]
        self.Acell = fem.Constant(self.params.const.Acell)

        self.Lc = utilities.mkparam(self.dm, self.params.neg.L, self.params.sep.L, self.params.pos.L)
        self.sigma_ref = utilities.mkparam(self.dm, self.params.neg.sigma_ref, 0, self.params.pos.sigma_ref)
        self.eps_s = utilities.mkparam(self.dm, self.params.neg.eps_s, 0, self.params.pos.eps_s)
        self.brug_sigma = utilities.mkparam(self.dm, self.params.neg.brug_sigma, 0, self.params.pos.brug_sigma)
        self.sigma_eff = self.sigma_ref * self.eps_s ** self.brug_sigma
        self.Rs = utilities.mkparam(self.dm, self.params.neg.Rs, 0, self.params.pos.Rs)
        self.a_s = fem.Constant(3) * utilities.mkparam(self.dm, self.params.neg.eps_s / self.params.neg.Rs, 0,
                                                       self.params.pos.eps_s / self.params.pos.Rs)

        self.eps_e = utilities.mkparam(self.dm, self.params.neg.eps_e, self.params.sep.eps_e, self.params.pos.eps_e)
        self.brug_De = utilities.mkparam(self.dm, self.params.neg.brug_De,
                                         self.params.sep.brug_De, self.params.pos.brug_De)
        self.De_ref = fem.Constant(self.params.const.De_ref)
        self.de_eff = self.De_ref * self.eps_e ** self.brug_De
        self.t_plus = fem.Constant(self.params.const.t_plus)

        self.k_norm_ref = utilities.mkparam(self.dm, self.params.neg.k_norm_ref, 0, self.params.pos.k_norm_ref)
        self.csmax = utilities.mkparam(self.dm, self.params.neg.csmax, 0, self.params.pos.csmax)
        self.alpha = utilities.mkparam(self.dm, self.params.neg.alpha, 0, self.params.pos.alpha)
        self.ce0 = fem.Constant(self.params.const.ce0)
        self.Tref = fem.Constant(self.params.const.Tref)

        # self.cs0 = utilities.mkparam(self.dm, self.params.csmax*)
        self.brug_kappa = utilities.mkparam(self.dm, self.params.neg.brug_kappa, self.params.sep.brug_kappa,
                                            self.params.pos.brug_kappa)
