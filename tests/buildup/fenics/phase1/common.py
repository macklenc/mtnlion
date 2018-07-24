import fenics as fem
import numpy as np

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


class Domain():
    def __init__(self, V, dx, ds, boundary_markers, domain_markers):
        self.V = V
        self.dx = dx
        self.ds = ds
        self.neg_marker, self.sep_marker, self.pos_marker = (1, 2, 3)
        self.boundary_markers = boundary_markers
        self.domain_markers = domain_markers


class Common:
    def __init__(self, time):
        self.time = time

        # Collect required data
        self.comsol_data, self.params = utilities.gather_data()
        self.time_ind = engine.find_ind_near(self.comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(self.comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        self.mesh, self.dx, self.ds, self.bm, self.dm = domain2.generate_domain(self.comsol_solution.mesh)
        self.neg_submesh = fem.SubMesh(self.mesh, self.dm, 1)
        self.sep_submesh = fem.SubMesh(self.mesh, self.dm, 2)
        self.pos_submesh = fem.SubMesh(self.mesh, self.dm, 3)

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

        self.V = fem.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.neg_V = fem.FunctionSpace(self.neg_submesh, 'Lagrange', 1)
        self.sep_V = fem.FunctionSpace(self.sep_submesh, 'Lagrange', 1)
        self.pos_V = fem.FunctionSpace(self.pos_submesh, 'Lagrange', 1)
        self.domain = Domain(self.V, self.dx, self.ds, self.bm, self.dm)


class Common2:
    def __init__(self, time, mesh):
        self.time = time

        # Collect required data
        self.comsol_data, self.params = utilities.gather_data()
        self.time_ind = engine.find_ind_near(self.comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(self.comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        self.mesh, self.dx, self.ds, self.bm, self.dm = domain2.generate_domain2(mesh)

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



        self.Ds = utilities.mkparam(self.dm, self.params.neg.Ds_ref, 0, self.params.pos.Ds_ref)
