import torch
import torch.nn as nn
import math
import numpy as np
from core.physics_constraints import Constants, CalphadPolynomials, RedlichKisterPolynomial, SublatticeModel

class CompoundEnergyFormalism:
    def __init__(self, sublattice_sites, sublattice_species, endmember_energies):
        self.sites = sublattice_sites
        self.species = sublattice_species
        self.n_sublattices = len(sublattice_sites)
        self.endmembers = endmember_energies
        self.interaction_params = {}

    def set_interaction(self, sublattice_idx, species_pair, L_params):
        self.interaction_params[(sublattice_idx, species_pair)] = L_params

    def reference_energy(self, site_fractions, T):
        g_ref = torch.tensor(0.0)
        n_endmembers = 1
        for s in range(self.n_sublattices):
            n_endmembers *= len(self.species[s])
        indices = self._generate_endmember_indices()
        for idx_tuple in indices:
            product = torch.tensor(1.0)
            for s in range(self.n_sublattices):
                product = product * site_fractions[s][idx_tuple[s]]
            key = tuple(self.species[s][idx_tuple[s]] for s in range(self.n_sublattices))
            if key in self.endmembers:
                g_ref = g_ref + product * self.endmembers[key](T)
        return g_ref

    def ideal_mixing_energy(self, site_fractions, T):
        g_id = torch.tensor(0.0)
        for s in range(self.n_sublattices):
            a_s = self.sites[s]
            for i in range(len(self.species[s])):
                y = site_fractions[s][i]
                if y > 1e-15:
                    g_id = g_id + a_s * Constants.R * T * y * torch.log(y)
        return g_id

    def excess_energy(self, site_fractions, T):
        g_ex = torch.tensor(0.0)
        for (sub_idx, pair), L_list in self.interaction_params.items():
            i_spec, j_spec = pair
            i_idx = self.species[sub_idx].index(i_spec)
            j_idx = self.species[sub_idx].index(j_spec)
            y_i = site_fractions[sub_idx][i_idx]
            y_j = site_fractions[sub_idx][j_idx]
            product_other = torch.tensor(1.0)
            for s in range(self.n_sublattices):
                if s != sub_idx:
                    for k in range(len(self.species[s])):
                        product_other = product_other * site_fractions[s][k]
            for nu, L_fn in enumerate(L_list):
                L_val = L_fn(T)
                g_ex = g_ex + product_other * y_i * y_j * L_val * (y_i - y_j)**nu
        return g_ex

    def total_gibbs(self, site_fractions, T):
        return self.reference_energy(site_fractions, T) + self.ideal_mixing_energy(site_fractions, T) + self.excess_energy(site_fractions, T)

    def chemical_potential(self, site_fractions, T, sublattice_idx, species_idx, dx=1e-6):
        y_orig = site_fractions[sublattice_idx][species_idx].clone()
        site_fractions[sublattice_idx][species_idx] = y_orig + dx
        g_plus = self.total_gibbs(site_fractions, T)
        site_fractions[sublattice_idx][species_idx] = y_orig - dx
        g_minus = self.total_gibbs(site_fractions, T)
        site_fractions[sublattice_idx][species_idx] = y_orig
        return (g_plus - g_minus) / (2 * dx)

    def driving_force(self, site_fractions, T, product_fractions):
        g_parent = self.total_gibbs(site_fractions, T)
        g_product = self.total_gibbs(product_fractions, T)
        return g_parent - g_product

    def _generate_endmember_indices(self):
        import itertools
        ranges = [range(len(self.species[s])) for s in range(self.n_sublattices)]
        return list(itertools.product(*ranges))

class MagneticContribution:
    def __init__(self, p=0.4, T_c_fn=None, beta_fn=None):
        self.p = p
        self.T_c_fn = T_c_fn if T_c_fn else lambda comp: torch.tensor(1043.0)
        self.beta_fn = beta_fn if beta_fn else lambda comp: torch.tensor(2.22)

    def inden_jarl(self, tau):
        A = 518.0/1125 + 11692.0/15975 * (1/self.p - 1)
        if tau <= 1.0:
            f = 1 - (79*tau**(-1)/(140*self.p) + 474/497*(1/self.p - 1)*(tau**3/6 + tau**9/135 + tau**15/600))
        else:
            f = -(tau**(-5)/10 + tau**(-15)/315 + tau**(-25)/1500) / A
        return f

    def g_mag(self, T, composition):
        T_c = self.T_c_fn(composition)
        beta = self.beta_fn(composition)
        tau = T / T_c
        f_tau = self.inden_jarl(tau.item())
        return Constants.R * T * torch.log(beta + 1) * f_tau

class ElasticEnergyContribution:
    def __init__(self, c11, c12, c44):
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44

    def stiffness_tensor_cubic(self):
        C = torch.zeros(6, 6)
        C[0,0] = C[1,1] = C[2,2] = self.c11
        C[0,1] = C[0,2] = C[1,0] = C[1,2] = C[2,0] = C[2,1] = self.c12
        C[3,3] = C[4,4] = C[5,5] = self.c44
        return C

    def compliance_tensor(self):
        return torch.linalg.inv(self.stiffness_tensor_cubic())

    def elastic_strain_energy(self, strain_voigt):
        C = self.stiffness_tensor_cubic()
        stress = torch.matmul(C, strain_voigt)
        return 0.5 * torch.dot(strain_voigt, stress)

    def misfit_strain_energy(self, delta_a_over_a, volume):
        eta = delta_a_over_a / 3.0
        strain = torch.tensor([eta, eta, eta, 0, 0, 0])
        return self.elastic_strain_energy(strain) * volume

    def anisotropy_factor(self):
        return 2 * self.c44 / (self.c11 - self.c12)

    def bulk_modulus(self):
        return (self.c11 + 2 * self.c12) / 3

    def shear_modulus_voigt(self):
        return (self.c11 - self.c12 + 3 * self.c44) / 5

    def shear_modulus_reuss(self):
        return 5 * self.c44 * (self.c11 - self.c12) / (4 * self.c44 + 3 * (self.c11 - self.c12))

    def youngs_modulus(self):
        K = self.bulk_modulus()
        G = (self.shear_modulus_voigt() + self.shear_modulus_reuss()) / 2
        return 9 * K * G / (3 * K + G)

    def poisson_ratio(self):
        K = self.bulk_modulus()
        G = (self.shear_modulus_voigt() + self.shear_modulus_reuss()) / 2
        return (3 * K - 2 * G) / (2 * (3 * K + G))

class InterfacialEnergyModel:
    def __init__(self, sigma_0, delta, V_m):
        self.sigma_0 = sigma_0
        self.delta = delta
        self.V_m = V_m

    def gibbs_thomson(self, r, T):
        return 2 * self.sigma_0 * self.V_m / (r * Constants.R * T)

    def capillarity_length(self, T, delta_G_v):
        return 2 * self.sigma_0 / (torch.abs(delta_G_v) + 1e-12)

    def nucleation_barrier(self, delta_G_v):
        return 16 * math.pi * self.sigma_0**3 / (3 * delta_G_v**2 + 1e-12)

    def critical_radius(self, delta_G_v):
        return -2 * self.sigma_0 / (delta_G_v + 1e-12)

    def nucleation_rate(self, T, delta_G_v, D, a0):
        delta_G_star = self.nucleation_barrier(delta_G_v)
        Z = torch.sqrt(torch.abs(delta_G_v) / (6 * math.pi * Constants.K_B * T * (self.critical_radius(delta_G_v))**2 + 1e-12))
        beta_star = 4 * math.pi * (self.critical_radius(delta_G_v))**2 * D / a0**4
        N_0 = 1e28
        return N_0 * Z * beta_star * torch.exp(-delta_G_star / (Constants.K_B * T))

class TransformationKinetics:
    @staticmethod
    def jmak(t, k, n):
        return 1 - torch.exp(-k * t**n)

    @staticmethod
    def jmak_rate(t, k, n):
        return n * k * t**(n-1) * torch.exp(-k * t**n)

    @staticmethod
    def austin_rickett(t, k, n):
        return 1 - 1 / (1 + k * t**n)

    @staticmethod
    def kissinger(T_p, beta, Ea, R=8.314):
        return torch.log(beta / T_p**2) + Ea / (R * T_p)

    @staticmethod
    def ozawa_flynn_wall(T, beta, Ea, A, R=8.314):
        p = Ea / (R * T)
        return torch.log(A * Ea / R) - torch.log(beta) - 5.331 - 1.052 * p

    @staticmethod
    def friedman(da_dt, T, Ea, R=8.314):
        return torch.log(da_dt) + Ea / (R * T)

    @staticmethod
    def diffusion_controlled_growth_rate(D, c_inf, c_interface, c_precipitate, r):
        return D * (c_inf - c_interface) / ((c_precipitate - c_interface) * r + 1e-12)

    @staticmethod
    def zener_growth_rate(D, c_0, c_eq, c_p, r, omega=1.0):
        supersaturation = (c_0 - c_eq) / (c_p - c_eq + 1e-12)
        return omega * D * supersaturation / (r + 1e-12)

    @staticmethod
    def soft_impingement_factor(X_e, X):
        return (X_e - X) / (X_e + 1e-12)

class DiffusionMulticomponent:
    def __init__(self, n_components):
        self.n = n_components
        self.D_matrix = torch.zeros(n_components - 1, n_components - 1)

    def set_diffusion_matrix(self, D):
        self.D_matrix = D

    def onsager_from_mobility(self, mobility_matrix, c, T):
        D = torch.zeros_like(self.D_matrix)
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                D[i, j] = sum(
                    mobility_matrix[i, k] * (
                        self._thermodynamic_factor(c, k, j, T)
                    ) for k in range(self.n - 1)
                )
        return D

    def _thermodynamic_factor(self, c, k, j, T, dx=1e-6):
        c_plus = c.clone()
        c_plus[j] += dx
        c_minus = c.clone()
        c_minus[j] -= dx
        mu_plus = Constants.R * T * torch.log(c_plus[k] + 1e-12)
        mu_minus = Constants.R * T * torch.log(c_minus[k] + 1e-12)
        return (mu_plus - mu_minus) / (2 * dx)

    def flux(self, c, grad_c):
        return -torch.matmul(self.D_matrix, grad_c[:self.n-1])

    def interdiffusion_coefficient(self, D_intrinsic, x, V_m):
        D_tilde = torch.zeros_like(D_intrinsic)
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                D_tilde[i, j] = D_intrinsic[i, j] - x[i] * sum(D_intrinsic[k, j] for k in range(self.n - 1))
        return D_tilde

class GrainBoundaryDiffusion:
    def __init__(self, D_gb0, Q_gb, delta_gb):
        self.D_gb0 = D_gb0
        self.Q_gb = Q_gb
        self.delta = delta_gb

    def D_gb(self, T):
        return self.D_gb0 * torch.exp(-self.Q_gb / (Constants.R * T))

    def effective_D_harrison_type_B(self, D_lattice, T, d_grain):
        D_gb = self.D_gb(T)
        alpha = self.delta * D_gb / (d_grain * D_lattice + 1e-12)
        if alpha < 0.1:
            return D_lattice * (1 + 2 * alpha)
        else:
            return D_lattice + self.delta * D_gb / d_grain

    def segregation_isotherm_mclean(self, X_bulk, delta_G_seg, T):
        return X_bulk * torch.exp(-delta_G_seg / (Constants.R * T)) / (1 + X_bulk * (torch.exp(-delta_G_seg / (Constants.R * T)) - 1))

class AdvancedElectrochemistry:
    @staticmethod
    def asymmetric_butler_volmer(eta, T, j0, alpha_a, alpha_c, n_e=1):
        f = n_e * Constants.F / (Constants.R * T)
        return j0 * (torch.exp(alpha_a * f * eta) - torch.exp(-alpha_c * f * eta))

    @staticmethod
    def marcus_electron_transfer(eta, T, lambda_reorg, H_ab, n_e=1):
        delta_G = n_e * Constants.F * eta
        exponent = -(lambda_reorg + delta_G)**2 / (4 * lambda_reorg * Constants.K_B * T * Constants.e)
        prefactor = (2 * math.pi / (Constants.h / (2*math.pi))) * H_ab**2 / torch.sqrt(4 * math.pi * lambda_reorg * Constants.K_B * T * Constants.e)
        return prefactor * torch.exp(exponent)

    @staticmethod
    def savant_butler_volmer_with_adsorption(eta, T, j0, theta, alpha_a=0.5):
        f = Constants.F / (Constants.R * T)
        frumkin = torch.exp(-2 * 3.0 * theta)
        return j0 * frumkin * ((1 - theta) * torch.exp(alpha_a * f * eta) - theta * torch.exp(-(1-alpha_a) * f * eta))

    @staticmethod
    def exchange_current_density(k0, c_ox, c_red, c_ref, alpha_a, alpha_c):
        return Constants.F * k0 * (c_ox / c_ref)**alpha_c * (c_red / c_ref)**alpha_a

    @staticmethod
    def double_layer_capacitance_gouy_chapman(c_bulk, T, epsilon_r=78.5):
        epsilon = epsilon_r * Constants.epsilon_0
        kappa = torch.sqrt(2 * Constants.F**2 * c_bulk * 1000 / (epsilon * Constants.R * T))
        return epsilon * kappa

    @staticmethod
    def stern_layer_capacitance(epsilon_r, d_stern):
        return epsilon_r * Constants.epsilon_0 / d_stern

class SEIGrowthModels:
    @staticmethod
    def single_electron_tunneling(L_sei, V_barrier, m_eff):
        kappa = torch.sqrt(2 * m_eff * Constants.e * V_barrier) / (Constants.h / (2*math.pi))
        return torch.exp(-2 * kappa * L_sei)

    @staticmethod
    def ploehn_model(t, D_s, k_s, c_s0, M_sei, rho_sei):
        alpha = D_s * c_s0 / (rho_sei / M_sei)
        return torch.sqrt(2 * alpha * t)

    @staticmethod
    def safari_model(j_sei, L_sei, D_ec, c_ec, kappa_sei, R_sei_0):
        R_film = R_sei_0 + L_sei / kappa_sei
        j_diff = Constants.F * D_ec * c_ec / (L_sei + 1e-12)
        return 1.0 / (1.0 / (j_sei + 1e-12) + 1.0 / (j_diff + 1e-12))

    @staticmethod
    def ramadass_sei(eta_sei, T, U_sei, j0_sei, alpha=0.5):
        eta_eff = eta_sei - U_sei
        f = Constants.F / (Constants.R * T)
        return -j0_sei * torch.exp(-alpha * f * eta_eff)

    @staticmethod
    def dual_layer_sei(L_inner, L_outer, D_inner, D_outer, c_solvent, k_inner, k_outer, T):
        j_inner = Constants.F * D_inner * c_solvent / (L_inner + 1e-12)
        j_outer = Constants.F * D_outer * c_solvent / (L_outer + 1e-12)
        dL_inner = (1e-6 / 2) * j_inner / Constants.F
        dL_outer = (1e-6 / 2) * j_outer / Constants.F
        return dL_inner, dL_outer

class LithiumPlatingModels:
    @staticmethod
    def arora_plating(eta_n, T, k_pl, alpha=0.5):
        U_Li = 0.0
        eta_pl = eta_n - U_Li
        f = Constants.F / (Constants.R * T)
        return -k_pl * torch.exp(-alpha * f * eta_pl) * (eta_pl < 0).float()

    @staticmethod
    def reversible_plating(c_Li_plated, k_strip, T):
        f = Constants.F / (Constants.R * T)
        return k_strip * c_Li_plated * torch.exp(-0.3 * f * 0.01)

    @staticmethod
    def dendrite_growth_rate(j_Li, D_Li, c_Li, t_plus, tip_radius):
        j_lim = 2 * Constants.F * D_Li * c_Li / (tip_radius * (1 - t_plus))
        return j_Li / (j_lim + 1e-12)

class ParticleMechanics:
    @staticmethod
    def zhang_diffusion_induced_stress(c, c_avg, omega, E, nu, r, R_p):
        sigma_r = (2 * omega * E) / (9 * (1 - nu)) * (c_avg - (3 / r**3) * torch.cumsum(c * r**2 * (R_p / len(c)), dim=0))
        sigma_t = (omega * E) / (9 * (1 - nu)) * (2 * c_avg + (3 / r**3) * torch.cumsum(c * r**2 * (R_p / len(c)), dim=0) - 3 * c)
        return sigma_r, sigma_t

    @staticmethod
    def crack_propagation_paris_law(delta_K, C_paris, m_paris):
        return C_paris * delta_K**m_paris

    @staticmethod
    def stress_intensity_factor_penny(sigma, a):
        return 2 * sigma * torch.sqrt(a / math.pi)

    @staticmethod
    def griffith_criterion(sigma, a, E, gamma_s):
        K_IC = torch.sqrt(2 * E * gamma_s)
        K_I = sigma * torch.sqrt(math.pi * a)
        return K_I >= K_IC

    @staticmethod
    def contact_loss_fraction(N_cycles, k_cl, m_cl):
        return 1 - torch.exp(-k_cl * N_cycles**m_cl)

class ThermalRunawayChain:
    @staticmethod
    def sei_decomposition_rate(T, A_sei=1.667e15, Ea_sei=1.3508e5):
        return A_sei * torch.exp(-Ea_sei / (Constants.R * T))

    @staticmethod
    def anode_electrolyte_rate(T, c_an, A_ae=2.5e13, Ea_ae=1.35e5):
        return c_an * A_ae * torch.exp(-Ea_ae / (Constants.R * T))

    @staticmethod
    def cathode_decomposition_rate(T, alpha_cat, A_cat=1.75e9, Ea_cat=1.145e5):
        return alpha_cat * A_cat * torch.exp(-Ea_cat / (Constants.R * T))

    @staticmethod
    def electrolyte_decomposition_rate(T, c_el, A_el=5.14e25, Ea_el=2.74e5):
        return c_el * A_el * torch.exp(-Ea_el / (Constants.R * T))

    @staticmethod
    def heat_generation(r_sei, r_ae, r_cat, r_el, H_sei=2.57e5, H_ae=1.714e6, H_cat=3.14e5, H_el=1.55e5):
        return r_sei * H_sei + r_ae * H_ae + r_cat * H_cat + r_el * H_el

    @staticmethod
    def thermal_runaway_ode(T, c_sei, c_an, alpha_cat, c_el, m_cp, hA, T_amb):
        r1 = ThermalRunawayChain.sei_decomposition_rate(T)
        r2 = ThermalRunawayChain.anode_electrolyte_rate(T, c_an)
        r3 = ThermalRunawayChain.cathode_decomposition_rate(T, alpha_cat)
        r4 = ThermalRunawayChain.electrolyte_decomposition_rate(T, c_el)
        Q = ThermalRunawayChain.heat_generation(r1, r2, r3, r4)
        dT_dt = (Q - hA * (T - T_amb)) / m_cp
        dc_sei = -r1
        dc_an = -r2
        dalpha_cat = r3
        dc_el = -r4
        return dT_dt, dc_sei, dc_an, dalpha_cat, dc_el
