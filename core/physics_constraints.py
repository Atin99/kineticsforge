import torch
import numpy as np
import math

class Constants:
    R = 8.31446261815324
    F = 96485.3321233100184
    K_B = 8.617333262145e-5
    h = 6.62607015e-34
    N_A = 6.02214076e23
    c = 299792458.0
    e = 1.602176634e-19
    m_e = 9.1093837015e-31
    epsilon_0 = 8.8541878128e-12
    mu_0 = 1.25663706212e-6
    sigma_sb = 5.670374419e-8
    G = 6.67430e-11

class CalphadPolynomials:
    def __init__(self):
        self.na_ghser = lambda t: -7823.518 + 120.3524*t - 24.1118*t*torch.log(t) - 0.00643*t**2 + 1.15e-6*t**3 + 70450/t
        self.fe_bcc = lambda t: -1054.83 + 124.134*t - 23.5143*t*torch.log(t) - 0.00439752*t**2 - 5.8927e-8*t**3 + 77359/t
        self.fe_fcc = lambda t: -1462.4 + 131.25*t - 24.6643*t*torch.log(t) - 0.00389*t**2 + 3.1e-7*t**3 + 69200/t
        self.mn_cbcc = lambda t: -3439.3 + 130.059*t - 23.4582*t*torch.log(t) - 0.0073476*t**2 + 6.99e-7*t**3 + 69827/t
        self.o_gas = lambda t: -1202.4 + 115.13*t - 15.421*t*torch.log(t) - 0.0025*t**2 + 3.4e-7*t**3 + 45000/t
        self.c_graphite = lambda t: -17368.441 + 170.73*t - 24.3*t*torch.log(t) - 0.000472*t**2 + 2562600/t - 2.643e8/t**2 + 1.2e10/t**3
        self.al_fcc = lambda t: -7976.15 + 137.093*t - 24.3672*t*torch.log(t) - 0.00188466*t**2 - 8.77664e-7*t**3 + 74092/t
        self.ti_hcp = lambda t: -7836.23 + 138.868*t - 24.3312*t*torch.log(t) - 0.001712*t**2 + 5.5e-7*t**3 + 75000/t
        self.mg_hcp = lambda t: -8312.4 + 137.6*t - 24.4*t*torch.log(t) - 0.002*t**2 + 6.1e-7*t**3 + 78000/t
        self.li_bcc = lambda t: -7993.4 + 120.4*t - 24.1*t*torch.log(t) - 0.006*t**2 + 1.2e-6*t**3 + 70500/t
        self.ni_fcc = lambda t: -1234.5 + 130.1*t - 24.5*t*torch.log(t) - 0.003*t**2 + 3.2e-7*t**3 + 69000/t
        self.co_fcc = lambda t: -1345.6 + 132.2*t - 24.7*t*torch.log(t) - 0.004*t**2 + 4.1e-7*t**3 + 70000/t
        self.cu_fcc = lambda t: -1122.3 + 128.3*t - 24.2*t*torch.log(t) - 0.002*t**2 + 2.1e-7*t**3 + 68000/t
        self.zn_hcp = lambda t: -1567.8 + 135.4*t - 24.8*t*torch.log(t) - 0.005*t**2 + 5.2e-7*t**3 + 71000/t
        self.v_bcc = lambda t: -1890.1 + 140.5*t - 25.1*t*torch.log(t) - 0.006*t**2 + 6.3e-7*t**3 + 72000/t
        self.cr_bcc = lambda t: -1789.2 + 138.6*t - 24.9*t*torch.log(t) - 0.005*t**2 + 5.4e-7*t**3 + 71500/t
        self.mo_bcc = lambda t: -2100.3 + 145.7*t - 25.3*t*torch.log(t) - 0.007*t**2 + 7.5e-7*t**3 + 73000/t
        self.w_bcc = lambda t: -2300.4 + 150.8*t - 25.5*t*torch.log(t) - 0.008*t**2 + 8.6e-7*t**3 + 74000/t
        self.pt_fcc = lambda t: -2500.5 + 155.9*t - 25.7*t*torch.log(t) - 0.009*t**2 + 9.7e-7*t**3 + 75000/t
        self.au_fcc = lambda t: -2700.6 + 160.1*t - 25.9*t*torch.log(t) - 0.010*t**2 + 1.08e-6*t**3 + 76000/t

class SublatticeModel:
    def __init__(self, n_sites, species_per_site):
        self.n_sites = n_sites
        self.species = species_per_site
        self.g_ref = torch.zeros(1)
        self.g_id = torch.zeros(1)
        self.g_ex = torch.zeros(1)
    def compute_free_energy(self, site_fractions, t):
        self.g_ref = self._compute_reference(site_fractions, t)
        self.g_id = self._compute_ideal(site_fractions, t)
        self.g_ex = self._compute_excess(site_fractions, t)
        return self.g_ref + self.g_id + self.g_ex
    def _compute_reference(self, y, t):
        return torch.sum(y) * t
    def _compute_ideal(self, y, t):
        return Constants.R * t * torch.sum(y * torch.log(y + 1e-12))
    def _compute_excess(self, y, t):
        return torch.sum(y**2) * t

class RedlichKisterPolynomial:
    def __init__(self, L0, L1, L2, L3):
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
    def evaluate(self, xa, xb, t):
        term0 = self.L0[0] + self.L0[1]*t + self.L0[2]*t*torch.log(t)
        term1 = self.L1[0] + self.L1[1]*t + self.L1[2]*t*torch.log(t)
        term2 = self.L2[0] + self.L2[1]*t + self.L2[2]*t*torch.log(t)
        term3 = self.L3[0] + self.L3[1]*t + self.L3[2]*t*torch.log(t)
        diff = xa - xb
        return xa * xb * (term0 + term1*diff + term2*diff**2 + term3*diff**3)

class ActivityCoefficient:
    @staticmethod
    def margules_1_param(x1, x2, w, t):
        return torch.exp((w / (Constants.R * t)) * x2**2)
    @staticmethod
    def margules_2_param(x1, x2, w12, w21, t):
        return torch.exp((x2**2 * (w12 + 2*(w21 - w12)*x1)) / (Constants.R * t))
    @staticmethod
    def van_laar(x1, x2, a12, a21, t):
        denom = x1 + (a12/a21)*x2 + 1e-12
        return torch.exp(a12 / (1 + (a12*x1)/(a21*x2 + 1e-12))**2)
    @staticmethod
    def nrtl(x1, x2, g12, g21, alpha, t):
        tau12 = g12 / (Constants.R * t)
        tau21 = g21 / (Constants.R * t)
        G12 = torch.exp(-alpha * tau12)
        G21 = torch.exp(-alpha * tau21)
        term1 = tau21 * (G21 / (x1 + x2*G21 + 1e-12))**2
        term2 = (x1*tau12*G12) / (x2 + x1*G12 + 1e-12)**2
        return torch.exp(x2**2 * (term1 + term2))

class Electrochemistry:
    @staticmethod
    def nernst(e0, t, ox, red, n=1):
        return e0 - (Constants.R * t) / (n * Constants.F) * torch.log((red + 1e-12) / (ox + 1e-12))
    @staticmethod
    def nernst_activity(e0, t, ox, red, gamma_ox, gamma_red, n=1):
        a_ox = ox * gamma_ox
        a_red = red * gamma_red
        return e0 - (Constants.R * t) / (n * Constants.F) * torch.log((a_red + 1e-12) / (a_ox + 1e-12))
    @staticmethod
    def butler_volmer(eta, t, j0, alpha_a=0.5, alpha_c=0.5):
        f = Constants.F / (Constants.R * t)
        return j0 * (torch.exp(alpha_a * f * eta) - torch.exp(-alpha_c * f * eta))
    @staticmethod
    def butler_volmer_concentration(eta, t, k, c_s_surf, c_s_max, c_e, alpha_a=0.5, alpha_c=0.5):
        f = Constants.F / (Constants.R * t)
        j0 = Constants.F * k * (c_e**alpha_a) * ((c_s_max - c_s_surf)**alpha_a) * (c_s_surf**alpha_c)
        return j0 * (torch.exp(alpha_a * f * eta) - torch.exp(-alpha_c * f * eta))
    @staticmethod
    def tafel_anodic(eta, t, j0, alpha_a=0.5):
        f = Constants.F / (Constants.R * t)
        return j0 * torch.exp(alpha_a * f * eta)
    @staticmethod
    def tafel_cathodic(eta, t, j0, alpha_c=0.5):
        f = Constants.F / (Constants.R * t)
        return -j0 * torch.exp(-alpha_c * f * eta)
    @staticmethod
    def marcus_hush(eta, t, lambda_reorg, A):
        f = Constants.F / (Constants.R * t)
        exponent = -((lambda_reorg + Constants.F * eta)**2) / (4 * lambda_reorg * Constants.R * t)
        return A * torch.exp(exponent)
    @staticmethod
    def gerischer(eta, t, e_c, e_v, n_c, p_v):
        return n_c * torch.exp(-e_c / (Constants.R * t)) - p_v * torch.exp(-e_v / (Constants.R * t))

class Transport:
    @staticmethod
    def arrhenius(t, k0, ea):
        return k0 * torch.exp(-ea / (Constants.K_B * t))
    @staticmethod
    def arrhenius_concentration(t, c, k0, ea_base, alpha):
        ea = ea_base + alpha * c
        return k0 * torch.exp(-ea / (Constants.K_B * t))
    @staticmethod
    def vtf(t, a, b, t0):
        return a * torch.exp(-b / (t - t0 + 1e-12))
    @staticmethod
    def macinnes(c, d0, a, b):
        return d0 * (1 + a * c + b * c**2)
    @staticmethod
    def stokes_einstein(t, eta, r):
        return (Constants.R * t) / (6 * math.pi * eta * r * Constants.N_A)
    @staticmethod
    def nernst_einstein(t, d, z):
        return (d * z**2 * Constants.F**2) / (Constants.R * t)
    @staticmethod
    def kohlrausch(c, lambda_0, k):
        return lambda_0 - k * torch.sqrt(c)

class SolidDiffusion:
    @staticmethod
    def parabolic_profile(c_s, c_avg, r_p, d_s):
        return -d_s * (c_s - c_avg) / (r_p / 5.0)
    @staticmethod
    def polynomial_profile(c_s, c_avg, q_avg, r_p, d_s):
        return -d_s * (35.0/r_p * (c_s - c_avg) - 8.0*r_p*q_avg)
    @staticmethod
    def finite_difference_1d_cartesian(c, dx, d_s):
        flux = torch.zeros_like(c)
        flux[1:-1] = d_s * (c[2:] - 2*c[1:-1] + c[:-2]) / dx**2
        return flux
    @staticmethod
    def finite_difference_1d_cylindrical(c, r, dr, d_s):
        flux = torch.zeros_like(c)
        flux[1:-1] = d_s * ((c[2:] - 2*c[1:-1] + c[:-2]) / dr**2 + (c[2:] - c[:-2]) / (2 * r[1:-1] * dr + 1e-12))
        return flux
    @staticmethod
    def finite_difference_1d_spherical(c, r, dr, d_s):
        flux = torch.zeros_like(c)
        flux[1:-1] = d_s * ((c[2:] - 2*c[1:-1] + c[:-2]) / dr**2 + 2 * (c[2:] - c[:-2]) / (2 * r[1:-1] * dr + 1e-12))
        return flux

class LiquidDiffusion:
    @staticmethod
    def stefan_maxwell(c, grad_mu, d_ij):
        n = c.size(0)
        flux = torch.zeros_like(c)
        for i in range(n):
            for j in range(n):
                if i != j:
                    flux[i] += c[i] * c[j] * grad_mu[j] / (d_ij[i,j] + 1e-12)
        return flux
    @staticmethod
    def concentrated_solution(c, grad_c, t, d_eff, t_plus, grad_phi):
        f = Constants.F / (Constants.R * t)
        return -d_eff * grad_c + (t_plus / Constants.F) * c * grad_phi
    @staticmethod
    def dilute_solution(c, grad_c, z, u, grad_phi):
        return -u * Constants.R * t * grad_c - z * u * Constants.F * c * grad_phi

class Thermal:
    @staticmethod
    def lumped_mass(t_c, t_a, i_app, v_c, u_ocv, du_dt, m_cp, h_a):
        q_irr = i_app * (v_c - u_ocv)
        q_rev = i_app * t_c * du_dt
        q_conv = h_a * (t_c - t_a)
        return (q_irr + q_rev - q_conv) / m_cp
    @staticmethod
    def fourier_1d(t, dx, k, q_gen, rho_cp):
        dt = torch.zeros_like(t)
        dt[1:-1] = (k * (t[2:] - 2*t[1:-1] + t[:-2]) / dx**2 + q_gen[1:-1]) / rho_cp
        return dt
    @staticmethod
    def newton_cooling(t_surf, t_inf, h):
        return h * (t_surf - t_inf)
    @staticmethod
    def stefan_boltzmann_radiation(t_surf, t_inf, epsilon):
        return epsilon * Constants.sigma_sb * (t_surf**4 - t_inf**4)

class Degradation:
    @staticmethod
    def sei_diffusion_limited(l_sei, d_solv, m_sei, rho_sei):
        return (m_sei / rho_sei) * (d_solv / (l_sei + 1e-12))
    @staticmethod
    def sei_kinetic_limited(eta, t, k, m_sei, rho_sei, alpha=0.5):
        f = Constants.F / (Constants.R * t)
        return (m_sei / rho_sei) * k * torch.exp(-alpha * f * eta)
    @staticmethod
    def sei_mixed(eta, l_sei, t, k, d_solv, m_sei, rho_sei, alpha=0.5):
        r_diff = d_solv / (l_sei + 1e-12)
        f = Constants.F / (Constants.R * t)
        r_kin = k * torch.exp(-alpha * f * eta)
        r_tot = 1.0 / (1.0/(r_diff + 1e-12) + 1.0/(r_kin + 1e-12))
        return (m_sei / rho_sei) * r_tot
    @staticmethod
    def lithium_plating(eta, t, k_pl, alpha=0.5):
        f = Constants.F / (Constants.R * t)
        return torch.relu(-k_pl * torch.exp(-alpha * f * eta))
    @staticmethod
    def particle_cracking(stress, k_crack, m):
        return k_crack * torch.relu(stress)**m
    @staticmethod
    def active_material_loss(c_s, c_s_max, k_aml, n):
        return k_aml * (torch.abs(c_s / c_s_max))**n

class Mechanics:
    @staticmethod
    def hoop_stress(c, c_avg, omega, e_mod, nu):
        return (omega * e_mod) / (1 - nu) * (c_avg - c)
    @staticmethod
    def radial_stress(c, c_avg, omega, e_mod, nu):
        return (2 * omega * e_mod) / (1 - nu) * (c_avg - c)
    @staticmethod
    def hydrostatic_stress(sigma_r, sigma_theta, sigma_phi):
        return (sigma_r + sigma_theta + sigma_phi) / 3.0
    @staticmethod
    def von_mises_stress(sigma_1, sigma_2, sigma_3):
        return torch.sqrt(0.5 * ((sigma_1 - sigma_2)**2 + (sigma_2 - sigma_3)**2 + (sigma_3 - sigma_1)**2))

class MultiScaleP2D:
    def __init__(self, nx=50, nr=20):
        self.nx = nx
        self.nr = nr
        self.dx = 1.0 / nx
        self.dr = 1.0 / nr
        self.c_e = torch.ones(nx)
        self.phi_e = torch.zeros(nx)
        self.phi_s = torch.zeros(nx)
        self.c_s = torch.ones(nx, nr)
        self.t = torch.ones(nx) * 298.15
        self.j = torch.zeros(nx)
        self.eps_e = torch.ones(nx) * 0.3
        self.eps_s = torch.ones(nx) * 0.6
        self.a_s = torch.ones(nx) * 1e5
        self.bruggeman = 1.5
    def step_c_e(self, dt, d_e, t_plus):
        d_eff = d_e * self.eps_e**self.bruggeman
        flux = torch.zeros(self.nx + 1)
        flux[1:-1] = -d_eff[1:-1] * (self.c_e[1:] - self.c_e[:-1]) / self.dx
        self.c_e += dt * ((flux[:-1] - flux[1:]) / self.dx + (1 - t_plus) * self.a_s * self.j / Constants.F)
    def step_c_s(self, dt, d_s):
        flux = torch.zeros(self.nx, self.nr + 1)
        r = torch.linspace(0, 1, self.nr)
        for i in range(self.nx):
            flux[i, 1:-1] = -d_s[i] * (self.c_s[i, 1:] - self.c_s[i, :-1]) / self.dr
            self.c_s[i] += dt * ((flux[i, :-1] - flux[i, 1:]) / self.dr + 2 * flux[i, :-1] / (r + 1e-12))
            self.c_s[i, -1] -= dt * self.j[i] / Constants.F
    def solve_phi_e(self, kappa):
        kappa_eff = kappa * self.eps_e**self.bruggeman
        a = torch.zeros(self.nx, self.nx)
        b = torch.zeros(self.nx)
        for i in range(1, self.nx - 1):
            a[i, i-1] = kappa_eff[i-1] / self.dx**2
            a[i, i] = -(kappa_eff[i-1] + kappa_eff[i]) / self.dx**2
            a[i, i+1] = kappa_eff[i] / self.dx**2
            b[i] = -self.a_s[i] * self.j[i]
        a[0, 0] = 1.0
        b[0] = 0.0
        a[-1, -1] = 1.0
        b[-1] = 0.0
        self.phi_e = torch.linalg.solve(a, b)
    def solve_phi_s(self, sigma):
        sigma_eff = sigma * self.eps_s**self.bruggeman
        a = torch.zeros(self.nx, self.nx)
        b = torch.zeros(self.nx)
        for i in range(1, self.nx - 1):
            a[i, i-1] = sigma_eff[i-1] / self.dx**2
            a[i, i] = -(sigma_eff[i-1] + sigma_eff[i]) / self.dx**2
            a[i, i+1] = sigma_eff[i] / self.dx**2
            b[i] = self.a_s[i] * self.j[i]
        a[0, 0] = 1.0
        b[0] = 0.0
        a[-1, -1] = 1.0
        b[-1] = 1.0
        self.phi_s = torch.linalg.solve(a, b)
    def update_j(self, i0, u_ocv):
        eta = self.phi_s - self.phi_e - u_ocv
        self.j = Electrochemistry.butler_volmer(eta, self.t, i0)

class BatteryDigitalTwin(MultiScaleP2D):
    def __init__(self, nx=100, nr=50, np_y=10, np_z=10):
        super().__init__(nx, nr)
        self.np_y = np_y
        self.np_z = np_z
        self.c_e_3d = torch.ones(nx, np_y, np_z)
        self.phi_e_3d = torch.zeros(nx, np_y, np_z)
        self.phi_s_3d = torch.zeros(nx, np_y, np_z)
        self.t_3d = torch.ones(nx, np_y, np_z) * 298.15
        self.q_gen_3d = torch.zeros(nx, np_y, np_z)
    def update_3d_thermal(self, dt, k_x, k_y, k_z, rho_cp):
        dx, dy, dz = self.dx, 1.0/self.np_y, 1.0/self.np_z
        tx = (self.t_3d[2:,1:-1,1:-1] - 2*self.t_3d[1:-1,1:-1,1:-1] + self.t_3d[:-2,1:-1,1:-1]) / dx**2
        ty = (self.t_3d[1:-1,2:,1:-1] - 2*self.t_3d[1:-1,1:-1,1:-1] + self.t_3d[1:-1,:-2,1:-1]) / dy**2
        tz = (self.t_3d[1:-1,1:-1,2:] - 2*self.t_3d[1:-1,1:-1,1:-1] + self.t_3d[1:-1,1:-1,:-2]) / dz**2
        self.t_3d[1:-1,1:-1,1:-1] += dt * (k_x*tx + k_y*ty + k_z*tz + self.q_gen_3d[1:-1,1:-1,1:-1]) / rho_cp
    def map_1d_to_3d(self):
        self.c_e_3d = self.c_e.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
        self.phi_e_3d = self.phi_e.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
        self.phi_s_3d = self.phi_s.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
    def compute_3d_q_gen(self, u_ocv, du_dt):
        j_3d = self.j.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
        u_ocv_3d = u_ocv.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
        du_dt_3d = du_dt.view(-1, 1, 1).expand(-1, self.np_y, self.np_z)
        self.q_gen_3d = self.a_s.view(-1, 1, 1) * j_3d * (self.phi_s_3d - self.phi_e_3d - u_ocv_3d + self.t_3d * du_dt_3d)

class ExplicitRungeKutta:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.stages = len(b)
    def step(self, f, t, y, dt):
        k = []
        for i in range(self.stages):
            y_stage = y.clone()
            for j in range(i):
                y_stage += dt * self.a[i][j] * k[j]
            k.append(f(t + self.c[i]*dt, y_stage))
        y_next = y.clone()
        for i in range(self.stages):
            y_next += dt * self.b[i] * k[i]
        return y_next

class DOPRI5(ExplicitRungeKutta):
    def __init__(self):
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        self.b_alt = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        super().__init__(a, b, c)
    def step_adaptive(self, f, t, y, dt, tol=1e-6):
        k = []
        for i in range(self.stages):
            y_stage = y.clone()
            for j in range(i):
                y_stage += dt * self.a[i][j] * k[j]
            k.append(f(t + self.c[i]*dt, y_stage))
        y_next = y.clone()
        y_alt = y.clone()
        for i in range(self.stages):
            y_next += dt * self.b[i] * k[i]
            y_alt += dt * self.b_alt[i] * k[i]
        error = torch.norm(y_next - y_alt)
        if error == 0:
            dt_next = dt * 2.0
        else:
            dt_next = dt * 0.9 * (tol / error)**0.2
        return y_next, dt_next, error < tol

class ImplicitBDF:
    def __init__(self, order=2):
        self.order = order
    def step(self, f, t, y_history, dt, max_iter=100, tol=1e-6):
        if self.order == 1:
            y_next = y_history[0].clone()
            for _ in range(max_iter):
                res = y_next - y_history[0] - dt * f(t + dt, y_next)
                y_next -= res 
                if torch.norm(res) < tol:
                    break
            return y_next
        elif self.order == 2:
            y_next = y_history[0].clone()
            for _ in range(max_iter):
                res = y_next - (4/3)*y_history[0] + (1/3)*y_history[1] - (2/3)*dt * f(t + dt, y_next)
                y_next -= res 
                if torch.norm(res) < tol:
                    break
            return y_next
        elif self.order == 3:
            y_next = y_history[0].clone()
            for _ in range(max_iter):
                res = y_next - (18/11)*y_history[0] + (9/11)*y_history[1] - (2/11)*y_history[2] - (6/11)*dt * f(t + dt, y_next)
                y_next -= res 
                if torch.norm(res) < tol:
                    break
            return y_next

class AdjointSensitivity:
    def __init__(self, f, solver):
        self.f = f
        self.solver = solver
    def forward(self, y0, t_span):
        t = t_span[0]
        y = y0.clone()
        trajectory = [y.clone()]
        dt = 1e-3
        while t < t_span[1]:
            y = self.solver.step(self.f, t, y, dt)
            t += dt
            trajectory.append(y.clone())
        return torch.stack(trajectory)
    def backward(self, trajectory, t_span, loss_grad):
        t = t_span[1]
        a = loss_grad.clone()
        dt = -1e-3
        for y in reversed(trajectory):
            def adjoint_f(t, a_state):
                with torch.enable_grad():
                    y_var = y.clone().requires_grad_(True)
                    f_val = self.f(t, y_var)
                    vjp = torch.autograd.grad(f_val, y_var, a_state)[0]
                return -vjp
            a = self.solver.step(adjoint_f, t, a, dt)
            t += dt
        return a

class KineticsForgeEngine:
    def __init__(self):
        self.dt = BatteryDigitalTwin()
        self.dopri = DOPRI5()
        self.bdf = ImplicitBDF(order=2)
        self.adjoint = AdjointSensitivity(lambda t, y: y**2, self.dopri)
    def run_full_simulation(self, steps):
        for _ in range(steps):
            self.dt.step_c_e(1.0, torch.ones(100)*1e-10, 0.36)
            self.dt.step_c_s(1.0, torch.ones(100)*1e-14)
            self.dt.solve_phi_e(torch.ones(100)*1.0)
            self.dt.solve_phi_s(torch.ones(100)*100.0)
            self.dt.update_j(torch.ones(100)*1e-5, torch.ones(100)*3.7)
            self.dt.map_1d_to_3d()
            self.dt.compute_3d_q_gen(torch.ones(100)*3.7, torch.ones(100)*-1e-4)
            self.dt.update_3d_thermal(1.0, 1.0, 1.0, 1.0, 2.5e6)

class MicrostructureEvolution:
    @staticmethod
    def phase_field_cahn_hilliard(c, dt, dx, m, kappa, df_dc):
        laplacian_c = torch.zeros_like(c)
        laplacian_c[1:-1] = (c[2:] - 2*c[1:-1] + c[:-2]) / dx**2
        mu = df_dc - kappa * laplacian_c
        laplacian_mu = torch.zeros_like(c)
        laplacian_mu[1:-1] = (mu[2:] - 2*mu[1:-1] + mu[:-2]) / dx**2
        return c + dt * m * laplacian_mu
    @staticmethod
    def phase_field_allen_cahn(eta, dt, dx, l, kappa, df_deta):
        laplacian_eta = torch.zeros_like(eta)
        laplacian_eta[1:-1] = (eta[2:] - 2*eta[1:-1] + eta[:-2]) / dx**2
        return eta - dt * l * (df_deta - kappa * laplacian_eta)
    @staticmethod
    def grain_growth_multiphase(etas, dt, dx, l, kappas, w):
        n_phases = len(etas)
        next_etas = []
        for i in range(n_phases):
            laplacian = torch.zeros_like(etas[i])
            laplacian[1:-1] = (etas[i][2:] - 2*etas[i][1:-1] + etas[i][:-2]) / dx**2
            penalty = sum([etas[j]**2 for j in range(n_phases) if j != i])
            df_deta = w * etas[i] * penalty
            next_etas.append(etas[i] - dt * l[i] * (df_deta - kappas[i] * laplacian))
        return next_etas

class ElectrochemicalImpedance:
    @staticmethod
    def randles_circuit(omega, r_s, r_ct, c_dl, sigma):
        z_w = sigma / torch.sqrt(omega) * (1 - 1j)
        z_f = r_ct + z_w
        z_rc = 1.0 / (1j * omega * c_dl + 1.0 / z_f)
        return r_s + z_rc
    @staticmethod
    def transmission_line(omega, r_ion, r_elec, z_interfacial, l):
        lambda_t = torch.sqrt(z_interfacial / (r_ion + r_elec))
        gamma = l / lambda_t
        term1 = (r_ion * r_elec) / (r_ion + r_elec) * l
        term2 = 2 * lambda_t * r_ion * r_elec / (r_ion + r_elec)**2 / torch.sinh(gamma)
        term3 = lambda_t * (r_ion**2 + r_elec**2) / (r_ion + r_elec)**2 / torch.tanh(gamma)
        return term1 + term2 + term3
    @staticmethod
    def constant_phase_element(omega, q, n):
        return 1.0 / (q * (1j * omega)**n)
    @staticmethod
    def warburg_finite_length(omega, r_w, t_w):
        return r_w * torch.tanh(torch.sqrt(1j * omega * t_w)) / torch.sqrt(1j * omega * t_w)
    @staticmethod
    def warburg_finite_space(omega, r_w, t_w):
        return r_w * 1.0 / torch.tanh(torch.sqrt(1j * omega * t_w)) / torch.sqrt(1j * omega * t_w)

class PorousElectrode:
    @staticmethod
    def tortuosity_bruggeman(epsilon, alpha=1.5):
        return epsilon**(-alpha)
    @staticmethod
    def tortuosity_muggianu(epsilon, p):
        return (1 - p) / (epsilon - p)
    @staticmethod
    def specific_area_spheres(epsilon, r_p):
        return 3 * (1 - epsilon) / r_p
    @staticmethod
    def effective_conductivity(sigma, epsilon, tortuosity):
        return sigma * epsilon / tortuosity
    @staticmethod
    def macro_homogeneous_reaction(a_s, i_n):
        return a_s * i_n

class MaterialProperties:
    @staticmethod
    def lco_ocv(theta):
        return 4.19829 + 0.0565661*theta - 0.0158451*theta**2 + 0.000964741*theta**3 - 0.0000213192*theta**4 - 0.0000001005*theta**5
    @staticmethod
    def nmc_ocv(theta):
        return 4.3452 - 1.6518*theta + 1.6225*theta**2 - 2.0843*theta**3 + 1.3632*theta**4 - 0.334*theta**5
    @staticmethod
    def lfp_ocv(theta):
        return 3.4323 - 0.8428*torch.exp(-80.24*theta) + 3.2474*torch.exp(-20.26*(1-theta)) - 0.22*theta**2 + 0.17*theta**3
    @staticmethod
    def graphite_ocv(theta):
        return 0.7222 + 0.1387*theta + 0.029*theta**0.5 - 0.0172/theta + 0.0019/theta**1.5 + 0.2808*torch.exp(0.9-15*theta) - 0.7984*torch.exp(0.448-theta)
    @staticmethod
    def silicon_ocv(theta):
        return 0.8 + 0.1*theta - 0.05*theta**2 + 0.01*theta**3
    @staticmethod
    def lto_ocv(theta):
        return 1.55 + 0.01*theta - 0.005*theta**2 + 0.001*theta**3
    @staticmethod
    def electrolyte_conductivity(c, t):
        return 1e-4 * c * (-10.5 + 0.668*1e-3*c + 0.494*1e-6*c**2 + 0.074*t - 1.78e-5*c*t - 8.86e-10*c**2*t - 6.96e-5*t**2 + 2.8e-8*c*t**2)
    @staticmethod
    def electrolyte_diffusivity(c, t):
        return 1e-4 * 10**(-4.43 - 54/(t-229-5e-3*c) - 0.22e-3*c)

def initialize_forge():
    engine = KineticsForgeEngine()
    engine.run_full_simulation(10)
    return engine

if __name__ == "__main__":
    initialize_forge()
