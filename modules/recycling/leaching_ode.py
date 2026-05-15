import torch
import torch.nn as nn
from core.physics_constraints import KineticConstraints

class ShrinkingCoreDiffusion(nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.D0 = nn.Parameter(torch.randn(n_species))
        self.Ea = nn.Parameter(torch.abs(torch.randn(n_species)))
    def forward(self, t, alpha, temp, c_acid, r0):
        D_eff = self.D0 * torch.exp(-self.Ea / (8.314 * temp))
        da_dt = (3 * D_eff * c_acid) / (r0**2 * 5000.0 * (1 - alpha)**(1/3) + 1e-6)
        return torch.relu(da_dt)

class ShrinkingCoreReaction(nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.k0 = nn.Parameter(torch.randn(n_species))
        self.Ea = nn.Parameter(torch.abs(torch.randn(n_species)))
    def forward(self, t, alpha, temp, c_acid, r0):
        k_r = self.k0 * torch.exp(-self.Ea / (8.314 * temp))
        da_dt = (3 * k_r * c_acid) / (r0 * 5000.0) * (1 - alpha)**(2/3)
        return torch.relu(da_dt)

class AvramiNucleation(nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.k_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, n_species)
        )
        self.n_exponent = nn.Parameter(torch.ones(n_species) * 2.0)
    def forward(self, t, temp, ph):
        inputs = torch.cat([temp.unsqueeze(-1), ph.unsqueeze(-1)], dim=-1)
        k_A = torch.exp(self.k_mlp(inputs))
        n = torch.clamp(self.n_exponent, 1.0, 4.0)
        alpha = 1 - torch.exp(-k_A * t**n)
        da_dt = k_A * n * t**(n-1) * torch.exp(-k_A * t**n)
        return alpha, da_dt

class BlendNetwork(nn.Module):
    def __init__(self, n_species):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, n_species),
            nn.Sigmoid()
        )
    def forward(self, temp, ph, particle_size):
        inputs = torch.cat([temp.unsqueeze(-1), ph.unsqueeze(-1), particle_size.unsqueeze(-1)], dim=-1)
        return self.net(inputs)

class ParticleSizeDistribution:
    def __init__(self, num_bins, min_r, max_r):
        self.r_bins = torch.linspace(min_r, max_r, num_bins)
        self.w_bins = self._log_normal(self.r_bins, (min_r+max_r)/2, (max_r-min_r)/4)
        self.w_bins /= torch.sum(self.w_bins)
    def _log_normal(self, x, mu, sigma):
        return 1.0 / (x * sigma * torch.sqrt(torch.tensor(2*torch.pi))) * torch.exp(-(torch.log(x) - mu)**2 / (2*sigma**2))

class MultiMechanismLeaching(nn.Module):
    def __init__(self, n_species, psd):
        super().__init__()
        self.sc_diff = ShrinkingCoreDiffusion(n_species)
        self.sc_rxn = ShrinkingCoreReaction(n_species)
        self.avrami = AvramiNucleation(n_species)
        self.blend = BlendNetwork(n_species)
        self.psd = psd
        self.n_species = n_species
    def forward(self, t, alpha, temp, ph, c_acid):
        da_dt_total = torch.zeros_like(alpha)
        for i, r0 in enumerate(self.psd.r_bins):
            w = self.psd.w_bins[i]
            alpha_bin = alpha[i]
            da_diff = self.sc_diff(t, alpha_bin, temp, c_acid, r0)
            da_rxn = self.sc_rxn(t, alpha_bin, temp, c_acid, r0)
            r_diff = 1.0 / (da_diff + 1e-12)
            r_rxn = 1.0 / (da_rxn + 1e-12)
            da_sc = 1.0 / (r_diff + r_rxn)
            _, da_avr = self.avrami(t, temp, ph)
            gamma = self.blend(temp, ph, r0.expand_as(temp))
            da_bin = gamma * da_sc + (1 - gamma) * da_avr
            da_dt_total[i] = da_bin * w
        return da_dt_total.sum(dim=0)

class SpeciesMassBalance:
    def __init__(self, reactor_volume, solid_mass, species_fractions):
        self.V = reactor_volume
        self.m0 = solid_mass
        self.x0 = species_fractions
    def compute_liquid_concentration(self, alpha):
        return (self.m0 * self.x0 * alpha) / self.V

class ThermalEnergyBalance:
    def __init__(self, m_cp_reactor, hA_loss, delta_H_rxn):
        self.m_cp = m_cp_reactor
        self.hA = hA_loss
        self.dH = delta_H_rxn
    def compute_dt_dt(self, T, T_amb, da_dt, m0, x0):
        q_gen = torch.sum(da_dt * m0 * x0 * self.dH)
        q_loss = self.hA * (T - T_amb)
        return (q_gen - q_loss) / self.m_cp

class CoupledLeachingSolver:
    def __init__(self, leaching_model, mass_balance, energy_balance):
        self.model = leaching_model
        self.mb = mass_balance
        self.eb = energy_balance
    def step(self, state, dt):
        t, alpha, temp, ph, c_acid, t_amb = state
        da_dt = self.model(t, alpha, temp, ph, c_acid)
        dt_dt = self.eb.compute_dt_dt(temp, t_amb, da_dt, self.mb.m0, self.mb.x0)
        c_liq = self.mb.compute_liquid_concentration(alpha)
        dph_dt = -0.1 * torch.sum(da_dt)
        dc_acid_dt = -0.5 * torch.sum(da_dt)
        alpha_next = alpha + dt * da_dt
        temp_next = temp + dt * dt_dt
        ph_next = ph + dt * dph_dt
        c_acid_next = c_acid + dt * dc_acid_dt
        return t + dt, alpha_next, temp_next, ph_next, c_acid_next, t_amb

class StochasticLeachingSimulator:
    def __init__(self, solver, noise_std):
        self.solver = solver
        self.std = noise_std
    def run(self, state0, steps, dt):
        traj = [state0]
        curr = state0
        for _ in range(steps):
            next_state = self.solver.step(curr, dt)
            noisy_alpha = next_state[1] + torch.randn_like(next_state[1]) * self.std
            noisy_alpha = torch.clamp(noisy_alpha, 0.0, 1.0)
            curr = list(next_state)
            curr[1] = noisy_alpha
            traj.append(curr)
        return traj

class ContinuousStirredTankReactor:
    def __init__(self, v, q_in, q_out):
        self.v = v
        self.q_in = q_in
        self.q_out = q_out
    def concentration_deriv(self, c, c_in, r_gen):
        return (self.q_in * c_in - self.q_out * c) / self.v + r_gen
    def volume_deriv(self):
        return self.q_in - self.q_out

class PlugFlowReactor:
    def __init__(self, u_z, d_z):
        self.u = u_z
        self.d = d_z
    def spatial_deriv(self, c, dx, r_gen):
        dc_dx = (c[2:] - c[:-2]) / (2*dx)
        d2c_dx2 = (c[2:] - 2*c[1:-1] + c[:-2]) / dx**2
        return self.d * d2c_dx2 - self.u * dc_dx + r_gen[1:-1]
