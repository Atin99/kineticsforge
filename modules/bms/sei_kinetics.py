import torch
import torch.nn as nn
import numpy as np
import math

class SEIGrowthModel(nn.Module):
    def __init__(self, n_cells=8):
        super().__init__()
        self.n_cells = n_cells
        self.k0_sei = nn.Parameter(torch.ones(n_cells) * 1.5e-6)
        self.Ea_sei = nn.Parameter(torch.ones(n_cells) * 0.4)
        self.D_sei = nn.Parameter(torch.ones(n_cells) * 1e-14)
        self.R = 8.617e-5

    def growth_rate(self, L_sei, T):
        k_sei = torch.abs(self.k0_sei) * torch.exp(-torch.abs(self.Ea_sei) / (self.R * T + 1e-10))
        diffusion_limited = k_sei / (2.0 * torch.clamp(L_sei, min=1e-10))
        return diffusion_limited

    def forward(self, L_sei, T, dt=1.0):
        dLdt = self.growth_rate(L_sei, T)
        L_new = L_sei + dLdt * dt
        return L_new, dLdt


class DendritePrecursorModel(nn.Module):
    def __init__(self, n_cells=8):
        super().__init__()
        self.n_cells = n_cells
        self.k_dendrite = nn.Parameter(torch.ones(n_cells) * 0.001)
        self.J_max = nn.Parameter(torch.ones(n_cells) * 0.02)
        self.alpha_bv = nn.Parameter(torch.ones(n_cells) * 0.5)
        self.i0 = nn.Parameter(torch.ones(n_cells) * 1e-3)
        self.F = 96485.0
        self.R_gas = 8.314

    def butler_volmer_current(self, eta, T):
        f = self.F / (self.R_gas * T + 1e-10)
        alpha = torch.clamp(torch.abs(self.alpha_bv), 0.1, 0.9)
        i0 = torch.abs(self.i0)
        J = i0 * (torch.exp(alpha * f * eta) - torch.exp(-(1.0 - alpha) * f * eta))
        return J

    def plating_rate(self, J_Li, T):
        excess = torch.relu(torch.abs(J_Li) - torch.abs(self.J_max))
        k = torch.abs(self.k_dendrite)
        T_factor = torch.exp(-0.3 / (self.R_gas / self.F * T + 1e-10))
        return k * excess * T_factor

    def forward(self, P_dendrite, eta, T, dt=1.0):
        J_Li = self.butler_volmer_current(eta, T)
        dPdt = self.plating_rate(J_Li, T)
        P_new = P_dendrite + dPdt * dt
        return P_new, dPdt, J_Li


class InternalResistanceModel(nn.Module):
    def __init__(self, n_cells=8):
        super().__init__()
        self.R0 = nn.Parameter(torch.ones(n_cells) * 0.02)
        self.alpha_sei = nn.Parameter(torch.ones(n_cells) * 0.5)
        self.beta_dendrite = nn.Parameter(torch.ones(n_cells) * 0.3)
        self.gamma_temp = nn.Parameter(torch.ones(n_cells) * 0.001)
        self.Ea_R = nn.Parameter(torch.ones(n_cells) * 0.3)

    def forward(self, L_sei, P_dendrite, T, SOC=None, eis_features=None):
        R_base = torch.abs(self.R0)
        R_sei = torch.abs(self.alpha_sei) * L_sei
        R_dendrite = torch.abs(self.beta_dendrite) * P_dendrite
        T_ref = 298.0
        R_temp = R_base * torch.exp(torch.abs(self.Ea_R) * (1.0/T - 1.0/T_ref) / 8.617e-5)
        R_total = R_temp + R_sei + R_dendrite
        if eis_features is not None:
            if isinstance(eis_features, dict):
                R_ohm = eis_features.get("R_ohm", 0.0)
                R_ct = eis_features.get("R_ct", 0.0)
                R_eis_sei = eis_features.get("R_sei", 0.0)
                sigma_w = eis_features.get("sigma_warburg", 0.0)
                R_ohm = torch.as_tensor(R_ohm, dtype=R_total.dtype, device=R_total.device)
                R_ct = torch.as_tensor(R_ct, dtype=R_total.dtype, device=R_total.device)
                R_eis_sei = torch.as_tensor(R_eis_sei, dtype=R_total.dtype, device=R_total.device)
                sigma_w = torch.as_tensor(sigma_w, dtype=R_total.dtype, device=R_total.device)
            else:
                arr = torch.as_tensor(eis_features, dtype=R_total.dtype, device=R_total.device)
                if arr.dim() == 1 and arr.numel() >= 4:
                    R_ohm, R_ct, R_eis_sei, sigma_w = arr[0], arr[1], arr[2], arr[3]
                else:
                    R_ohm, R_ct, R_eis_sei, sigma_w = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            eis_delta = torch.clamp(R_ohm, min=0.0, max=5.0) + 0.35 * torch.clamp(R_ct, min=0.0, max=50.0) + 0.50 * torch.clamp(R_eis_sei, min=0.0, max=50.0) + 0.02 * torch.clamp(sigma_w, min=0.0, max=100.0)
            R_total = 0.75 * R_total + 0.25 * eis_delta
        if SOC is not None:
            soc_factor = 1.0 + 0.5 * torch.exp(-10.0 * SOC) + 0.3 * torch.exp(-10.0 * (1.0 - SOC))
            R_total = R_total * soc_factor
        return R_total


class ThermalModel(nn.Module):
    def __init__(self, n_cells=8):
        super().__init__()
        self.Cp = nn.Parameter(torch.ones(n_cells) * 1000.0)
        self.mass = nn.Parameter(torch.ones(n_cells) * 0.045)
        self.h_conv = nn.Parameter(torch.ones(n_cells) * 10.0)
        self.A_surface = nn.Parameter(torch.ones(n_cells) * 0.005)
        self.k_thermal = nn.Parameter(torch.ones(n_cells, n_cells) * 0.5)

    def heat_generation(self, I, R_int, T, V_ocv, V_terminal):
        Q_joule = I**2 * R_int
        Q_reversible = I * T * 0.0001
        return Q_joule + torch.abs(Q_reversible)

    def thermal_coupling(self, T_cells, adjacency):
        n = T_cells.shape[-1]
        k = torch.abs(self.k_thermal[:n, :n])
        k_sym = (k + k.T) / 2.0
        T_diff = T_cells.unsqueeze(-1) - T_cells.unsqueeze(-2)
        Q_coupling = torch.sum(k_sym * adjacency * T_diff, dim=-1)
        return Q_coupling

    def forward(self, T_cells, I, R_int, T_ambient, adjacency, dt=1.0):
        Q_gen = self.heat_generation(I, R_int, T_cells, None, None)
        Q_conv = torch.abs(self.h_conv) * torch.abs(self.A_surface) * (T_cells - T_ambient)
        Q_coupling = self.thermal_coupling(T_cells, adjacency)
        Cp_m = torch.abs(self.Cp) * torch.abs(self.mass)
        dTdt = (Q_gen - Q_conv - Q_coupling) / (Cp_m + 1e-10)
        T_new = T_cells + dTdt * dt
        return T_new, dTdt


class CoupledSEIModel(nn.Module):
    def __init__(self, n_cells=8, state_dim=4):
        super().__init__()
        self.n_cells = n_cells
        self.state_dim = state_dim
        self.sei = SEIGrowthModel(n_cells)
        self.dendrite = DendritePrecursorModel(n_cells)
        self.resistance = InternalResistanceModel(n_cells)
        self.thermal = ThermalModel(n_cells)
        self.risk_weights = nn.Parameter(torch.tensor([2.0, 1.4, 1.6, 1.1]))
        self.risk_bias = nn.Parameter(torch.tensor(-2.2))

        self.state = None
        self.adjacency = None
        self._build_default_adjacency()

    def _build_default_adjacency(self):
        adj = torch.zeros(self.n_cells, self.n_cells)
        for i in range(self.n_cells):
            if i % 4 < 3:
                adj[i, i+1] = 1.0
                adj[i+1, i] = 1.0
            if i < 4 and i + 4 < self.n_cells:
                adj[i, i+4] = 0.5
                adj[i+4, i] = 0.5
        self.adjacency = adj

    def init_state(self, T_ambient=308.0):
        self.state = torch.zeros(self.n_cells, self.state_dim)
        self.state[:, 0] = 1e-9
        self.state[:, 1] = 0.0
        self.state[:, 2] = T_ambient
        self.state[:, 3] = 1.0
        return self.state

    def step(self, I_cells, V_cells, T_ambient, dt=1.0, eis_features=None):
        L_sei = self.state[:, 0]
        P_dendrite = self.state[:, 1]
        T_cells = self.state[:, 2]
        SOC = self.state[:, 3]

        L_new, dL = self.sei(L_sei, T_cells, dt)
        eta = V_cells - 3.7
        P_new, dP, J_Li = self.dendrite(P_dendrite, eta, T_cells, dt)
        R_int = self.resistance(L_new, P_new, T_cells, SOC, eis_features=eis_features)
        adj = self.adjacency.to(T_cells.device)
        T_new, dT = self.thermal(T_cells, I_cells, R_int, T_ambient, adj, dt)
        dSOC = -I_cells / (3600.0 * 2.5 + 1e-10)
        SOC_new = torch.clamp(SOC + dSOC * dt, 0.0, 1.0)

        L_new = torch.clamp(torch.nan_to_num(L_new, nan=1e-6, posinf=5e-2, neginf=1e-10), 1e-10, 5e-2)
        P_new = torch.clamp(torch.nan_to_num(P_new, nan=0.0, posinf=2.0, neginf=0.0), 0.0, 2.0)
        T_new = torch.clamp(torch.nan_to_num(T_new, nan=350.0, posinf=620.0, neginf=260.0), 260.0, 620.0)
        SOC_new = torch.clamp(torch.nan_to_num(SOC_new, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        self.state = torch.stack([L_new, P_new, T_new, SOC_new], dim=-1)
        return self.state, R_int

    def compute_risk_score(self):
        L_sei = self.state[:, 0]
        P_dendrite = self.state[:, 1]
        T_cells = self.state[:, 2]
        SOC = self.state[:, 3]

        R_int = self.resistance(L_sei, P_dendrite, T_cells, SOC)
        delta_R = R_int / (torch.abs(self.resistance.R0) + 1e-10) - 1.0
        delta_T = (T_cells - 308.0) / 50.0
        P_norm = P_dendrite / 0.02
        L_norm = L_sei / 0.004

        features = torch.stack([delta_R, delta_T, P_norm, L_norm], dim=-1)
        w = torch.abs(self.risk_weights)
        logit = torch.sum(features * w, dim=-1) + self.risk_bias
        risk = torch.sigmoid(logit)
        return risk

    def get_cell_diagnostics(self):
        risk = self.compute_risk_score()
        diagnostics = []
        for i in range(self.n_cells):
            diagnostics.append({
                'cell_id': i,
                'L_sei': self.state[i, 0].item(),
                'P_dendrite': self.state[i, 1].item(),
                'T': self.state[i, 2].item(),
                'SOC': self.state[i, 3].item(),
                'risk': risk[i].item(),
                'status': 'CRITICAL' if risk[i].item() > 0.75 else 'WARNING' if risk[i].item() > 0.5 else 'NORMAL'
            })
        return diagnostics
