import torch
import torch.nn as nn
import numpy as np
import math
from core.phase_transition import NaIonDegradationPhysics

class SINDyBasisLibrary(nn.Module):
    def __init__(self, state_dim=3, poly_degree=3, include_trig=True, include_exp=True):
        super().__init__()
        self.state_dim = state_dim
        self.poly_degree = poly_degree
        self.include_trig = include_trig
        self.include_exp = include_exp
        self.n_features = self._count_features()

    def _count_features(self):
        count = 1
        count += self.state_dim
        if self.poly_degree >= 2:
            count += self.state_dim * (self.state_dim + 1) // 2
        if self.poly_degree >= 3:
            for i in range(self.state_dim):
                for j in range(i, self.state_dim):
                    for k in range(j, self.state_dim):
                        count += 1
        if self.include_trig:
            count += self.state_dim * 2
        if self.include_exp:
            count += self.state_dim
        return count

    def forward(self, x):
        batch_shape = x.shape[:-1]
        features = [torch.ones(*batch_shape, 1, device=x.device)]
        for i in range(self.state_dim):
            features.append(x[..., i:i+1])
        if self.poly_degree >= 2:
            for i in range(self.state_dim):
                for j in range(i, self.state_dim):
                    features.append(x[..., i:i+1] * x[..., j:j+1])
        if self.poly_degree >= 3:
            for i in range(self.state_dim):
                for j in range(i, self.state_dim):
                    for k in range(j, self.state_dim):
                        features.append(x[..., i:i+1] * x[..., j:j+1] * x[..., k:k+1])
        if self.include_trig:
            for i in range(self.state_dim):
                features.append(torch.sin(x[..., i:i+1]))
                features.append(torch.cos(x[..., i:i+1]))
        if self.include_exp:
            for i in range(self.state_dim):
                features.append(torch.exp(-torch.clamp(x[..., i:i+1], -10, 10)))
        return torch.cat(features, dim=-1)


class SINDySparseCoefficients(nn.Module):
    def __init__(self, n_basis, output_dim, sparsity_threshold=0.01):
        super().__init__()
        self.Xi = nn.Parameter(torch.randn(n_basis, output_dim) * 0.01)
        self.threshold = sparsity_threshold
        self.mask = None

    def forward(self, theta):
        if self.mask is not None:
            Xi_sparse = self.Xi * self.mask
        else:
            Xi_sparse = self.Xi
        return torch.matmul(theta, Xi_sparse)

    def apply_threshold(self):
        with torch.no_grad():
            self.mask = (torch.abs(self.Xi) > self.threshold).float()
            n_active = self.mask.sum().item()
            n_total = self.mask.numel()
        return n_active, n_total

    def get_active_terms(self):
        if self.mask is None:
            self.apply_threshold()
        active = []
        for i in range(self.Xi.shape[0]):
            for j in range(self.Xi.shape[1]):
                if self.mask[i, j] > 0:
                    active.append((i, j, self.Xi[i, j].item()))
        return active

    def sparsity_loss(self):
        return torch.mean(torch.abs(self.Xi))


class CompositionConditioner(nn.Module):
    def __init__(self, comp_dim=5, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(comp_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.scale_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softplus()
        )
        self.shift_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, comp):
        z = self.embed(comp)
        scale = self.scale_net(z)
        shift = self.shift_net(z)
        return z, scale, shift


class NeuralResidualField(nn.Module):
    def __init__(self, state_dim=3, comp_embed_dim=32, hidden_dim=128, n_layers=4):
        super().__init__()
        input_dim = state_dim + comp_embed_dim + 1
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [state_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.05))
        self.net = nn.Sequential(*layers)
        self._init_small()

    def _init_small(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, z_comp, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(state.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([state, z_comp, t], dim=-1)
        return self.net(x)


class OCVModel(nn.Module):
    def __init__(self, comp_embed_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + comp_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_stoich, z_comp):
        inp = torch.cat([x_stoich, z_comp], dim=-1)
        return self.net(inp).squeeze(-1)

    def dUdx(self, x_stoich, z_comp):
        x_req = x_stoich.clone().requires_grad_(True)
        U = self.forward(x_req, z_comp)
        dU = torch.autograd.grad(U.sum(), x_req, create_graph=True)[0]
        return dU


class ArrheniusKinetics(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_k0 = nn.Parameter(torch.tensor(math.log(1e-4)))
        self.Ea = nn.Parameter(torch.tensor(0.6))
        self.R = 8.617e-5

    def forward(self, T):
        k0 = torch.exp(self.log_k0)
        return k0 * torch.exp(-self.Ea / (self.R * T))

    def get_params(self):
        return {
            'k0': torch.exp(self.log_k0).item(),
            'Ea': self.Ea.item()
        }


class DegradationVectorField(nn.Module):
    def __init__(self, comp_dim=5, comp_embed_dim=32, hidden_dim=128,
                 n_residual_layers=4, sindy_poly_degree=3):
        super().__init__()
        self.state_dim = 3

        self.conditioner = CompositionConditioner(comp_dim, comp_embed_dim, hidden_dim=64)
        self.sindy_basis = SINDyBasisLibrary(self.state_dim, sindy_poly_degree)
        self.sindy_coeffs = SINDySparseCoefficients(self.sindy_basis.n_features, self.state_dim)
        self.neural_residual = NeuralResidualField(self.state_dim, comp_embed_dim, hidden_dim, n_residual_layers)
        self.arrhenius = ArrheniusKinetics()
        self.ocv = OCVModel(comp_embed_dim)
        self.na_ion_physics = NaIonDegradationPhysics()

        self.alpha_sindy = nn.Parameter(torch.tensor(0.7))
        self.alpha_neural = nn.Parameter(torch.tensor(0.3))
        self._last_physics_diagnostics = {}

        self._current_comp = None
        self._current_z = None
        self._current_scale = None
        self._current_shift = None
        self._current_T = torch.tensor(318.0)

    def set_composition(self, comp_vec, T=318.0):
        self._current_comp = comp_vec
        z, scale, shift = self.conditioner(comp_vec)
        self._current_z = z
        self._current_scale = scale
        self._current_shift = shift
        self._current_T = torch.tensor(T, device=comp_vec.device) if not isinstance(T, torch.Tensor) else T

    def forward(self, t, state):
        Q = state[..., 0:1]
        V = state[..., 1:2]
        x = state[..., 2:3]

        k_fade = self.arrhenius(self._current_T)

        z_comp = self._current_z
        if z_comp.dim() == 1:
            z_comp = z_comp.unsqueeze(0).expand(state.shape[0], -1)
        elif z_comp.shape[0] != state.shape[0]:
            z_comp = z_comp.expand(state.shape[0], -1)

        t_input = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=state.device)
        f_neural = self.neural_residual(state, z_comp, t_input)
        physics_state, diagnostics = self.na_ion_physics(state, self._current_comp, self._current_T, t_input, k_fade)
        self._last_physics_diagnostics = diagnostics
        alpha_n = torch.sigmoid(self.alpha_neural)
        dstate_known_residual = physics_state + alpha_n * f_neural
        dxdt = dstate_known_residual[..., 2:3]
        dUdx = self.ocv.dUdx(x, z_comp)
        I_app = torch.ones_like(Q) * 0.001
        R_int = 0.05 + 0.001 * (1.0 - Q / (Q.detach().max() + 1e-8))
        dVdt_ocv = dUdx * dxdt - I_app * R_int
        dstate = torch.cat(
            [
                dstate_known_residual[..., 0:1],
                dstate_known_residual[..., 1:2] + 0.2 * dVdt_ocv,
                dxdt,
            ],
            dim=-1,
        )

        if self._current_scale is not None:
            scale = self._current_scale
            shift = self._current_shift
            if scale.dim() == 1:
                scale = scale.unsqueeze(0).expand(state.shape[0], -1)
                shift = shift.unsqueeze(0).expand(state.shape[0], -1)
            dstate = scale * dstate + shift * 0.01

        return dstate

    def sindy_discovered_equation(self):
        self.sindy_coeffs.apply_threshold()
        active = self.sindy_coeffs.get_active_terms()
        basis_names = self._get_basis_names()
        state_names = ['Q', 'V', 'x']
        equations = {s: [] for s in state_names}
        for basis_idx, state_idx, coeff in active:
            if basis_idx < len(basis_names):
                equations[state_names[state_idx]].append(
                    f"{coeff:+.4f}*{basis_names[basis_idx]}"
                )
        result = {}
        for s in state_names:
            if equations[s]:
                result[f'd{s}/dt'] = ' '.join(equations[s])
            else:
                result[f'd{s}/dt'] = 'neural_only'
        return result

    def _get_basis_names(self):
        names = ['1']
        state_vars = ['Q', 'V', 'x']
        for s in state_vars:
            names.append(s)
        if self.sindy_basis.poly_degree >= 2:
            for i, si in enumerate(state_vars):
                for j in range(i, len(state_vars)):
                    names.append(f"{si}*{state_vars[j]}")
        if self.sindy_basis.poly_degree >= 3:
            for i, si in enumerate(state_vars):
                for j in range(i, len(state_vars)):
                    for k in range(j, len(state_vars)):
                        names.append(f"{si}*{state_vars[j]}*{state_vars[k]}")
        if self.sindy_basis.include_trig:
            for s in state_vars:
                names.append(f"sin({s})")
                names.append(f"cos({s})")
        if self.sindy_basis.include_exp:
            for s in state_vars:
                names.append(f"exp(-{s})")
        return names

    def get_physics_summary(self):
        physics_diag = {}
        for key, value in self._last_physics_diagnostics.items():
            if isinstance(value, torch.Tensor):
                physics_diag[key] = float(value.detach().mean().cpu())
        return {
            'arrhenius': self.arrhenius.get_params(),
            'sindy_equations': self.sindy_discovered_equation(),
            'sindy_status': 'standalone_analysis_only_not_used_in_live_ode_rhs',
            'na_ion_physics': physics_diag,
            'neural_alpha': torch.sigmoid(self.alpha_neural).item(),
            'n_sindy_active': len(self.sindy_coeffs.get_active_terms()),
            'sparsity_loss': self.sindy_coeffs.sparsity_loss().item()
        }


class CathodeDegradationSimulator:
    def __init__(self, model, solver='euler', dt=0.01, max_cycles=500):
        self.model = model
        self.solver = solver
        self.dt = dt
        self.max_cycles = max_cycles

    def simulate(self, comp_vec, T=318.0, n_cycles=500, initial_state=None):
        self.model.set_composition(comp_vec, T)
        if initial_state is None:
            Q0 = 150.0
            V0 = 3.8
            x0 = 1.0
            state = torch.tensor([[Q0, V0, x0]], dtype=torch.float32)
        else:
            state = initial_state.unsqueeze(0) if initial_state.dim() == 1 else initial_state

        trajectory = [state.detach().clone()]
        t = torch.tensor(0.0)

        for cycle in range(n_cycles):
            t_cycle = torch.tensor(float(cycle))
            if self.solver == 'euler':
                dstate = self.model(t_cycle, state)
                state = state + self.dt * dstate
            elif self.solver == 'rk4':
                k1 = self.model(t_cycle, state)
                k2 = self.model(t_cycle + 0.5 * self.dt, state + 0.5 * self.dt * k1)
                k3 = self.model(t_cycle + 0.5 * self.dt, state + 0.5 * self.dt * k2)
                k4 = self.model(t_cycle + self.dt, state + self.dt * k3)
                state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            state = torch.clamp(state, min=0.0)
            state[..., 0] = torch.clamp(state[..., 0], min=10.0, max=300.0)
            state[..., 1] = torch.clamp(state[..., 1], min=2.0, max=5.0)
            state[..., 2] = torch.clamp(state[..., 2], min=0.0, max=1.0)

            trajectory.append(state.detach().clone())

        trajectory = torch.cat(trajectory, dim=0)
        return {
            'trajectory': trajectory,
            'capacity': trajectory[:, 0].numpy(),
            'voltage': trajectory[:, 1].numpy(),
            'stoichiometry': trajectory[:, 2].numpy(),
            'cycles': np.arange(n_cycles + 1),
            'fade_pct': float(1.0 - trajectory[-1, 0].item() / trajectory[0, 0].item())
        }


class MultiTemperatureTrainer:
    def __init__(self, model, temperatures=None):
        self.model = model
        self.temperatures = temperatures or [298, 308, 318, 328, 338]

    def compute_arrhenius_consistency(self, comp_vec):
        k_values = []
        for T in self.temperatures:
            self.model.set_composition(comp_vec, T)
            k = self.model.arrhenius(torch.tensor(float(T)))
            k_values.append(k)
        k_tensor = torch.stack(k_values)
        T_tensor = torch.tensor(self.temperatures, dtype=torch.float32)
        ln_k = torch.log(k_tensor + 1e-30)
        inv_T = 1.0 / T_tensor
        n = len(self.temperatures)
        mean_x = inv_T.mean()
        mean_y = ln_k.mean()
        slope = ((inv_T - mean_x) * (ln_k - mean_y)).sum() / ((inv_T - mean_x)**2).sum()
        ss_res = ((ln_k - mean_y - slope * (inv_T - mean_x))**2).sum()
        ss_tot = ((ln_k - mean_y)**2).sum()
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)
        Ea_estimated = -slope * 8.617e-5
        return {
            'r_squared': r_squared,
            'Ea_estimated': Ea_estimated,
            'k_values': k_tensor,
            'linearity_loss': 1.0 - r_squared
        }
