import torch
import torch.nn as nn
import numpy as np

class QuadraticLyapunovCandidate(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        L_init = torch.eye(state_dim) * 0.1
        self.L = nn.Parameter(L_init)

    def P_matrix(self):
        return torch.matmul(self.L, self.L.T) + 1e-4 * torch.eye(self.L.shape[0], device=self.L.device)

    def forward(self, z, z_eq=None):
        if z_eq is None:
            z_eq = torch.zeros_like(z)
        delta = z - z_eq
        P = self.P_matrix()
        return torch.sum(delta * torch.matmul(delta, P), dim=-1)

    def eigenvalues(self):
        P = self.P_matrix()
        return torch.linalg.eigvalsh(P)


class NeuralLyapunovCandidate(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = []
        dims = [state_dim] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.eps = 1e-4

    def forward(self, z, z_eq=None):
        if z_eq is None:
            z_eq = torch.zeros_like(z)
        delta = z - z_eq
        raw = self.net(delta).squeeze(-1)
        norm_sq = torch.sum(delta**2, dim=-1)
        return raw**2 + self.eps * norm_sq


class InputConvexNeuralNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, n_layers=3):
        super().__init__()
        self.first = nn.Linear(state_dim, hidden_dim)
        self.layers_z = nn.ModuleList()
        self.layers_x = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers_z.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers_x.append(nn.Linear(state_dim, hidden_dim))
        self.final_z = nn.Linear(hidden_dim, 1)
        self.final_x = nn.Linear(state_dim, 1)

    def forward(self, z, z_eq=None):
        if z_eq is None:
            z_eq = torch.zeros_like(z)
        x = z - z_eq
        h = torch.relu(self.first(x))
        for lz, lx in zip(self.layers_z, self.layers_x):
            Wz = torch.relu(lz.weight)
            h = torch.relu(torch.matmul(h, Wz.T) + lz.bias + lx(x))
        Wf = torch.relu(self.final_z.weight)
        out = torch.matmul(h, Wf.T) + self.final_z.bias + self.final_x(x)
        return out.squeeze(-1) + 1e-4 * torch.sum(x**2, dim=-1)


class LyapunovStabilityLoss:
    def __init__(self, lyapunov_candidate, vector_field_fn,
                 lambda_decrease=10.0, lambda_positive=1.0, lambda_eq=5.0):
        self.V = lyapunov_candidate
        self.f = vector_field_fn
        self.lam_dec = lambda_decrease
        self.lam_pos = lambda_positive
        self.lam_eq = lambda_eq

    def compute_dVdt(self, z, t=0.0):
        z_req = z.clone().requires_grad_(True)
        V_val = self.V(z_req)
        dV_dz = torch.autograd.grad(V_val.sum(), z_req, create_graph=True)[0]
        f_val = self.f(t, z_req)
        dVdt = torch.sum(dV_dz * f_val, dim=-1)
        return dVdt, V_val

    def __call__(self, z, t=0.0, z_eq=None):
        dVdt, V_val = self.compute_dVdt(z, t)
        loss_decrease = self.lam_dec * torch.mean(torch.relu(dVdt))
        loss_positive = self.lam_pos * torch.mean(torch.relu(-V_val))
        V_eq = self.V(z_eq if z_eq is not None else torch.zeros_like(z[:1]))
        loss_eq = self.lam_eq * V_eq.mean()
        return loss_decrease + loss_positive + loss_eq

    def is_stable(self, z, t=0.0, z_eq=None, threshold=0.0):
        with torch.no_grad():
            dVdt, V_val = self.compute_dVdt(z, t)
            positive = (V_val > 0).all()
            decreasing = (dVdt <= threshold).all()
        return positive.item() and decreasing.item()

    def region_of_attraction(self, z_eq, n_samples=500, radius=2.0, t=0.0):
        dim = z_eq.shape[-1]
        directions = torch.randn(n_samples, dim)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        max_stable_radii = []
        for d in directions:
            lo, hi = 0.0, radius
            for _ in range(20):
                mid = (lo + hi) / 2
                test_point = z_eq + mid * d.unsqueeze(0)
                if self.is_stable(test_point, t, z_eq):
                    lo = mid
                else:
                    hi = mid
            max_stable_radii.append(lo)
        return {
            'mean_radius': float(np.mean(max_stable_radii)),
            'min_radius': float(np.min(max_stable_radii)),
            'max_radius': float(np.max(max_stable_radii)),
            'volume_estimate': float(np.mean(max_stable_radii)**dim * np.pi**(dim/2))
        }


class LyapunovRegularizedTrainer:
    def __init__(self, model, lyapunov_loss, data_loss_fn, lyapunov_weight=0.1):
        self.model = model
        self.lyap_loss = lyapunov_loss
        self.data_loss_fn = data_loss_fn
        self.lyap_weight = lyapunov_weight

    def compute_total_loss(self, pred, target, z_states, t=0.0):
        data_loss = self.data_loss_fn(pred, target)
        lyap_loss = self.lyap_loss(z_states, t)
        return data_loss + self.lyap_weight * lyap_loss, data_loss.item(), lyap_loss.item()

    def stability_certificate(self, z_test, t=0.0, z_eq=None):
        stable = self.lyap_loss.is_stable(z_test, t, z_eq)
        with torch.no_grad():
            dVdt, V_val = self.lyap_loss.compute_dVdt(z_test, t)
        return {
            'stable': stable,
            'V_mean': V_val.mean().item(),
            'V_max': V_val.max().item(),
            'dVdt_max': dVdt.max().item(),
            'dVdt_mean': dVdt.mean().item(),
            'all_positive': (V_val > 0).all().item(),
            'all_decreasing': (dVdt <= 0).all().item()
        }
