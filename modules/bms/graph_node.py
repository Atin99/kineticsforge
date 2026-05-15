import torch
import torch.nn as nn
from core.neural_ode import GeneralGraphNeuralODE, DOPRI5, AdjointODE

class EdgeFeatureExtractor(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x, edge_index):
        row, col = edge_index
        src = x[row]
        dst = x[col]
        return self.net(torch.cat([src, dst], dim=-1))

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, heads * out_features))
        self.a = nn.Parameter(torch.randn(heads, 2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(0.2)
    def forward(self, h, edge_index):
        row, col = edge_index
        Wh = torch.matmul(h, self.weight).view(-1, self.heads, self.out_features)
        Wh_src = Wh[row]
        Wh_dst = Wh[col]
        a_input = torch.cat([Wh_src, Wh_dst], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        e_max = torch.zeros(h.shape[0], self.heads, device=h.device).scatter_reduce(0, col.unsqueeze(-1).expand_as(e), e, 'amax')
        e_exp = torch.exp(e - e_max[col])
        e_sum = torch.zeros(h.shape[0], self.heads, device=h.device).scatter_add(0, col.unsqueeze(-1).expand_as(e_exp), e_exp)
        alpha = e_exp / (e_sum[col] + 1e-12)
        h_prime_edge = alpha.unsqueeze(-1) * Wh_src
        h_prime = torch.zeros(h.shape[0], self.heads, self.out_features, device=h.device).scatter_add(0, col.unsqueeze(-1).unsqueeze(-1).expand_as(h_prime_edge), h_prime_edge)
        return h_prime.view(-1, self.heads * self.out_features)

class ContinuousGAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, heads=4):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_dim, hidden_dim, heads)
        self.gat2 = GraphAttentionLayer(hidden_dim * heads, node_dim, 1)
        self.node_dim = node_dim
    def forward(self, t, x, edge_index):
        h = self.gat1(x, edge_index)
        h = nn.functional.elu(h)
        h = self.gat2(h, edge_index)
        return h

class TemporalEdgeMemory(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(edge_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, edge_dim)

    def forward(self, edge_attr, previous_memory=None):
        if previous_memory is None:
            previous_memory = torch.zeros(edge_attr.shape[0], self.gru.hidden_size, device=edge_attr.device, dtype=edge_attr.dtype)
        memory = self.gru(edge_attr, previous_memory)
        return self.proj(memory), memory


class TemporalGraphNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_extractor = EdgeFeatureExtractor(node_dim, hidden_dim, edge_dim)
        self.edge_memory = TemporalEdgeMemory(edge_dim, hidden_dim)
        self.mpnn = GeneralGraphNeuralODE(node_dim, edge_dim, hidden_dim, [hidden_dim, hidden_dim], [hidden_dim, hidden_dim])
        self.gat = ContinuousGAT(node_dim, hidden_dim, 4)
        self.state_gate = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.edge_decay = nn.Parameter(torch.tensor(0.015))
    def forward(self, t, x, context):
        if len(context) == 2:
            edge_index, static_edge_attr = context
            previous_edge_memory = None
        else:
            edge_index, static_edge_attr, previous_edge_memory = context
        dynamic_edge_attr = self.edge_extractor(x, edge_index)
        memory_edge_attr, _ = self.edge_memory(static_edge_attr + dynamic_edge_attr, previous_edge_memory)
        decay = torch.exp(-torch.abs(self.edge_decay) * torch.as_tensor(t, dtype=x.dtype, device=x.device))
        edge_attr = static_edge_attr + dynamic_edge_attr + decay * memory_edge_attr
        dx_mpnn = self.mpnn(t, x, edge_index, edge_attr)
        dx_gat = self.gat(t, x, edge_index)
        row, col = edge_index
        edge_context = torch.zeros(x.shape[0], edge_attr.shape[-1], device=x.device, dtype=x.dtype)
        edge_context.scatter_add_(0, col.unsqueeze(-1).expand(-1, edge_attr.shape[-1]), edge_attr)
        counts = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(0, col.unsqueeze(-1), torch.ones_like(col, dtype=x.dtype).unsqueeze(-1))
        edge_context = edge_context / torch.clamp(counts, min=1.0)
        gate = self.state_gate(torch.cat([x, edge_context], dim=-1))
        mixed = torch.sigmoid(self.alpha) * dx_mpnn + (1 - torch.sigmoid(self.alpha)) * dx_gat
        return gate * mixed


class AsymmetricAlertLoss(nn.Module):
    def __init__(self, late_weight=10.0, early_weight=0.1, missed_weight=25.0, threshold=0.5):
        super().__init__()
        self.late_weight = late_weight
        self.early_weight = early_weight
        self.missed_weight = missed_weight
        self.threshold = threshold

    def forward(self, pred_risk, true_failure_step, current_step=None):
        risk = pred_risk.squeeze(-1)
        if risk.dim() == 1:
            risk = risk.unsqueeze(0)
        batch, steps = risk.shape[0], risk.shape[1]
        device = risk.device
        if current_step is None:
            timeline = torch.arange(steps, device=device).float().unsqueeze(0).expand(batch, -1)
        else:
            timeline = current_step.to(device).float()
            if timeline.dim() == 1:
                timeline = timeline.unsqueeze(0).expand(batch, -1)
        fail = true_failure_step.to(device).float().view(batch, 1)
        alert_mask = risk > self.threshold
        alert_time = torch.where(alert_mask, timeline, torch.full_like(timeline, float("inf"))).min(dim=1).values
        missed = ~torch.isfinite(alert_time)
        alert_time = torch.where(missed, fail.squeeze(1) + 1.0, alert_time)
        lead = fail.squeeze(1) - alert_time
        late = torch.relu(-lead)
        too_early = torch.relu(lead - 4.0 * torch.clamp(fail.squeeze(1), min=1.0))
        confidence_loss = torch.mean(torch.relu(self.threshold - risk.max(dim=1).values) ** 2)
        return self.late_weight * torch.mean(late ** 2) + self.early_weight * torch.mean(too_early ** 2) + self.missed_weight * missed.float().mean() + confidence_loss


class SpatioTemporalGraphODE(TemporalGraphNetwork):
    pass

class PackTopology:
    def __init__(self, series, parallel):
        self.s = series
        self.p = parallel
        self.n_cells = series * parallel
        self.edge_index = self._build_topology()
        self.thermal_dist = self._build_thermal_distances()
    def _build_topology(self):
        edges = []
        for p_idx in range(self.p):
            for s_idx in range(self.s - 1):
                idx = p_idx * self.s + s_idx
                edges.append([idx, idx + 1])
                edges.append([idx + 1, idx])
        for s_idx in range(self.s):
            for p_idx in range(self.p - 1):
                idx = p_idx * self.s + s_idx
                next_p = (p_idx + 1) * self.s + s_idx
                edges.append([idx, next_p])
                edges.append([next_p, idx])
        return torch.tensor(edges, dtype=torch.long).T
    def _build_thermal_distances(self):
        row, col = self.edge_index
        row_s, row_p = row % self.s, row // self.s
        col_s, col_p = col % self.s, col // self.s
        dist_s = torch.abs(row_s - col_s).float()
        dist_p = torch.abs(row_p - col_p).float()
        return torch.sqrt(dist_s**2 + dist_p**2).unsqueeze(-1)

class BMSGraphIntegrator:
    def __init__(self, node_dim=7, hidden_dim=64):
        self.vector_field = SpatioTemporalGraphODE(node_dim, 3, hidden_dim)
        self.solver = DOPRI5()
        self.ode = AdjointODE(self.vector_field, self.solver)
        self.topology = PackTopology(4, 2)
        self.edge_attr = torch.cat([
            self.topology.thermal_dist,
            torch.ones_like(self.topology.thermal_dist) * 0.5,
            torch.ones_like(self.topology.thermal_dist) * 0.01
        ], dim=-1)
    def simulate_pack(self, initial_state, duration, dt_report=1.0):
        t_span = torch.tensor([0.0, duration])
        context = (self.topology.edge_index, self.edge_attr)
        traj, t_seq = self.ode.solve(initial_state, t_span, context=context)
        return traj

class PrecursorAnomalyDetector(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(node_dim, hidden_dim, num_layers=2, batch_first=True)
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, trajectory):
        batch, seq, dim = trajectory.shape
        out, _ = self.rnn(trajectory)
        risk = self.detector(out)
        return risk.squeeze(-1)

class CascadeFailureSimulator:
    def __init__(self, bms_integrator):
        self.bms = bms_integrator
    def trigger_thermal_runaway(self, state, cell_idx, intensity):
        state[cell_idx, 2] += intensity
        return state
    def simulate_cascade(self, initial_state, duration):
        traj = []
        curr = initial_state
        for t in range(int(duration)):
            curr_traj = self.bms.simulate_pack(curr, 1.0)
            curr = curr_traj[-1]
            traj.append(curr.clone())
            if torch.any(curr[:, 2] > 100.0):
                neighbors = self.bms.topology.edge_index[1][self.bms.topology.edge_index[0] == torch.argmax(curr[:, 2])]
                curr[neighbors, 2] += 5.0
        return torch.stack(traj)

class PackEquilibration:
    def __init__(self, n_cells, balance_current):
        self.n = n_cells
        self.i_bal = balance_current
        self.adj = torch.zeros(n_cells, n_cells)
    def active_balancing(self, voltages):
        v_mean = torch.mean(voltages)
        currents = torch.zeros_like(voltages)
        currents[voltages > v_mean + 0.01] = -self.i_bal
        currents[voltages < v_mean - 0.01] = self.i_bal
        return currents
    def passive_balancing(self, voltages):
        v_min = torch.min(voltages)
        currents = torch.zeros_like(voltages)
        currents[voltages > v_min + 0.01] = -self.i_bal
        return currents

class ThermalLaplacian:
    def __init__(self, topology):
        row, col = topology.edge_index
        n = topology.n_cells
        self.L = torch.zeros(n, n)
        for r, c, d in zip(row, col, topology.thermal_dist):
            w = 1.0 / (d.item() + 1e-6)
            self.L[r, c] = -w
            self.L[r, r] += w
        self.eigenvals, self.eigenvecs = torch.linalg.eigh(self.L)
    def diffuse(self, t, T_init, k_thermal):
        exp_L = torch.matrix_exp(-k_thermal * t * self.L)
        return exp_L @ T_init
    def spectral_diffuse(self, t, T_init, k_thermal):
        coeffs = self.eigenvecs.T @ T_init
        decay = torch.exp(-k_thermal * t * self.eigenvals)
        return self.eigenvecs @ (decay.unsqueeze(-1) * coeffs)

class SEIGraphPropagation:
    def __init__(self, n_cells):
        self.M = torch.eye(n_cells)
    def couple_resistance(self, r_int, L_sei, temp):
        return r_int + 0.1 * L_sei * torch.exp((temp - 25.0)/10.0)
    def inter_cell_stress(self, L_sei, edge_index):
        row, col = edge_index
        stress = torch.zeros_like(L_sei)
        diff = L_sei[row] - L_sei[col]
        stress.scatter_add_(0, row, torch.relu(diff))
        return stress

class StateObserver:
    def __init__(self, f_sys, h_obs, Q, R):
        self.f = f_sys
        self.h = h_obs
        self.Q = Q
        self.R = R
    def ekf_step(self, x, P, y, u, dt):
        x_req = x.clone().requires_grad_(True)
        f_val = self.f(x_req, u)
        F = torch.zeros(x.shape[0], x.shape[0])
        for i in range(x.shape[0]):
            F[i] = torch.autograd.grad(f_val[i], x_req, retain_graph=True)[0]
        x_pred = x + dt * f_val
        P_pred = P + dt * (F @ P + P @ F.T + self.Q)
        x_pred_req = x_pred.clone().requires_grad_(True)
        h_val = self.h(x_pred_req)
        H = torch.zeros(y.shape[0], x_pred.shape[0])
        for i in range(y.shape[0]):
            H[i] = torch.autograd.grad(h_val[i], x_pred_req, retain_graph=True)[0]
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ torch.inverse(S)
        x_upd = x_pred + K @ (y - h_val)
        P_upd = (torch.eye(x.shape[0]) - K @ H) @ P_pred
        return x_upd, P_upd

class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, kappa=0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kappa = kappa
        self.W_m = torch.zeros(2 * dim_x + 1)
        self.W_c = torch.zeros(2 * dim_x + 1)
        self.W_m[0] = kappa / (dim_x + kappa)
        self.W_c[0] = kappa / (dim_x + kappa)
        for i in range(1, 2 * dim_x + 1):
            self.W_m[i] = 1.0 / (2 * (dim_x + kappa))
            self.W_c[i] = 1.0 / (2 * (dim_x + kappa))
    def sigma_points(self, x, P):
        n = x.shape[0]
        L = torch.linalg.cholesky((n + self.kappa) * P)
        sigmas = [x]
        for i in range(n):
            sigmas.append(x + L[:, i])
        for i in range(n):
            sigmas.append(x - L[:, i])
        return torch.stack(sigmas)
    def step(self, x, P, y, f, h, dt):
        sigmas = self.sigma_points(x, P)
        sigmas_f = torch.stack([f(s, dt) for s in sigmas])
        x_pred = torch.sum(self.W_m.unsqueeze(-1) * sigmas_f, dim=0)
        P_pred = torch.zeros_like(P)
        for i in range(len(sigmas)):
            diff = sigmas_f[i] - x_pred
            P_pred += self.W_c[i] * torch.outer(diff, diff)
        sigmas_h = torch.stack([h(s) for s in sigmas_f])
        z_pred = torch.sum(self.W_m.unsqueeze(-1) * sigmas_h, dim=0)
        P_zz = torch.zeros(self.dim_z, self.dim_z)
        P_xz = torch.zeros(self.dim_x, self.dim_z)
        for i in range(len(sigmas)):
            dz = sigmas_h[i] - z_pred
            dx = sigmas_f[i] - x_pred
            P_zz += self.W_c[i] * torch.outer(dz, dz)
            P_xz += self.W_c[i] * torch.outer(dx, dz)
        K = P_xz @ torch.inverse(P_zz)
        x_upd = x_pred + K @ (y - z_pred)
        P_upd = P_pred - K @ P_zz @ K.T
        return x_upd, P_upd
