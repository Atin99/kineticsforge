import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, List, Tuple, Optional


class TopologyFactory:
    @staticmethod
    def series_string(n_series: int) -> Tuple[torch.Tensor, int]:
        edges = []
        for i in range(n_series - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long), n_series

    @staticmethod
    def parallel_bank(n_parallel: int) -> Tuple[torch.Tensor, int]:
        edges = []
        for i in range(n_parallel):
            for j in range(i + 1, n_parallel):
                edges.append([i, j])
                edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long), n_parallel

    @staticmethod
    def series_parallel(n_s: int, n_p: int) -> Tuple[torch.Tensor, int]:
        n = n_s * n_p
        edges = []
        for p in range(n_p):
            for s in range(n_s - 1):
                a = p * n_s + s
                b = a + 1
                edges.append([a, b])
                edges.append([b, a])
        for s in range(n_s):
            for p in range(n_p - 1):
                a = p * n_s + s
                b = (p + 1) * n_s + s
                edges.append([a, b])
                edges.append([b, a])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long), n

    @staticmethod
    def pouch_stack(n_layers: int, cells_per_layer: int) -> Tuple[torch.Tensor, int]:
        n = n_layers * cells_per_layer
        edges = []
        for layer in range(n_layers):
            for c in range(cells_per_layer - 1):
                a = layer * cells_per_layer + c
                b = a + 1
                edges.append([a, b])
                edges.append([b, a])
        for layer in range(n_layers - 1):
            for c in range(cells_per_layer):
                a = layer * cells_per_layer + c
                b = (layer + 1) * cells_per_layer + c
                edges.append([a, b])
                edges.append([b, a])
        return torch.tensor(edges, dtype=torch.long).T, n

    @staticmethod
    def cylindrical_module(n_rows: int, n_cols: int) -> Tuple[torch.Tensor, int]:
        n = n_rows * n_cols
        edges = []
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                if c < n_cols - 1:
                    edges.append([idx, idx + 1])
                    edges.append([idx + 1, idx])
                if r < n_rows - 1:
                    nb = (r + 1) * n_cols + c
                    edges.append([idx, nb])
                    edges.append([nb, idx])
                if r < n_rows - 1 and c < n_cols - 1:
                    diag = (r + 1) * n_cols + c + 1
                    edges.append([idx, diag])
                    edges.append([diag, idx])
        return torch.tensor(edges, dtype=torch.long).T, n

    @staticmethod
    def thermal_distances(edge_index: torch.Tensor, n_cells: int,
                          cell_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cell_positions is None:
            cell_positions = torch.randn(n_cells, 3) * 0.05
        row, col = edge_index
        diff = cell_positions[row] - cell_positions[col]
        return torch.norm(diff, dim=-1, keepdim=True)


class ThermalAdjacencyCalibrator(nn.Module):
    def __init__(self, n_cells: int, n_edges: int):
        super().__init__()
        self.log_k = nn.Parameter(torch.zeros(n_edges))
        self.base_conductance = nn.Parameter(torch.ones(1) * 0.5)
        self.temperature_sensitivity = nn.Parameter(torch.ones(1) * 0.001)

    def forward(self, thermal_dist: torch.Tensor, T_cells: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        k = torch.exp(self.log_k) * self.base_conductance
        row, col = edge_index
        T_mean_edge = (T_cells[row] + T_cells[col]) / 2.0
        T_correction = 1.0 + self.temperature_sensitivity * (T_mean_edge - 298.0)
        conductance = k * T_correction.squeeze(-1) / (thermal_dist.squeeze(-1) + 1e-6)
        return conductance


class AlertLeadTimeLoss(nn.Module):
    def __init__(self, target_lead_time: float = 300.0, penalty_late: float = 5.0,
                 penalty_early: float = 0.5):
        super().__init__()
        self.target = target_lead_time
        self.w_late = penalty_late
        self.w_early = penalty_early

    def forward(self, pred_risk: torch.Tensor, true_failure_step: torch.Tensor,
                current_step: torch.Tensor) -> torch.Tensor:
        alert_mask = pred_risk > 0.5
        if not alert_mask.any():
            return self.w_late * torch.mean(torch.relu(pred_risk - 0.3) ** 2)

        remaining = true_failure_step - current_step
        lead_time = remaining * alert_mask.float()
        too_late = torch.relu(self.target - lead_time) * alert_mask.float()
        too_early = torch.relu(lead_time - self.target * 3.0) * alert_mask.float()
        return self.w_late * torch.mean(too_late ** 2) + self.w_early * torch.mean(too_early ** 2)


class FalsePositivePenalty(nn.Module):
    def __init__(self, fp_weight: float = 2.0, fn_weight: float = 5.0):
        super().__init__()
        self.fp_w = fp_weight
        self.fn_w = fn_weight

    def forward(self, pred_risk: torch.Tensor, true_risk: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        pred_pos = (pred_risk > threshold).float()
        true_pos = (true_risk > threshold).float()
        fp = pred_pos * (1.0 - true_pos)
        fn = (1.0 - pred_pos) * true_pos
        base_mse = torch.mean((pred_risk - true_risk) ** 2)
        return base_mse + self.fp_w * torch.mean(fp * pred_risk ** 2) + \
               self.fn_w * torch.mean(fn * (1.0 - pred_risk) ** 2)


class HeterogeneousGraphConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.electrical_msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.thermal_msg = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        src, dst = x[row], x[col]
        cat = torch.cat([src, dst, edge_attr], dim=-1)
        elec_mask = (edge_type == 0).unsqueeze(-1).float()
        therm_mask = (edge_type == 1).unsqueeze(-1).float()
        msg = self.electrical_msg(cat) * elec_mask + self.thermal_msg(cat) * therm_mask
        aggr = torch.zeros(x.shape[0], msg.shape[-1], device=x.device)
        aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)
        elec_aggr = torch.zeros_like(aggr)
        elec_aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg * elec_mask)
        therm_aggr = torch.zeros_like(aggr)
        therm_aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg * therm_mask)
        return self.update_fn(torch.cat([x, elec_aggr, therm_aggr], dim=-1))


class TemporalAttentionAggregator(nn.Module):
    def __init__(self, node_dim: int, n_heads: int = 4, window: int = 16):
        super().__init__()
        self.window = window
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.qkv = nn.Linear(node_dim, node_dim * 3)
        self.proj = nn.Linear(node_dim, node_dim)

    def forward(self, node_history: torch.Tensor) -> torch.Tensor:
        B, T, D = node_history.shape
        T_use = min(T, self.window)
        x = node_history[:, -T_use:]
        qkv = self.qkv(x).reshape(B, T_use, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(T_use, T_use, device=x.device))
        attn = attn.masked_fill(causal == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, T_use, D)
        return self.proj(out[:, -1])


class MultiScaleGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, cells_per_module: int = 4):
        super().__init__()
        self.cells_per_module = cells_per_module
        self.cell_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.module_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pack_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_feat = self.cell_encoder(x)
        n_cells = x.shape[0]
        n_modules = max(1, n_cells // self.cells_per_module)
        mod_feats = []
        for m in range(n_modules):
            start = m * self.cells_per_module
            end = min(start + self.cells_per_module, n_cells)
            mod_feats.append(torch.mean(cell_feat[start:end], dim=0))
        module_feat = self.module_encoder(torch.stack(mod_feats))
        pack_feat = self.pack_encoder(torch.mean(module_feat, dim=0, keepdim=True))
        return cell_feat, module_feat, pack_feat


class SpectralGraphFilter(nn.Module):
    def __init__(self, node_dim: int, K: int = 3):
        super().__init__()
        self.K = K
        self.theta = nn.ParameterList([nn.Parameter(torch.randn(node_dim, node_dim) * 0.01)
                                        for _ in range(K)])

    def _compute_laplacian(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        A = torch.zeros(n, n, device=edge_index.device)
        row, col = edge_index
        A[row, col] = 1.0
        D = torch.diag(A.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(A.sum(dim=1)) + 1e-8))
        L_norm = torch.eye(n, device=edge_index.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L_norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        L = self._compute_laplacian(edge_index, n)
        T_prev = x
        T_curr = L @ x
        result = T_prev @ self.theta[0]
        if self.K > 1:
            result = result + T_curr @ self.theta[1]
        for k in range(2, self.K):
            T_next = 2.0 * L @ T_curr - T_prev
            result = result + T_next @ self.theta[k]
            T_prev = T_curr
            T_curr = T_next
        return result


class DynamicEdgeUpdate(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(node_dim * 2, 32), nn.GELU(), nn.Linear(32, edge_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        delta = self.edge_net(torch.cat([x[row], x[col]], dim=-1))
        return edge_attr + 0.1 * delta


class TopologyAwareBMSModel(nn.Module):
    def __init__(self, node_dim: int = 7, edge_dim: int = 3, hidden_dim: int = 64,
                 n_heads: int = 4, cheb_k: int = 3):
        super().__init__()
        self.hetero_conv = HeterogeneousGraphConv(node_dim, edge_dim, hidden_dim)
        self.spectral = SpectralGraphFilter(node_dim, cheb_k)
        self.temporal_attn = TemporalAttentionAggregator(node_dim, n_heads)
        self.multi_scale = MultiScaleGraphEncoder(node_dim, hidden_dim)
        self.dynamic_edge = DynamicEdgeUpdate(node_dim, edge_dim)
        self.mix_alpha = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        self.risk_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_type: torch.Tensor,
                node_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.mix_alpha, dim=0)
        dx_hetero = self.hetero_conv(x, edge_index, edge_attr, edge_type)
        dx_spectral = self.spectral(x, edge_index)
        dx = alpha[0] * dx_hetero + alpha[1] * dx_spectral
        if node_history is not None and node_history.shape[1] > 1:
            temporal_feat = self.temporal_attn(node_history)
            dx = dx + alpha[2] * temporal_feat
        edge_attr = self.dynamic_edge(x, edge_index, edge_attr)
        risk = self.risk_head(x + dx)
        return dx, risk.squeeze(-1)


class SyntheticPackGraphGenerator:
    def __init__(self, seed: int = 20260511):
        self.rng = np.random.RandomState(seed)

    def sample_topology(self, family: str = "mixed") -> Dict[str, torch.Tensor]:
        if family == "series":
            n_s, n_p = int(self.rng.choice([14, 16])), 1
        elif family == "parallel":
            n_s, n_p = int(self.rng.choice([7, 8])), int(self.rng.choice([4, 6]))
        else:
            n_s, n_p = int(self.rng.choice([7, 14, 16])), int(self.rng.choice([1, 2, 4]))
        edge_index, n_cells = TopologyFactory.series_parallel(n_s, n_p)
        positions = self._positions(n_s, n_p)
        thermal_dist = TopologyFactory.thermal_distances(edge_index, n_cells, positions)
        row, col = edge_index
        same_parallel = ((row // n_s) == (col // n_s)).long()
        edge_type = torch.where(same_parallel > 0, torch.zeros_like(same_parallel), torch.ones_like(same_parallel))
        thermal_k = torch.tensor(self.rng.uniform(0.5, 5.0, size=(edge_index.shape[1], 1)), dtype=torch.float32)
        bus_resistance = torch.tensor(self.rng.uniform(2e-4, 8e-3, size=(edge_index.shape[1], 1)), dtype=torch.float32)
        edge_attr = torch.cat([thermal_dist.float(), thermal_k, bus_resistance], dim=-1)
        capacity = torch.tensor(self.rng.normal(1.0, self.rng.uniform(0.02, 0.05), size=(n_cells, 1)), dtype=torch.float32).clamp(0.82, 1.15)
        failure = self._failure_metadata(n_cells)
        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_type": edge_type,
            "capacity_multiplier": capacity,
            "positions": positions,
            "n_cells": torch.tensor(n_cells),
            "n_series": torch.tensor(n_s),
            "n_parallel": torch.tensor(n_p),
            "failure_cell": torch.tensor(failure["cell"]),
            "failure_mode": torch.tensor(failure["mode_id"]),
            "failure_onset_step": torch.tensor(failure["onset_step"]),
        }

    def batch(self, n: int, family: str = "mixed") -> List[Dict[str, torch.Tensor]]:
        return [self.sample_topology(family=family) for _ in range(n)]

    def _positions(self, n_s: int, n_p: int) -> torch.Tensor:
        coords = []
        for p in range(n_p):
            for s in range(n_s):
                coords.append([0.021 * s, 0.023 * p, 0.002 * ((s + p) % 2)])
        jitter = self.rng.normal(0.0, 0.0015, size=(len(coords), 3))
        return torch.tensor(np.asarray(coords, dtype=float) + jitter, dtype=torch.float32)

    def _failure_metadata(self, n_cells: int) -> Dict[str, int]:
        modes = {"single_cell_thermal_runaway": 0, "cascading_dendrite": 1, "sei_gas_venting": 2}
        name = str(self.rng.choice(list(modes)))
        return {
            "cell": int(self.rng.randint(0, n_cells)),
            "mode_id": int(modes[name]),
            "onset_step": int(self.rng.randint(180, 900)),
        }
