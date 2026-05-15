import torch
import torch.nn as nn
import math

class ElementalFeatureDatabase:
    def __init__(self):
        self.atomic_number = {'Na': 11, 'Mn': 25, 'Fe': 26, 'O': 8, 'Al': 13, 'Ti': 22, 'Mg': 12, 'Li': 3, 'Ni': 28, 'Co': 27, 'V': 23, 'C': 6}
        self.electronegativity = {'Na': 0.93, 'Mn': 1.55, 'Fe': 1.83, 'O': 3.44, 'Al': 1.61, 'Ti': 1.54, 'Mg': 1.31, 'Li': 0.98, 'Ni': 1.91, 'Co': 1.88, 'V': 1.63, 'C': 2.55}
        self.atomic_radius = {'Na': 186, 'Mn': 127, 'Fe': 126, 'O': 60, 'Al': 143, 'Ti': 147, 'Mg': 160, 'Li': 152, 'Ni': 124, 'Co': 125, 'V': 134, 'C': 77}
        self.ionization_energy = {'Na': 5.14, 'Mn': 7.43, 'Fe': 7.90, 'O': 13.62, 'Al': 5.99, 'Ti': 6.83, 'Mg': 7.65, 'Li': 5.39, 'Ni': 7.64, 'Co': 7.88, 'V': 6.75, 'C': 11.26}
        self.electron_affinity = {'Na': 0.55, 'Mn': 0.0, 'Fe': 0.15, 'O': 1.46, 'Al': 0.43, 'Ti': 0.08, 'Mg': 0.0, 'Li': 0.62, 'Ni': 1.16, 'Co': 0.66, 'V': 0.53, 'C': 1.26}
        self.valences = {'Na': [1], 'Mn': [2,3,4,6,7], 'Fe': [2,3,6], 'O': [-2], 'Al': [3], 'Ti': [3,4], 'Mg': [2], 'Li': [1], 'Ni': [2,3], 'Co': [2,3], 'V': [2,3,4,5], 'C': [4,-4]}
        self.elements = list(self.atomic_number.keys())
        self.n_elements = len(self.elements)
        self.feature_dim = 6
        self.feature_matrix = self._build_matrix()
    def _build_matrix(self):
        mat = torch.zeros(self.n_elements, self.feature_dim)
        for i, el in enumerate(self.elements):
            mat[i, 0] = self.atomic_number[el] / 100.0
            mat[i, 1] = self.electronegativity[el] / 4.0
            mat[i, 2] = self.atomic_radius[el] / 200.0
            mat[i, 3] = self.ionization_energy[el] / 20.0
            mat[i, 4] = self.electron_affinity[el] / 4.0
            mat[i, 5] = sum(self.valences[el]) / len(self.valences[el]) / 8.0
        return mat

class MultiHeadElementAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadElementAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x

class CompositionEmbedder(nn.Module):
    def __init__(self, input_dim=5, embed_dim=32):
        super().__init__()
        self.db = ElementalFeatureDatabase()
        self.elem_embedding = nn.Linear(self.db.feature_dim, embed_dim)
        self.frac_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, 4, embed_dim * 4) for _ in range(3)
        ])
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        self.property_predictors = nn.ModuleDict({
            'lattice_a': nn.Linear(embed_dim, 1),
            'lattice_b': nn.Linear(embed_dim, 1),
            'lattice_c': nn.Linear(embed_dim, 1),
            'formation_energy': nn.Linear(embed_dim, 1),
            'band_gap': nn.Linear(embed_dim, 1)
        })
        self.legacy_mapper = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, embed_dim)
        )
        self.use_legacy = True
    def set_mode(self, legacy=False):
        self.use_legacy = legacy
    def forward(self, x):
        if self.use_legacy:
            return self.legacy_mapper(x)
        batch_size = x.shape[0]
        na_idx = self.db.elements.index('Na')
        mn_idx = self.db.elements.index('Mn')
        fe_idx = self.db.elements.index('Fe')
        dopant_indices = x[:, 3].long()
        fracs = torch.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 4]], dim=-1)
        indices = torch.stack([torch.full((batch_size,), na_idx, device=x.device), torch.full((batch_size,), mn_idx, device=x.device), torch.full((batch_size,), fe_idx, device=x.device), dopant_indices], dim=-1)
        elem_feats = self.db.feature_matrix.to(x.device)[indices]
        e_emb = self.elem_embedding(elem_feats)
        f_emb = self.frac_embedding(fracs.unsqueeze(-1))
        seq = self.combiner(torch.cat([e_emb, f_emb], dim=-1))
        for layer in self.transformer:
            seq = layer(seq)
        pooled = self.pooler(seq.mean(dim=1))
        return pooled
    def predict_properties(self, z):
        return {k: v(z) for k, v in self.property_predictors.items()}

class PhaseDiagramNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
    def forward(self, z):
        return self.net(z)

class StructuralDescriptor(nn.Module):
    def __init__(self, max_atoms, embed_dim):
        super().__init__()
        self.max_atoms = max_atoms
        self.coord_net = nn.Sequential(
            nn.Linear(embed_dim, max_atoms * 3),
            nn.Tanh()
        )
    def forward(self, z):
        coords = self.coord_net(z).view(-1, self.max_atoms, 3)
        return coords

class SmoothOverlapOfAtomicPositions(nn.Module):
    def __init__(self, n_max, l_max, r_cut):
        super().__init__()
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        self.sigma = 0.5
    def forward(self, coords):
        batch = coords.shape[0]
        n_atoms = coords.shape[1]
        soap = torch.zeros(batch, self.n_max, self.n_max, self.l_max + 1, device=coords.device)
        return soap.view(batch, -1)

class E3EquivariantNetwork(nn.Module):
    def __init__(self, l_max, embed_dim):
        super().__init__()
        self.l_max = l_max
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(embed_dim, embed_dim)) for _ in range(l_max + 1)])
    def forward(self, features, spherical_harmonics):
        out = torch.zeros_like(features)
        for l in range(self.l_max + 1):
            out += torch.matmul(features, self.weights[l]) * spherical_harmonics[l]
        return out

class CompositionStabilityHeuristicDatabase:
    def __init__(self):
        self.sublattices = 3
        self.sites = [1, 1, 2]
    def calculate_gibbs(self, z, T):
        h = torch.sum(z**2, dim=-1) * T
        s = torch.sum(z * torch.log(torch.abs(z) + 1e-9), dim=-1)
        g = h - T * s
        return g

def parse_composition(comp_dict):
    dopant_map = {None: 0.0, 'Al': 4.0, 'Ti': 5.0, 'Mg': 6.0}
    dopant_id = dopant_map.get(comp_dict.get('dopant'), 0.0)
    return torch.tensor([
        comp_dict.get('Na', 1.0),
        comp_dict.get('Mn', 0.5),
        comp_dict.get('Fe', 0.5),
        dopant_id,
        comp_dict.get('dopant_frac', 0.0)
    ], dtype=torch.float32)

class PhysicsInformedCompositionEmbedder(CompositionEmbedder):
    def __init__(self, input_dim=5, embed_dim=32):
        super().__init__(input_dim, embed_dim)
        self.stability_heuristic = CompositionStabilityHeuristicDatabase()
        self.phase_net = PhaseDiagramNet(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 1 + 3, embed_dim),
            nn.GELU()
        )
    def forward(self, x, T):
        z_base = super().forward(x)
        g = self.stability_heuristic.calculate_gibbs(z_base, T).unsqueeze(-1)
        p = self.phase_net(z_base)
        return self.fusion(torch.cat([z_base, g, p], dim=-1))


# V4 naming: "CALPHAD" is removed because we do NOT use real Gibbs energy
# databases (Thermo-Calc, FactSage). These are composition-informed heuristics.
# Backward-compat aliases kept but honestly named.
StabilityHeuristicDatabase = CompositionStabilityHeuristicDatabase
PhysicsInformedEmbedder = PhysicsInformedCompositionEmbedder

# Deprecated aliases — do NOT use in new code
CALPHADThermodynamicDatabase = CompositionStabilityHeuristicDatabase
HybridCALPHADEmbedder = PhysicsInformedCompositionEmbedder
