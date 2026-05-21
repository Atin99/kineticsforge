import torch
import torch.nn as nn


def _as_batch(x: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device=device, dtype=torch.float32)
    if x.dim() == 0:
        x = x.view(1, 1).expand(batch, 1)
    elif x.dim() == 1:
        if x.numel() == batch:
            x = x.view(batch, 1)
        else:
            x = x.view(1, -1).expand(batch, -1)
    elif x.shape[0] != batch:
        x = x.expand(batch, -1)
    return x


class JahnTellerCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.mn3_weight = nn.Parameter(torch.tensor(1.0))
        self.fe_damping = nn.Parameter(torch.tensor(0.45))
        self.dopant_damping = nn.Parameter(torch.tensor(0.70))
        self.temperature_gain = nn.Parameter(torch.tensor(0.018))

    def forward(self, composition: torch.Tensor, temperature_K: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
        batch = soc.shape[0]
        comp = _as_batch(composition, batch, soc.device)
        T = _as_batch(temperature_K, batch, soc.device)
        mn = torch.clamp(comp[:, 1:2] if comp.shape[1] > 1 else torch.zeros_like(soc), 0.0, 1.5)
        fe = torch.clamp(comp[:, 2:3] if comp.shape[1] > 2 else torch.zeros_like(soc), 0.0, 1.5)
        dopant = torch.clamp(comp[:, 4:5] if comp.shape[1] > 4 else torch.zeros_like(soc), 0.0, 0.25)
        mn3_proxy = mn * torch.clamp(1.15 - soc, 0.0, 1.0)
        thermal = torch.exp(torch.clamp((T - 298.15) * torch.abs(self.temperature_gain), -4.0, 4.0))
        damping = torch.exp(-torch.abs(self.fe_damping) * fe - torch.abs(self.dopant_damping) * dopant)
        return torch.clamp(torch.abs(self.mn3_weight) * mn3_proxy * thermal * damping, 0.0, 4.0)


class P2O2PhaseTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_soc_crit = nn.Parameter(torch.tensor(0.78))
        self.width = nn.Parameter(torch.tensor(0.045))
        self.k_transition = nn.Parameter(torch.tensor(1.6e-3))
        self.mn_sensitivity = nn.Parameter(torch.tensor(0.09))
        self.fe_stabilization = nn.Parameter(torch.tensor(0.06))
        self.dopant_stabilization = nn.Parameter(torch.tensor(0.18))
        self.temperature_sensitivity = nn.Parameter(torch.tensor(0.024))

    def forward(self, state: torch.Tensor, composition: torch.Tensor, temperature_K: torch.Tensor, jt_factor: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        comp = _as_batch(composition, batch, state.device)
        T = _as_batch(temperature_K, batch, state.device)
        soc = torch.clamp(state[..., 2:3], 0.0, 1.0)
        mn = torch.clamp(comp[:, 1:2] if comp.shape[1] > 1 else torch.zeros_like(soc), 0.0, 1.5)
        fe = torch.clamp(comp[:, 2:3] if comp.shape[1] > 2 else torch.zeros_like(soc), 0.0, 1.5)
        dopant = torch.clamp(comp[:, 4:5] if comp.shape[1] > 4 else torch.zeros_like(soc), 0.0, 0.25)
        soc_crit = (
            torch.clamp(self.base_soc_crit, 0.55, 0.95)
            - torch.abs(self.mn_sensitivity) * mn
            + torch.abs(self.fe_stabilization) * fe
            + torch.abs(self.dopant_stabilization) * dopant
        )
        width = torch.clamp(torch.abs(self.width), 0.01, 0.20)
        high_soc_gate = torch.sigmoid((soc - soc_crit) / width)
        thermal = torch.exp(torch.clamp((T - 298.15) * torch.abs(self.temperature_sensitivity) / 25.0, -3.0, 3.0))
        rate = torch.abs(self.k_transition) * high_soc_gate * thermal * (1.0 + 0.35 * jt_factor)
        return torch.clamp(rate, 0.0, 0.08)


class NaDesolvationBarrier(nn.Module):
    """Na+ desolvation barrier model for layered oxide cathodes.
    
    Physical basis: Na+ must shed its solvation shell (typically 4-6 carbonate
    molecules in EC/PC electrolytes) before intercalating. The energy barrier
    for this process is 0.4-0.6 eV for Na+ (vs ~0.25-0.35 eV for Li+) due to
    Na+'s larger ionic radius creating stronger ion-dipole interactions.
    
    References:
        - Jian et al., Adv. Energy Mater. 2016: 0.5-0.6 eV for Na+ in PC
        - Komaba et al., Chem. Rev. 2014: Na+ desolvation dominates at low T
        - Okoshi et al., J. Electrochem. Soc. 2017: DFT-computed barriers
    
    Note: 0.18 eV (the old value) is the barrier for Na+ migration through
    the SEI layer, which is a different process entirely.
    """
    def __init__(self):
        super().__init__()
        # Initialized at physical value: 0.50 eV (middle of 0.4-0.6 range)
        # Learnable so training can adjust to specific electrolyte systems
        self.base_barrier_eV = nn.Parameter(torch.tensor(0.50))
        self.mn_penalty = nn.Parameter(torch.tensor(0.025))  # Mn3+ distorts intercalation sites
        self.fe_relief = nn.Parameter(torch.tensor(0.014))    # Fe stabilizes local structure
        self.dopant_relief = nn.Parameter(torch.tensor(0.050))  # dopants widen Na+ channels
        self.beta_base = nn.Parameter(torch.tensor(0.48))

    def barrier(self, composition: torch.Tensor, temperature_K: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
        batch = soc.shape[0]
        comp = _as_batch(composition, batch, soc.device)
        T = _as_batch(temperature_K, batch, soc.device)
        mn = torch.clamp(comp[:, 1:2] if comp.shape[1] > 1 else torch.zeros_like(soc), 0.0, 1.5)
        fe = torch.clamp(comp[:, 2:3] if comp.shape[1] > 2 else torch.zeros_like(soc), 0.0, 1.5)
        dopant = torch.clamp(comp[:, 4:5] if comp.shape[1] > 4 else torch.zeros_like(soc), 0.0, 0.25)
        barrier = torch.abs(self.base_barrier_eV) + torch.abs(self.mn_penalty) * mn - torch.abs(self.fe_relief) * fe - torch.abs(self.dopant_relief) * dopant
        thermal = torch.exp(torch.clamp(barrier / (8.617e-5 * T + 1e-10), -2.0, 4.0))
        soc_penalty = 1.0 + 0.25 * torch.relu(soc - 0.85)
        return torch.clamp(thermal * soc_penalty, 0.2, 30.0)

    def dynamic_beta(self, composition: torch.Tensor, temperature_K: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
        desolv = self.barrier(composition, temperature_K, soc)
        beta = torch.clamp(self.beta_base, 0.25, 0.70) - 0.035 * torch.log1p(desolv) + 0.025 * torch.clamp(soc - 0.5, -0.5, 0.5)
        return torch.clamp(beta, 0.25, 0.75)


class NaIonDegradationPhysics(nn.Module):
    """Composite Na-ion degradation physics module.
    
    Combines JT coupling, P2-O2 phase transition, and desolvation barrier
    into a single differentiable vector field for the UDE.
    
    The C-rate (charging/discharging current relative to capacity) is now
    a required parameter rather than hardcoded, because:
    - SOC dynamics (dx/dt) depend directly on I_app
    - Higher C-rates cause faster Na+ extraction → different phase behavior
    - The P2→O2 transition onset depends on how quickly x_Na drops
    """
    def __init__(self, Q_nominal_mAh: float = 130.0):
        super().__init__()
        self.jahn_teller = JahnTellerCoupling()
        self.phase_transition = P2O2PhaseTransition()
        self.desolvation = NaDesolvationBarrier()
        self.sei_scale = nn.Parameter(torch.tensor(1.0))
        self.phase_capacity_scale = nn.Parameter(torch.tensor(1.0))
        self.jt_capacity_scale = nn.Parameter(torch.tensor(7.5e-4))
        self.desolvation_capacity_scale = nn.Parameter(torch.tensor(2.5e-4))
        # Nominal capacity in mAh — used to compute I_app from C-rate
        # For typical P2-type NaMnO2 cathodes: 100-160 mAh/g
        self.Q_nominal_mAh = Q_nominal_mAh

    def forward(
        self,
        state: torch.Tensor,
        composition: torch.Tensor,
        temperature_K: torch.Tensor,
        t: torch.Tensor,
        base_arrhenius_rate: torch.Tensor,
        c_rate: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        batch = state.shape[0]
        comp = _as_batch(composition, batch, state.device)
        T = _as_batch(temperature_K, batch, state.device)
        k_base = _as_batch(base_arrhenius_rate, batch, state.device)
        Q = torch.clamp(state[..., 0:1], min=1e-6)
        soc = torch.clamp(state[..., 2:3], 0.0, 1.0)
        jt = self.jahn_teller(comp, T, soc)
        p2o2_rate = self.phase_transition(state, comp, T, jt)
        desolv = self.desolvation.barrier(comp, T, soc)
        beta = self.desolvation.dynamic_beta(comp, T, soc)
        sei_rate = torch.abs(self.sei_scale) * k_base
        dQdt = -Q * (
            sei_rate
            + torch.abs(self.phase_capacity_scale) * p2o2_rate
            + torch.abs(self.jt_capacity_scale) * jt
            + torch.abs(self.desolvation_capacity_scale) * torch.log1p(desolv)
        )
        # I_app from C-rate and nominal capacity: I = C_rate * Q_nominal / 3600
        # This makes the SOC dynamics (dx/dt) properly C-rate dependent.
        # At 1C with 130 mAh/g: I_app ≈ 0.036 A/g (36 mA/g)
        # The old hardcoded 0.001 A was ~0.03C — almost no current.
        I_app = torch.ones_like(Q) * (c_rate * self.Q_nominal_mAh / 3600.0)
        # dx/dt: rate of Na extraction from cathode
        # Negative sign: during charging, Na is extracted (x decreases)
        # beta modulates the insertion asymmetry (Butler-Volmer transfer coefficient)
        dxdt = -I_app * (1.0 + 0.2 * (0.5 - beta)) / (Q * 96485.0 + 1e-10)
        dVdt = -0.014 * p2o2_rate - 0.0025 * jt - 0.0007 * torch.log1p(desolv)
        dstate = torch.cat([dQdt, dVdt, dxdt], dim=-1)
        diagnostics = {
            "sei_rate": sei_rate,
            "p2o2_rate": p2o2_rate,
            "jahn_teller_factor": jt,
            "na_desolvation_factor": desolv,
            "dynamic_bv_beta": beta,
            "applied_current_A": I_app,
        }
        return dstate, diagnostics
