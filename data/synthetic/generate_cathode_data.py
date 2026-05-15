import torch
import numpy as np
import os
import multiprocessing as mp
from core.physics_constraints import KineticConstraints
from core.neural_ode import DOPRI5

class HeteroscedasticNoiseModel:
    def __init__(self, base_noise, cycle_scaling, temp_scaling):
        self.base_noise = base_noise
        self.cycle_scaling = cycle_scaling
        self.temp_scaling = temp_scaling
    def sample(self, q, cycle, temp):
        std = self.base_noise * q + self.cycle_scaling * cycle + self.temp_scaling * (temp - 298.15)
        return torch.randn_like(q) * std

class AutoRegressiveNoise:
    def __init__(self, phi, std):
        self.phi = phi
        self.std = std
        self.prev = 0.0
    def sample(self, size):
        eps = torch.randn(size) * self.std
        out = torch.zeros(size)
        out[0] = eps[0]
        for i in range(1, size):
            out[i] = self.phi * out[i-1] + eps[i]
        return out

class SyntheticCathodeGenerator:
    def __init__(self):
        self.k0 = 1e-4
        self.Ea = 0.6
        self.solver = DOPRI5()
    def base_capacity(self, comp):
        na, mn, fe, dopant, frac = comp
        q0 = 120 + 40*mn - 20*fe + 15*(dopant == 1.0)
        return q0 + np.random.normal(0, 8)
    def degradation_ode(self, t, state, temp, k_fade):
        q = state[0]
        v = state[1]
        dq = -k_fade * q * (0.01 + 0.001 * q**2)
        dv = 0.001 * v * torch.exp(-q / 100.0)
        return torch.stack([dq, dv])
    def simulate_composition(self, comp, temp, cycles):
        q0 = self.base_capacity(comp)
        v0 = 3.8
        state = torch.tensor([q0, v0], dtype=torch.float32)
        k_fade = KineticConstraints.generalized_arrhenius(torch.tensor(temp), self.k0, self.Ea)
        traj = [state.clone()]
        dt = 3600.0 * 24.0
        t_curr = 0.0
        for _ in range(cycles):
            state, _, _, _ = self.solver.step(self.degradation_ode, t_curr, state, dt, args=(temp, k_fade))
            t_curr += dt
            traj.append(state.clone())
        return torch.stack(traj)
    def generate_single(self, args):
        idx, comp, temp, out_dir = args
        clean_traj = self.simulate_composition(comp, temp, 500)
        q = clean_traj[:, 0]
        v = clean_traj[:, 1]
        noise_model = HeteroscedasticNoiseModel(0.01, 0.0001, 0.005)
        ar_noise = AutoRegressiveNoise(0.8, 0.005)
        noisy_q = q + noise_model.sample(q, torch.arange(501), temp)
        noisy_v = v + ar_noise.sample(501)
        drop_mask = torch.rand(501) > 0.05
        noisy_q = noisy_q[drop_mask]
        noisy_v = noisy_v[drop_mask]
        cycles = torch.arange(501)[drop_mask]
        np.savez(os.path.join(out_dir, f'cathode_{idx}.npz'), cycles=cycles.numpy(), capacity=noisy_q.numpy(), voltage=noisy_v.numpy(), comp=comp, temp=temp)
    def run_batch(self, compositions, temps, out_dir, num_workers=4):
        os.makedirs(out_dir, exist_ok=True)
        args_list = []
        idx = 0
        for c in compositions:
            for t in temps:
                args_list.append((idx, c, t, out_dir))
                idx += 1
        with mp.Pool(num_workers) as pool:
            pool.map(self.generate_single, args_list)

def main():
    comps = []
    for na in np.linspace(0.9, 1.1, 5):
        for mn in np.linspace(0.2, 0.8, 10):
            fe = 1.0 - mn
            for dopant in [0, 1, 2, 3]:
                frac = 0.05 if dopant > 0 else 0.0
                comps.append([na, mn, fe, dopant, frac])
    np.random.shuffle(comps)
    selected = comps[:100]
    temps = [298.15, 313.15, 318.15, 323.15]
    gen = SyntheticCathodeGenerator()
    gen.run_batch(selected, temps, 'data/synthetic/cathode', num_workers=min(2, mp.cpu_count()))

if __name__ == '__main__':
    main()
