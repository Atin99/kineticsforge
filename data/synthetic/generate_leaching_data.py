import torch
import numpy as np
import os
import multiprocessing as mp
from modules.recycling.leaching_ode import MultiMechanismLeaching, ParticleSizeDistribution

class LeachingDataGenerator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.psd = ParticleSizeDistribution(10, 10.0, 100.0)
        self.model = MultiMechanismLeaching(3, self.psd)

    def generate_condition(self, args):
        idx, temp, ph, c_acid, particle_size = args
        steps = 180
        dt = 1.0
        alpha = torch.zeros(3)
        t_curr = 0.0
        traj = []
        for _ in range(steps):
            da_dt = self.model(t_curr, alpha, torch.tensor(temp), torch.tensor(ph), torch.tensor(c_acid))
            alpha = alpha + dt * da_dt
            noise = torch.randn_like(alpha) * 0.02
            alpha_noisy = torch.clamp(alpha + noise, 0.0, 1.0)
            traj.append(alpha_noisy.detach().numpy())
            t_curr += dt
        traj = np.stack(traj)
        np.savez(os.path.join(self.out_dir, f'leaching_run_{idx}.npz'), alpha=traj, temp=temp, ph=ph, c_acid=c_acid, particle_size=particle_size)

    def run_grid(self, num_workers=4):
        temps = [323.15, 333.15, 343.15, 353.15, 363.15]
        phs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        concs = [0.5, 1.5, 3.0]
        sizes = [10.0, 50.0, 100.0]
        args_list = []
        idx = 0
        for t in temps:
            for p in phs:
                for c in concs:
                    for s in sizes:
                        args_list.append((idx, t, p, c, s))
                        idx += 1
        with mp.Pool(num_workers) as pool:
            pool.map(self.generate_condition, args_list)

if __name__ == '__main__':
    gen = LeachingDataGenerator('data/synthetic/leaching')
    gen.run_grid(num_workers=min(2, mp.cpu_count()))
