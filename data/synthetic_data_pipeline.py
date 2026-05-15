import os
import numpy as np
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

SYNTH_DIR = Path(__file__).resolve().parent / "synthetic"

class CathodePhysicsGenerator:
    def __init__(self):
        self.R = 8.314
        self.F = 96485.0
        self.save_dir = SYNTH_DIR / "cathode"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def composition_grid(self, n_na=5, n_mn=10, dopants=(None, 'Al', 'Ti', 'Mg')):
        comps = []
        for Na in np.linspace(0.9, 1.1, n_na):
            for Mn_frac in np.linspace(0.2, 0.8, n_mn):
                Fe_frac = 1.0 - Mn_frac
                for dopant in dopants:
                    comps.append({
                        'Na': Na, 'Mn': Mn_frac, 'Fe': Fe_frac,
                        'dopant': dopant, 'dopant_frac': 0.05 if dopant else 0.0
                    })
        return comps

    def initial_capacity(self, comp):
        q0 = 120.0 + 40.0 * comp['Mn'] - 20.0 * comp['Fe']
        if comp['dopant'] == 'Al': q0 += 15
        elif comp['dopant'] == 'Ti': q0 += 8
        elif comp['dopant'] == 'Mg': q0 += 5
        q0 *= (0.95 + 0.1 * comp['Na'])
        q0 += np.random.normal(0, 8)
        return max(q0, 60.0)

    def arrhenius_rate(self, T, Ea, k0):
        return k0 * np.exp(-Ea * self.F / (self.R * T))

    def ocv_curve(self, x, comp):
        a0 = 3.5 + 0.3 * comp['Mn'] - 0.1 * comp['Fe']
        a1 = -0.5 * comp['Mn']
        a2 = 0.15 * comp['Fe']
        return a0 + a1 * x + a2 * x**2 - 0.02 * np.log(x + 1e-8) + 0.02 * np.log(1 - x + 1e-8)

    def sei_growth(self, t, T, k0_sei=1e-4, Ea_sei=0.4):
        k = self.arrhenius_rate(T, Ea_sei, k0_sei)
        return k * np.sqrt(t + 1)

    def capacity_fade_ode(self, Q, t, T, comp, k0=1e-4, Ea=0.6):
        k_fade = self.arrhenius_rate(T, Ea, k0)
        dopant_factor = 1.0
        if comp['dopant'] == 'Al': dopant_factor = 0.85
        elif comp['dopant'] == 'Ti': dopant_factor = 0.92
        elif comp['dopant'] == 'Mg': dopant_factor = 0.95
        Mn_dissolution = 0.001 * comp['Mn']**2 * np.exp(-0.3 * self.F / (self.R * T))
        cracking = 0.0005 * (Q / self.initial_capacity(comp))**1.5
        return -k_fade * Q * dopant_factor - Mn_dissolution * Q - cracking * Q

    def simulate_cycling(self, comp, T=318.0, n_cycles=500, dt=1.0):
        q0 = self.initial_capacity(comp)
        Q = q0
        capacity = np.zeros(n_cycles)
        voltage_curves = np.zeros((n_cycles, 100))
        resistance = np.zeros(n_cycles)
        sei_thickness = np.zeros(n_cycles)
        temperature = np.ones(n_cycles) * T

        for c in range(n_cycles):
            capacity[c] = Q + np.random.normal(0, 0.03 * Q)
            x_range = np.linspace(0.05, 0.95, 100)
            R_int = 0.01 + 0.0001 * c + 0.005 * self.sei_growth(c, T)
            I_discharge = 0.5
            voltage_curves[c] = self.ocv_curve(x_range, comp) - I_discharge * R_int + np.random.normal(0, 0.010, 100)
            resistance[c] = R_int
            sei_thickness[c] = self.sei_growth(c, T)
            dQ = self.capacity_fade_ode(Q, c, T, comp)
            Q = max(Q + dQ * dt, q0 * 0.3)
            T_cycle = T + np.random.normal(0, 2)
            temperature[c] = T_cycle

        missing = np.random.choice(n_cycles, size=int(0.05 * n_cycles), replace=False)
        capacity[missing] = np.nan

        return {
            'cycles': np.arange(n_cycles),
            'capacity': capacity,
            'voltage_curves': voltage_curves,
            'resistance': resistance,
            'sei_thickness': sei_thickness,
            'temperature': temperature,
            'composition': comp
        }

    def generate_all(self, n_compositions=100, temperatures=(308, 313, 318, 323, 328)):
        comps = self.composition_grid()
        if len(comps) > n_compositions:
            indices = np.random.choice(len(comps), n_compositions, replace=False)
            comps = [comps[i] for i in indices]

        log.info(f"Generating cathode cycling data for {len(comps)} compositions at {len(temperatures)} temperatures...")
        total = 0
        for i, comp in enumerate(comps):
            for T in temperatures:
                data = self.simulate_cycling(comp, T=T)
                fname = f"cathode_comp{i:03d}_T{int(T)}.npz"
                np.savez(self.save_dir / fname,
                         cycles=data['cycles'], capacity=data['capacity'],
                         voltage_curves=data['voltage_curves'], resistance=data['resistance'],
                         sei_thickness=data['sei_thickness'], temperature=data['temperature'],
                         Na=comp['Na'], Mn=comp['Mn'], Fe=comp['Fe'],
                         dopant=str(comp['dopant']), dopant_frac=comp['dopant_frac'])
                total += 1

            if (i + 1) % 20 == 0:
                log.info(f"  Generated {total} files ({i+1}/{len(comps)} compositions)")

        log.info(f"Total cathode files generated: {total}")

class BMSPhysicsGenerator:
    def __init__(self, n_cells=8):
        self.n_cells = n_cells
        self.R = 8.314
        self.F = 96485.0
        self.save_dir = SYNTH_DIR / "bms"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def drive_cycle_profile(self, duration=28800):
        I = np.zeros(duration)
        T_amb = np.ones(duration) * 303.15
        phase_1_end = 5400
        phase_2_end = 9000
        phase_3_end = 18000
        phase_4_end = 23400

        for t in range(phase_1_end):
            cycle_pos = t % 40
            if cycle_pos < 10: I[t] = 5.0 * (2 * np.random.rand() - 1)
            T_amb[t] = 303.15 + (315.15 - 303.15) * t / phase_1_end

        for t in range(phase_1_end, phase_2_end):
            I[t] = 1.0 + 0.1 * np.random.randn()
            T_amb[t] = 315.15

        for t in range(phase_2_end, phase_3_end):
            I[t] = -0.5
            T_amb[t] = 318.15

        for t in range(phase_3_end, phase_4_end):
            cycle_pos = t % 40
            if cycle_pos < 10: I[t] = 5.0 * (2 * np.random.rand() - 1)
            T_amb[t] = 318.15 - (318.15 - 308.15) * (t - phase_3_end) / (phase_4_end - phase_3_end)

        for t in range(phase_4_end, duration):
            frac = (t - phase_4_end) / (duration - phase_4_end)
            if frac < 0.7: I[t] = -0.2
            else: I[t] = -0.2 * np.exp(-3 * (frac - 0.7) / 0.3)
            T_amb[t] = 308.15 - 5 * frac

        I += 0.05 * np.random.randn(duration)
        T_amb += 2 * np.random.randn(duration)
        return I, T_amb

    def simulate_pack(self, I_profile, T_amb, failure_type=None, fail_cell=None, degradation_pct=0.0):
        duration = len(I_profile)
        V = np.zeros((duration, self.n_cells))
        T = np.zeros((duration, self.n_cells))
        SOC = np.ones(self.n_cells) * 0.8
        L_sei = np.ones(self.n_cells) * 1e-9
        R_int = np.ones(self.n_cells) * 0.01
        risk = np.zeros((duration, self.n_cells))

        for t in range(duration):
            I_cell = I_profile[t] / 2.0
            for c in range(self.n_cells):
                SOC[c] -= I_cell * 1.0 / (3600 * 5.0 * (1 - degradation_pct))
                SOC[c] = np.clip(SOC[c], 0.05, 0.95)
                OCV = 3.0 + 1.2 * SOC[c] - 0.3 * SOC[c]**2
                V[t, c] = OCV - I_cell * R_int[c] + np.random.normal(0, 0.005)
                T[t, c] = T_amb[t] + I_cell**2 * R_int[c] * 50
                k_sei = 1e-12 * np.exp(-0.4 * self.F / (self.R * T[t, c]))
                L_sei[c] += k_sei / (2 * L_sei[c] + 1e-12)
                R_int[c] = 0.01 * (1 + degradation_pct) + 100 * L_sei[c]

            if failure_type == 'sei' and fail_cell is not None and t > duration * 0.3:
                L_sei[fail_cell] *= 1.0001
                R_int[fail_cell] = 0.01 + 100 * L_sei[fail_cell]
            elif failure_type == 'dendrite' and fail_cell is not None and t > duration * 0.5:
                spike = 0.01 * np.exp(0.0005 * (t - duration * 0.5))
                R_int[fail_cell] += spike
            elif failure_type == 'thermal' and fail_cell is not None and t > duration * 0.4:
                T[t, fail_cell] += 0.005 * (t - duration * 0.4)
                for neighbor in self._get_neighbors(fail_cell):
                    T[t, neighbor] += 0.001 * (t - duration * 0.4)

            for c in range(self.n_cells):
                dR = R_int[c] - 0.01
                dT = T[t, c] - 298.15
                dV = (V[t, c] - V[t-1, c]) if t > 0 else 0
                risk[t, c] = 1.0 / (1 + np.exp(-(0.3 * dR * 1000 + 0.01 * dT + 0.2 * L_sei[c] * 1e6 - 2.0)))

        return {'V': V, 'T': T, 'risk': risk, 'I': I_profile, 'T_amb': T_amb}

    def _get_neighbors(self, cell_idx):
        adjacency = {
            0: [1, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [2, 7],
            4: [0, 5], 5: [1, 4, 6], 6: [2, 5, 7], 7: [3, 6]
        }
        return adjacency.get(cell_idx, [])

    def generate_all(self, n_normal=35, n_failure=15):
        log.info(f"Generating {n_normal + n_failure} BMS drive cycle datasets...")
        total = 0
        for i in range(n_normal):
            I, T_amb = self.drive_cycle_profile()
            deg = np.random.uniform(0, 0.1)
            data = self.simulate_pack(I, T_amb, degradation_pct=deg)
            np.savez(self.save_dir / f"bms_normal_{i:03d}.npz",
                     V=data['V'], T=data['T'], risk=data['risk'],
                     I=data['I'], T_amb=data['T_amb'],
                     failure_type='none', fail_cell=-1)
            total += 1

        failure_types = ['sei', 'dendrite', 'thermal']
        for i in range(n_failure):
            I, T_amb = self.drive_cycle_profile()
            ftype = failure_types[i % 3]
            fcell = np.random.randint(0, self.n_cells)
            data = self.simulate_pack(I, T_amb, failure_type=ftype, fail_cell=fcell)
            np.savez(self.save_dir / f"bms_failure_{i:03d}.npz",
                     V=data['V'], T=data['T'], risk=data['risk'],
                     I=data['I'], T_amb=data['T_amb'],
                     failure_type=ftype, fail_cell=fcell)
            total += 1

        log.info(f"Total BMS files generated: {total}")

class LeachingPhysicsGenerator:
    def __init__(self):
        self.R = 8.314
        self.F = 96485.0
        self.save_dir = SYNTH_DIR / "leaching"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_grid(self):
        temps = [323, 333, 343, 353, 363]
        pHs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        concs = [0.5, 1.5, 3.0]
        particles = [10e-6, 50e-6, 100e-6]

        log.info(f"Generating leaching data: {len(temps)}x{len(pHs)}x{len(concs)}x{len(particles)} = {len(temps)*len(pHs)*len(concs)*len(particles)} conditions...")

        all_data = []
        for T in temps:
            for pH in pHs:
                for c_acid in concs:
                    for r0 in particles:
                        alpha = self._simulate_leaching(T, pH, c_acid, r0)
                        all_data.append({
                            'T': T, 'pH': pH, 'c_acid': c_acid, 'r0': r0,
                            'alpha_Mn': alpha[0], 'alpha_Fe': alpha[1], 'alpha_Na': alpha[2]
                        })

        alpha_grid = np.zeros((len(all_data), 3, 180))
        for idx, cond in enumerate(all_data):
            alpha_grid[idx] = self._simulate_leaching_trajectory(
                cond['T'], cond['pH'], cond['c_acid'], cond['r0'])

        np.savez(self.save_dir / "leaching_grid.npz",
                 conditions=all_data,
                 alpha_trajectories=alpha_grid,
                 time_minutes=np.arange(180))
        log.info(f"Saved {len(all_data)} leaching conditions with trajectories")

    def _simulate_leaching(self, T, pH, c_acid, r0, duration=180):
        alpha = np.zeros(3)
        D0 = [1e-12, 8e-13, 2e-12]
        for step in range(duration):
            for s in range(3):
                D_eff = D0[s] * np.exp(-0.35 * self.F / (self.R * T))
                if alpha[s] < 0.999:
                    da = (3 * D_eff * c_acid * 1000) / (r0**2 * 5000 * max((1 - alpha[s])**(1/3), 1e-6))
                    alpha[s] = min(alpha[s] + da * 60, 1.0)
        return alpha

    def _simulate_leaching_trajectory(self, T, pH, c_acid, r0, duration=180):
        alpha = np.zeros((3, duration))
        state = np.zeros(3)
        D0 = [1e-12, 8e-13, 2e-12]
        for step in range(duration):
            for s in range(3):
                D_eff = D0[s] * np.exp(-0.35 * self.F / (self.R * T))
                if state[s] < 0.999:
                    da = (3 * D_eff * c_acid * 1000) / (r0**2 * 5000 * max((1 - state[s])**(1/3), 1e-6))
                    state[s] = min(state[s] + da * 60, 1.0)
            alpha[:, step] = state + np.random.normal(0, 0.02, 3)
            alpha[:, step] = np.clip(alpha[:, step], 0, 1)
        return alpha

class MasterSyntheticPipeline:
    def __init__(self):
        self.cathode_gen = CathodePhysicsGenerator()
        self.bms_gen = BMSPhysicsGenerator()
        self.leaching_gen = LeachingPhysicsGenerator()

    def run_all(self):
        log.info("=" * 60)
        log.info("MASTER SYNTHETIC DATA PIPELINE")
        log.info("=" * 60)
        self.cathode_gen.generate_all(n_compositions=100, temperatures=(308, 313, 318, 323, 328))
        self.bms_gen.generate_all(n_normal=35, n_failure=15)
        self.leaching_gen.generate_grid()
        self._print_summary()

    def _print_summary(self):
        total_files = 0
        total_bytes = 0
        for f in SYNTH_DIR.rglob("*"):
            if f.is_file():
                total_files += 1
                total_bytes += f.stat().st_size
        log.info(f"Total synthetic files: {total_files}")
        log.info(f"Total synthetic data size: {total_bytes / 1e6:.1f} MB")

if __name__ == "__main__":
    pipeline = MasterSyntheticPipeline()
    pipeline.run_all()
