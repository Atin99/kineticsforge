import torch
import numpy as np
from modules.bms.sei_kinetics import CoupledSEIModel

class IndianDriveCycle:
    def __init__(self, duration_hrs=8, dt=1.0):
        self.duration = int(duration_hrs * 3600)
        self.dt = dt
        self.n_steps = int(self.duration / dt)
        self.time = np.arange(0, self.duration, dt, dtype=np.float32)
        self.current = np.zeros(self.n_steps, dtype=np.float32)
        self.temperature_ambient = np.zeros(self.n_steps, dtype=np.float32)
        self._generate_profile()

    def _generate_profile(self):
        for i, t in enumerate(self.time):
            t_min = t / 60.0
            if t_min < 90:
                phase_t = t_min % 0.67
                if phase_t < 0.17:
                    self.current[i] = 12.5
                elif phase_t < 0.33:
                    self.current[i] = -7.5
                else:
                    self.current[i] = 0.0
                self.temperature_ambient[i] = 30.0 + 12.0 * (t_min / 90.0)
            elif t_min < 150:
                self.current[i] = 2.5
                self.temperature_ambient[i] = 42.0
            elif t_min < 300:
                self.current[i] = -1.25
                self.temperature_ambient[i] = 42.0 + 3.0 * ((t_min - 150) / 150.0)
            elif t_min < 390:
                phase_t = (t_min - 300) % 0.67
                if phase_t < 0.17:
                    self.current[i] = 12.5
                elif phase_t < 0.33:
                    self.current[i] = -7.5
                else:
                    self.current[i] = 0.0
                self.temperature_ambient[i] = 40.0 - 5.0 * ((t_min - 300) / 90.0)
            else:
                soc_proxy = (t_min - 390) / 90.0
                self.current[i] = -0.5 * (1.0 - min(soc_proxy, 0.99))
                self.temperature_ambient[i] = 35.0
        self.current += np.random.normal(0, 0.05, self.n_steps).astype(np.float32)
        self.temperature_ambient += np.random.normal(0, 2.0, self.n_steps).astype(np.float32)

    def get_tensors(self):
        return torch.from_numpy(self.time), torch.from_numpy(self.current), torch.from_numpy(self.temperature_ambient)


class FailureInjector:
    def __init__(self, failure_type='sei', target_cell=2, onset_fraction=0.5):
        self.failure_type = failure_type
        self.target_cell = target_cell
        self.onset_fraction = onset_fraction

    def inject(self, t_step, n_steps, cell_states, n_cells):
        progress = t_step / max(n_steps, 1)
        if progress < self.onset_fraction:
            return cell_states
        severity = (progress - self.onset_fraction) / (1.0 - self.onset_fraction + 1e-8)
        if self.failure_type == 'sei':
            cell_states[self.target_cell, 0] += 1.5e-5 * severity
        elif self.failure_type == 'dendrite':
            if severity > 0.5:
                cell_states[self.target_cell, 1] += 0.025 * (severity - 0.5)
        elif self.failure_type == 'thermal_cascade':
            cell_states[self.target_cell, 2] += 2.0 * severity
            for nb in self._neighbors(self.target_cell, n_cells):
                cell_states[nb, 2] += 0.45 * severity
        cell_states[:, 0] = torch.clamp(cell_states[:, 0], 1e-10, 5e-2)
        cell_states[:, 1] = torch.clamp(cell_states[:, 1], 0.0, 2.0)
        cell_states[:, 2] = torch.clamp(cell_states[:, 2], 260.0, 620.0)
        return cell_states

    def _neighbors(self, cid, n):
        nbs = []
        if cid % 4 > 0: nbs.append(cid - 1)
        if cid % 4 < 3 and cid + 1 < n: nbs.append(cid + 1)
        if cid + 4 < n: nbs.append(cid + 4)
        if cid - 4 >= 0: nbs.append(cid - 4)
        return nbs


class PackSimulator:
    def __init__(self, n_cells=8):
        self.n_cells = n_cells
        self.cell_capacity = 2.5

    def ocv_from_soc(self, soc):
        s = np.clip(soc, 0.01, 0.99)
        return 3.0 + 1.2 * s - 0.3 * s**2 + 0.1 * np.log(s / (1 - s + 1e-10))

    def simulate(self, drive_cycle, sei_model, failure_injector=None, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        time_arr, current_arr, T_amb_arr = drive_cycle.get_tensors()
        n_steps = len(time_arr)
        hist = {k: np.zeros((n_steps, self.n_cells), dtype=np.float32)
                for k in ['V_cells','T_cells','SOC_cells','risk','L_sei','P_dendrite','R_int']}
        hist['time'] = time_arr.numpy()
        hist['I_pack'] = current_arr.numpy()
        hist['alert_times'] = []
        hist['alert_cells'] = []
        state = sei_model.init_state(T_ambient=T_amb_arr[0].item() + 273.15)
        consec = np.zeros(self.n_cells, dtype=int)
        for step in range(n_steps):
            T_amb = T_amb_arr[step].item() + 273.15
            I_per_cell = current_arr[step].item() / 2.0
            I_cells = torch.ones(self.n_cells) * I_per_cell
            soc_np = state[:, 3].detach().numpy()
            V_cells = torch.tensor([self.ocv_from_soc(s) for s in soc_np])
            state_out, R_int = sei_model.step(I_cells, V_cells, T_amb)
            if failure_injector:
                sd = state_out.detach().clone()
                sd = failure_injector.inject(step, n_steps, sd, self.n_cells)
                sei_model.state = sd
                state_out = sd
            risk = sei_model.compute_risk_score().detach().numpy()
            hist['V_cells'][step] = V_cells.detach().numpy()
            hist['T_cells'][step] = state_out[:, 2].detach().numpy()
            hist['SOC_cells'][step] = state_out[:, 3].detach().numpy()
            hist['risk'][step] = risk
            hist['L_sei'][step] = state_out[:, 0].detach().numpy()
            hist['P_dendrite'][step] = state_out[:, 1].detach().numpy()
            hist['R_int'][step] = R_int.detach().numpy()
            for c in range(self.n_cells):
                if risk[c] > 0.75:
                    consec[c] += 1
                    if consec[c] >= 3 and c not in hist['alert_cells']:
                        hist['alert_times'].append(step)
                        hist['alert_cells'].append(c)
                else:
                    consec[c] = 0
        return hist


def run_drive_cycle(seed=42, inject_failure=False, n_cells=8, duration_seconds=28800):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cycle = IndianDriveCycle(duration_hrs=duration_seconds/3600.0)
    sei = CoupledSEIModel(n_cells=n_cells)
    fi = None
    if inject_failure:
        ftypes = ['sei', 'dendrite', 'thermal_cascade']
        fi = FailureInjector(ftypes[seed % 3], seed % n_cells, 0.5)
    pack = PackSimulator(n_cells)
    hist = pack.simulate(cycle, sei, fi, seed)
    alert_fired = len(hist['alert_times']) > 0
    lead_time_min = 0
    if alert_fired:
        remaining = len(hist['time']) - hist['alert_times'][0]
        lead_time_min = remaining / 60.0
    return {'history': hist, 'alert_fired': alert_fired, 'lead_time_min': lead_time_min,
            'n_alerts': len(hist['alert_times']), 'alert_cells': hist['alert_cells']}
