import torch
import numpy as np
from modules.bms.sei_kinetics import CoupledSEIModel
from modules.bms.drive_cycle_sim import IndianDriveCycle, PackSimulator, FailureInjector

class PrecursorFeatureExtractor:
    def __init__(self, window_size=60, window_sizes=None):
        if window_sizes is None:
            window_sizes = (30, 60, 120, 240)
        if window_size not in window_sizes:
            window_sizes = tuple(sorted(set(tuple(window_sizes) + (window_size,))))
        self.windows = tuple(int(w) for w in window_sizes)
        self.window = max(self.windows)

    def extract(self, history, step):
        features = {}
        for c in range(history['risk'].shape[1]):
            cell_features = {
                'P_dendrite': float(history['P_dendrite'][step, c]),
                'L_sei': float(history['L_sei'][step, c])
            }
            risk_votes = []
            slope_votes = []
            for window in self.windows:
                start = max(0, step - window)
                risk_window = history['risk'][start:step+1, c]
                T_window = history['T_cells'][start:step+1, c]
                R_window = history['R_int'][start:step+1, c]
                risk_slope = self._slope(risk_window)
                T_slope = self._slope(T_window)
                R_slope = self._slope(R_window)
                cell_features.update({
                    f'risk_mean_w{window}': float(np.mean(risk_window)),
                    f'risk_max_w{window}': float(np.max(risk_window)),
                    f'risk_slope_w{window}': float(risk_slope),
                    f'T_mean_w{window}': float(np.mean(T_window)),
                    f'T_max_w{window}': float(np.max(T_window)),
                    f'T_slope_w{window}': float(T_slope),
                    f'R_mean_w{window}': float(np.mean(R_window)),
                    f'R_slope_w{window}': float(R_slope),
                })
                risk_votes.append(float(np.max(risk_window)))
                slope_votes.append(float(0.5 * risk_slope + 0.3 * T_slope + 0.2 * R_slope))
            weights = np.linspace(1.0, 2.0, len(self.windows))
            weights = weights / weights.sum()
            cell_features['risk_multiscale'] = float(np.dot(weights, risk_votes))
            cell_features['precursor_slope_multiscale'] = float(np.dot(weights, slope_votes))
            features[c] = cell_features
        return features

    @staticmethod
    def _slope(values):
        values = np.asarray(values, dtype=float)
        if len(values) <= 1:
            return 0.0
        return float(np.polyfit(np.arange(len(values)), values, 1)[0])


class AlertEngine:
    def __init__(self, risk_threshold=0.75, consecutive_required=3, cooldown=300, uncertainty_weight=0.5):
        self.threshold = risk_threshold
        self.consecutive = consecutive_required
        self.cooldown = cooldown
        self.uncertainty_weight = uncertainty_weight

    def process_history(self, history):
        n_steps = history['risk'].shape[0]
        n_cells = history['risk'].shape[1]
        alerts = []
        consec = np.zeros(n_cells, dtype=int)
        last_alert_step = np.full(n_cells, -self.cooldown - 1)
        risk_uncertainty = history.get('risk_uncertainty')
        for step in range(n_steps):
            for c in range(n_cells):
                threshold = self.threshold
                if risk_uncertainty is not None:
                    threshold = max(0.35, self.threshold - self.uncertainty_weight * float(risk_uncertainty[step, c]))
                if history['risk'][step, c] > threshold:
                    consec[c] += 1
                    if consec[c] >= self.consecutive and (step - last_alert_step[c]) > self.cooldown:
                        severity = 'CRITICAL' if history['risk'][step, c] > 0.9 else 'WARNING'
                        T_val = history['T_cells'][step, c]
                        action = 'REDUCE CHARGE RATE' if T_val > 330 else 'MONITOR CLOSELY'
                        failure_step = self._failure_step(history, c, n_steps)
                        alerts.append({
                            'step': step,
                            'time_s': float(history['time'][step]),
                            'cell': c,
                            'risk': float(history['risk'][step, c]),
                            'threshold': float(threshold),
                            'severity': severity,
                            'T': float(T_val),
                            'action': action,
                            'eta_min': float(max(failure_step - step, 0) / 60.0),
                            'lead_steps': int(failure_step - step)
                        })
                        last_alert_step[c] = step
                else:
                    consec[c] = 0
        return alerts

    @staticmethod
    def _failure_step(history, cell, default):
        if 'failure_step' in history:
            value = history['failure_step']
            if np.ndim(value) == 0:
                return int(value)
            if len(value) > cell:
                return int(value[cell])
        if 'failed' in history and bool(np.asarray(history['failed']).any()):
            temps = history.get('T_cells')
            if temps is not None:
                idx = np.where(temps[:, cell] > 360.0)[0]
                if len(idx):
                    return int(idx[0])
        return int(default)


def run_drive_cycle(seed=42, inject_failure=False, n_cells=8, duration_seconds=28800):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cycle = IndianDriveCycle(duration_hrs=duration_seconds / 3600.0)
    sei = CoupledSEIModel(n_cells=n_cells)
    fi = None
    if inject_failure:
        ftypes = ['sei', 'dendrite', 'thermal_cascade']
        fi = FailureInjector(ftypes[seed % 3], seed % n_cells, 0.35)
    pack = PackSimulator(n_cells)
    history = pack.simulate(cycle, sei, fi, seed)
    engine = AlertEngine()
    alerts = engine.process_history(history)
    alert_fired = len(alerts) > 0
    lead_time_min = 0
    if alert_fired:
        remaining = len(history['time']) - alerts[0]['step']
        lead_time_min = remaining / 60.0
    return {
        'history': history,
        'alerts': alerts,
        'alert_fired': alert_fired,
        'lead_time_min': lead_time_min,
        'n_alerts': len(alerts),
        'alert_cells': list(set(a['cell'] for a in alerts))
    }
