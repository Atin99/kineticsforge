import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class TelemetryFrame:
    t_s: float
    voltage_V: float
    current_A: float
    temperature_K: float
    ambient_K: float
    cell_id: int = 0


@dataclass
class TwinState:
    soc: float = 0.80
    soh: float = 1.00
    sei_nm: float = 12.0
    dendrite_index: float = 0.0
    resistance_mohm: float = 18.0
    core_temperature_K: float = 308.0
    risk: float = 0.0


@dataclass
class AssimilationConfig:
    nominal_capacity_Ah: float = 3.0
    thermal_mass_J_K: float = 920.0
    cooling_W_K: float = 0.42
    entropic_heat_W_A: float = 0.035
    process_noise: float = 0.015
    voltage_noise_V: float = 0.018
    temperature_noise_K: float = 1.8
    risk_threshold: float = 0.75
    required_consecutive_alerts: int = 3


class BatteryDigitalTwin:
    def __init__(self, config: Optional[AssimilationConfig] = None):
        self.config = config or AssimilationConfig()
        self.state = TwinState()
        self.cov = np.diag([0.04, 0.01, 6.0, 0.02, 4.0, 6.0, 0.04])
        self.alert_streak = 0

    def reset(self, state: Optional[TwinState] = None) -> None:
        self.state = state or TwinState()
        self.cov = np.diag([0.04, 0.01, 6.0, 0.02, 4.0, 6.0, 0.04])
        self.alert_streak = 0

    def vector(self) -> np.ndarray:
        s = self.state
        return np.array([s.soc, s.soh, s.sei_nm, s.dendrite_index, s.resistance_mohm, s.core_temperature_K, s.risk], dtype=float)

    def set_vector(self, x: Sequence[float]) -> None:
        self.state = TwinState(
            soc=float(np.clip(x[0], 0.0, 1.0)),
            soh=float(np.clip(x[1], 0.0, 1.2)),
            sei_nm=float(np.clip(x[2], 1.0, 500.0)),
            dendrite_index=float(np.clip(x[3], 0.0, 1.0)),
            resistance_mohm=float(np.clip(x[4], 1.0, 500.0)),
            core_temperature_K=float(np.clip(x[5], 250.0, 520.0)),
            risk=float(np.clip(x[6], 0.0, 1.0)),
        )

    def predict(self, frame: TelemetryFrame, dt_s: float) -> None:
        c = self.config
        x = self.vector()
        soc, soh, sei, dend, r_mohm, temp, risk = x
        current = float(frame.current_A)
        ambient = float(frame.ambient_K)
        throughput_Ah = abs(current) * max(dt_s, 0.0) / 3600.0
        c_rate = abs(current) / max(c.nominal_capacity_Ah, 1e-9)
        soc_delta = -current * dt_s / (3600.0 * c.nominal_capacity_Ah * max(soh, 0.2))
        temp_factor = math.exp(min(4.0, max(-4.0, (temp - 298.15) / 22.0)))
        sei_growth = 0.0025 * temp_factor * math.sqrt(max(throughput_Ah, 1e-9))
        plating_drive = max(0.0, c_rate - 1.35) * max(0.0, 293.15 - temp) / 35.0
        dend_growth = 0.010 * plating_drive * max(dt_s, 0.0) / 60.0
        ohmic_heat = (current ** 2) * (r_mohm / 1000.0)
        reversible_heat = c.entropic_heat_W_A * current
        cooling = c.cooling_W_K * (temp - ambient)
        temp_delta = (ohmic_heat + reversible_heat - cooling) * dt_s / max(c.thermal_mass_J_K, 1e-9)
        soh_loss = 0.00002 * throughput_Ah * temp_factor + 0.00008 * dend_growth
        r_next = 14.0 + 0.32 * sei + 42.0 * dend + 18.0 * (1.0 - soh)
        risk_next = self._risk_model(soc + soc_delta, soh - soh_loss, sei + sei_growth, dend + dend_growth, r_next, temp + temp_delta, c_rate)
        x_next = np.array(
            [
                soc + soc_delta,
                soh - soh_loss,
                sei + sei_growth,
                dend + dend_growth,
                r_next,
                temp + temp_delta,
                risk_next,
            ],
            dtype=float,
        )
        q = c.process_noise
        self.cov = self.cov + np.diag([q * 0.15, q * 0.03, q * 12.0, q * 0.08, q * 8.0, q * 16.0, q * 0.12])
        self.set_vector(x_next)

    def update(self, frame: TelemetryFrame) -> Dict[str, float]:
        x = self.vector()
        z = np.array([frame.voltage_V, frame.temperature_K], dtype=float)
        pred_z = np.array([self._voltage_model(x[0], x[1], x[4], frame.current_A, x[5]), x[5]], dtype=float)
        h = self._measurement_jacobian(frame.current_A)
        r = np.diag([self.config.voltage_noise_V ** 2, self.config.temperature_noise_K ** 2])
        residual = z - pred_z
        s = h @ self.cov @ h.T + r
        k_gain = self.cov @ h.T @ np.linalg.pinv(s)
        x_new = x + k_gain @ residual
        self.cov = (np.eye(len(x)) - k_gain @ h) @ self.cov
        self.set_vector(x_new)
        self.state.risk = self._risk_model(
            self.state.soc,
            self.state.soh,
            self.state.sei_nm,
            self.state.dendrite_index,
            self.state.resistance_mohm,
            self.state.core_temperature_K,
            abs(frame.current_A) / max(self.config.nominal_capacity_Ah, 1e-9),
        )
        alert = self._update_alert_streak(self.state.risk)
        return {
            "voltage_residual_V": float(residual[0]),
            "temperature_residual_K": float(residual[1]),
            "risk": float(self.state.risk),
            "alert": float(alert),
            "cov_trace": float(np.trace(self.cov)),
        }

    def _voltage_model(self, soc: float, soh: float, r_mohm: float, current_A: float, temp_K: float) -> float:
        ocv = 2.65 + 0.92 * soc + 0.07 * math.tanh((soc - 0.5) * 6.0)
        temp_shift = -0.00055 * (temp_K - 298.15)
        aging_shift = -0.08 * (1.0 - soh)
        ohmic = current_A * r_mohm / 1000.0
        return float(ocv + temp_shift + aging_shift - ohmic)

    def _measurement_jacobian(self, current_A: float) -> np.ndarray:
        h = np.zeros((2, 7), dtype=float)
        soc = self.state.soc
        h[0, 0] = 0.92 + 0.42 * (1.0 - math.tanh((soc - 0.5) * 6.0) ** 2)
        h[0, 1] = 0.08
        h[0, 4] = -current_A / 1000.0
        h[0, 5] = -0.00055
        h[1, 5] = 1.0
        return h

    def _risk_model(self, soc: float, soh: float, sei: float, dend: float, r_mohm: float, temp: float, c_rate: float) -> float:
        temp_drive = (temp - 318.15) / 7.5
        resistance_drive = (r_mohm - 20.0) / 7.0
        dend_drive = 4.0 * dend
        soc_drive = max(0.0, soc - 0.92) * 3.0
        soh_drive = max(0.0, 0.78 - soh) * 4.0
        rate_drive = max(0.0, c_rate - 1.0) * 1.0
        monsoon_hot_zone = max(0.0, temp - 315.15) / 30.0
        logit = -1.0 + temp_drive + resistance_drive + dend_drive + soc_drive + soh_drive + rate_drive + monsoon_hot_zone + 0.006 * max(0.0, sei - 25.0)
        return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))

    def _update_alert_streak(self, risk: float) -> bool:
        if risk >= self.config.risk_threshold:
            self.alert_streak += 1
        else:
            self.alert_streak = 0
        return self.alert_streak >= self.config.required_consecutive_alerts

    def ingest(self, frames: Iterable[TelemetryFrame]) -> Dict[str, Any]:
        rows: List[Dict[str, float]] = []
        alerts: List[Dict[str, float]] = []
        prev_t: Optional[float] = None
        for frame in frames:
            dt = 1.0 if prev_t is None else max(0.1, float(frame.t_s) - prev_t)
            prev_t = float(frame.t_s)
            self.predict(frame, dt)
            update = self.update(frame)
            state = asdict(self.state)
            row = {"t_s": frame.t_s, **state, **update}
            rows.append(row)
            if update["alert"] >= 1.0 and not alerts:
                alerts.append({"t_s": frame.t_s, "risk": update["risk"], "core_temperature_K": self.state.core_temperature_K})
        return {"trajectory": rows, "alerts": alerts, "final_state": asdict(self.state)}


def simulate_telemetry(n: int = 240, seed: int = 42, inject_fault: bool = True) -> List[TelemetryFrame]:
    rng = np.random.RandomState(seed)
    frames: List[TelemetryFrame] = []
    soc = 0.88
    temp = 313.15
    r_mohm = 20.0
    for i in range(n):
        t = float(i * 10)
        current = 2.8 + 1.2 * math.sin(i / 15.0)
        ambient = 310.0 + 8.0 * math.sin(i / 80.0)
        if inject_fault and i > int(n * 0.55):
            current += 2.3
            r_mohm += 0.05 * (i - int(n * 0.55))
            temp += 0.035 * (i - int(n * 0.55))
        soc = max(0.05, soc - current * 10.0 / (3600.0 * 3.0))
        ocv = 2.65 + 0.92 * soc
        temp = temp + (current ** 2 * r_mohm / 1000.0 - 0.35 * (temp - ambient)) * 10.0 / 920.0
        voltage = ocv - current * r_mohm / 1000.0 + rng.normal(0, 0.012)
        frames.append(TelemetryFrame(t_s=t, voltage_V=voltage, current_A=current, temperature_K=temp + rng.normal(0, 0.8), ambient_K=ambient))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/digital_twin_assimilation_v2.json")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--no-fault", action="store_true")
    args = parser.parse_args()
    twin = BatteryDigitalTwin()
    result = twin.ingest(simulate_telemetry(n=args.n, inject_fault=not args.no_fault))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(result["trajectory"]), "alerts": len(result["alerts"]), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
