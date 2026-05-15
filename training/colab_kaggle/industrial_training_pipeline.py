import argparse
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class DataContractIssue:
    file: str
    severity: str
    message: str


@dataclass
class DatasetFingerprint:
    file_count: int
    total_bytes: int
    sha256: str
    issues: List[DataContractIssue] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelCard:
    model_name: str
    task: str
    generated_at: str
    intended_use: str
    out_of_scope: List[str]
    data_fingerprint: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    validation_protocol: List[str]
    known_failure_modes: List[str]


class DataContractChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    @staticmethod
    def leaching_trajectory_array(data: Any) -> Optional[np.ndarray]:
        for key in ("alpha_trajectories", "trajectories"):
            if key in data.files:
                return np.asarray(data[key], dtype=float)
        return None

    def fingerprint_files(self, files: Sequence[Path]) -> DatasetFingerprint:
        h = hashlib.sha256()
        total = 0
        for path in sorted(files):
            rel = str(path.relative_to(self.project_root)).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            data = path.read_bytes()
            h.update(hashlib.sha256(data).hexdigest().encode("ascii"))
            total += len(data)
        return DatasetFingerprint(file_count=len(files), total_bytes=total, sha256=h.hexdigest())

    def cathode(self, max_files: int = 40) -> DatasetFingerprint:
        files = [Path(p) for p in glob.glob(str(self.project_root / "data" / "synthetic" / "cathode" / "cathode_*.npz"))][:max_files]
        fp = self.fingerprint_files(files)
        caps = []
        for path in files:
            try:
                d = np.load(path, allow_pickle=True)
                for key in ["capacity", "cycles", "temperature", "resistance"]:
                    if key not in d.files:
                        fp.issues.append(DataContractIssue(str(path), "critical", f"Missing key {key}."))
                if "capacity" in d.files:
                    cap = np.asarray(d["capacity"], dtype=float)
                    caps.extend(cap[np.isfinite(cap)].tolist())
                    if np.nanmax(cap) > 260 or np.nanmin(cap) < 0:
                        fp.issues.append(DataContractIssue(str(path), "major", "Capacity outside expected physical bounds."))
            except Exception as exc:
                fp.issues.append(DataContractIssue(str(path), "critical", f"Could not read NPZ: {exc}"))
        arr = np.asarray(caps, dtype=float)
        fp.stats = {"capacity_mean": _nanmean(arr), "capacity_min": _nanmin(arr), "capacity_max": _nanmax(arr)}
        return fp

    def bms(self, max_files: int = 20) -> DatasetFingerprint:
        files = [Path(p) for p in glob.glob(str(self.project_root / "data" / "synthetic" / "bms" / "bms_*.npz"))][:max_files]
        fp = self.fingerprint_files(files)
        risks = []
        temps = []
        for path in files:
            try:
                d = np.load(path, allow_pickle=True)
                for key in ["V", "T", "I", "risk"]:
                    if key not in d.files:
                        fp.issues.append(DataContractIssue(str(path), "critical", f"Missing key {key}."))
                if "risk" in d.files:
                    risk = np.asarray(d["risk"], dtype=float)
                    risks.extend(risk[np.isfinite(risk)].ravel().tolist())
                if "T" in d.files:
                    temp = np.asarray(d["T"], dtype=float)
                    temps.extend(temp[np.isfinite(temp)].ravel().tolist())
            except Exception as exc:
                fp.issues.append(DataContractIssue(str(path), "critical", f"Could not read NPZ: {exc}"))
        fp.stats = {"risk_max": _nanmax(np.asarray(risks)), "temperature_max": _nanmax(np.asarray(temps))}
        return fp

    def recycling(self) -> DatasetFingerprint:
        files = [self.project_root / "data" / "synthetic" / "leaching" / "leaching_grid.npz"]
        files = [f for f in files if f.exists()]
        fp = self.fingerprint_files(files)
        if not files:
            fp.issues.append(DataContractIssue("data/synthetic/leaching/leaching_grid.npz", "critical", "Leaching grid is missing."))
            return fp
        d = np.load(files[0], allow_pickle=True)
        for key in ["conditions"]:
            if key not in d.files:
                fp.issues.append(DataContractIssue(str(files[0]), "critical", f"Missing key {key}."))
        traj = self.leaching_trajectory_array(d)
        if traj is None:
            fp.issues.append(DataContractIssue(str(files[0]), "critical", "Missing key alpha_trajectories or trajectories."))
        else:
            if traj.ndim != 3 or traj.shape[1] != 3:
                fp.issues.append(DataContractIssue(str(files[0]), "critical", f"Unexpected trajectory shape {traj.shape}; expected (conditions, 3, time)."))
            if np.nanmin(traj) < 0.0 or np.nanmax(traj) > 1.0:
                fp.issues.append(DataContractIssue(str(files[0]), "major", "Leaching trajectory values outside [0, 1]."))
            fp.stats = {"trajectory_min": _nanmin(traj), "trajectory_max": _nanmax(traj), "trajectory_mean": _nanmean(traj)}
        d.close()
        return fp


class BaselineEvaluator:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def cathode_exponential_fade(self, max_files: int = 30) -> Dict[str, float]:
        files = [Path(p) for p in glob.glob(str(self.project_root / "data" / "synthetic" / "cathode" / "cathode_*.npz"))][:max_files]
        errors = []
        for path in files:
            d = np.load(path, allow_pickle=True)
            q = np.asarray(d["capacity"], dtype=float)
            x = np.arange(len(q), dtype=float)
            mask = np.isfinite(q) & (q > 1.0)
            if mask.sum() < 10:
                continue
            y = np.log(q[mask] / max(q[mask][0], 1e-9))
            slope = np.polyfit(x[mask], y, 1)[0]
            pred = q[mask][0] * np.exp(slope * x)
            errors.extend(np.abs(pred[mask] - q[mask]).tolist())
        return _error_metrics(errors, "mAh_g")

    def bms_threshold_baseline(self, max_files: int = 12) -> Dict[str, float]:
        files = [Path(p) for p in glob.glob(str(self.project_root / "data" / "synthetic" / "bms" / "bms_*.npz"))][:max_files]
        lead_times = []
        missed = 0
        for path in files:
            d = np.load(path, allow_pickle=True)
            risk = np.asarray(d["risk"], dtype=float)
            temp = np.asarray(d["T"], dtype=float)
            risk_line = risk.max(axis=1) if risk.ndim > 1 else risk
            temp_line = temp.max(axis=1) if temp.ndim > 1 else temp
            failure = np.where(temp_line > 333.15)[0]
            alert = np.where(risk_line > 0.75)[0]
            if len(failure) and len(alert):
                lead_times.append(float(failure[0] - alert[0]))
            elif len(failure):
                missed += 1
        arr = np.asarray(lead_times, dtype=float)
        return {"mean_lead_steps": _nanmean(arr), "min_lead_steps": _nanmin(arr), "missed_failures": float(missed), "evaluated_files": float(len(files))}

    def recycling_last_point_baseline(self) -> Dict[str, float]:
        path = self.project_root / "data" / "synthetic" / "leaching" / "leaching_grid.npz"
        if not path.exists():
            return {"mae_fraction": math.nan, "evaluated_points": 0.0}
        d = np.load(path, allow_pickle=True)
        traj = DataContractChecker.leaching_trajectory_array(d)
        d.close()
        if traj is None:
            return {"mae_fraction": math.nan, "rmse_fraction": math.nan, "n": 0.0, "evaluated_points": 0.0}
        if traj.ndim < 3:
            return {"mae_fraction": math.nan, "evaluated_points": 0.0}
        final = traj[:, :, -1]
        simple = np.repeat(final[:, :, None], traj.shape[2], axis=2)
        errors = np.abs(simple - traj).ravel()
        return _error_metrics(errors.tolist(), "fraction")


class IndustrialTrainingPipeline:
    def __init__(self, project_root: Path, task: str = "all", profile: str = "smoke"):
        self.project_root = project_root.resolve()
        self.task = task
        self.profile = profile
        self.out_dir = self.project_root / "training" / "colab_kaggle" / "runs" / f"industrial_{task}_{profile}_{int(time.time())}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.contracts = DataContractChecker(self.project_root)
        self.baselines = BaselineEvaluator(self.project_root)

    def run(self) -> Dict[str, Any]:
        tasks = ["cathode", "bms", "recycling"] if self.task == "all" else [self.task]
        payload: Dict[str, Any] = {"task": self.task, "profile": self.profile, "generated_at": _now(), "contracts": {}, "baselines": {}, "model_cards": {}}
        for task in tasks:
            fp = getattr(self.contracts, task)()
            payload["contracts"][task] = _fingerprint_to_dict(fp)
            if any(issue.severity == "critical" for issue in fp.issues):
                payload["baselines"][task] = {"skipped": 1.0}
                continue
            if task == "cathode":
                metrics = self.baselines.cathode_exponential_fade()
            elif task == "bms":
                metrics = self.baselines.bms_threshold_baseline()
            else:
                metrics = self.baselines.recycling_last_point_baseline()
            payload["baselines"][task] = metrics
            payload["model_cards"][task] = asdict(self.model_card(task, fp, metrics))
        (self.out_dir / "industrial_training_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def model_card(self, task: str, fp: DatasetFingerprint, metrics: Dict[str, float]) -> ModelCard:
        return ModelCard(
            model_name=f"kineticsforge_v2_{task}_{self.profile}",
            task=task,
            generated_at=_now(),
            intended_use=self._intended_use(task),
            out_of_scope=[
                "Do not treat synthetic-only metrics as experimental validation.",
                "Do not claim regulatory certification from simulation outputs.",
                "Do not extrapolate to chemistries outside the registered material system without an OOD audit.",
            ],
            data_fingerprint=_fingerprint_to_dict(fp),
            baseline_metrics=metrics,
            validation_protocol=self._validation_protocol(task),
            known_failure_modes=self._failure_modes(task),
        )

    def _intended_use(self, task: str) -> str:
        return {
            "cathode": "Rank Na-Mn-Fe-O cathode candidates and generate a lab shortlist under explicit uncertainty.",
            "bms": "Estimate pack risk trajectory from V/I/T telemetry and prioritize earlier warnings for review.",
            "recycling": "Optimize leaching conditions and quantify recovery-cost-impurity tradeoffs.",
        }.get(task, "KineticsForge industrial training governance.")

    def _validation_protocol(self, task: str) -> List[str]:
        return {
            "cathode": [
                "Hold out entire compositions, not random cycles.",
                "Report physics-only, exponential baseline, and SINDy-NODE side by side.",
                "Require a lab replay on at least one known literature composition before investor-facing claims.",
            ],
            "bms": [
                "Hold out full drive cycles and fault types.",
                "Report missed-failure count, early-warning lead time, and false-alert time.",
                "Stress test threshold under sensor noise and missing temperature channel.",
            ],
            "recycling": [
                "Hold out full leaching conditions, not random time points.",
                "Report recovery, impurity, reagent cost, and waste index together.",
                "Validate recommended point with at least duplicate wet-lab leaches before scale-up claims.",
            ],
        }.get(task, [])

    def _failure_modes(self, task: str) -> List[str]:
        return {
            "cathode": [
                "Surrogate phase stability can overrate metastable compositions.",
                "Sodium volatility and moisture sensitivity are underrepresented in synthetic curves.",
                "Dopant benefit can reverse if actual site occupancy differs from assumed mixing.",
            ],
            "bms": [
                "Telemetry-only model cannot see separator defects directly.",
                "Sensor bias can masquerade as thermal drift.",
                "Pack topology mismatch can understate inter-cell propagation.",
            ],
            "recycling": [
                "Black mass impurity distribution can vary by supplier and dismantling method.",
                "Acid consumption and downstream purification cost can dominate recovery gains.",
                "Recovered salt purity may be insufficient for cathode synthesis without cleanup.",
            ],
        }.get(task, [])


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _nanmean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if arr.size else math.nan


def _nanmin(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if arr.size else math.nan


def _nanmax(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if arr.size else math.nan


def _error_metrics(errors: Sequence[float], unit: str) -> Dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"mae_{unit}": math.nan, f"rmse_{unit}": math.nan, "n": 0.0}
    return {f"mae_{unit}": float(np.mean(np.abs(arr))), f"rmse_{unit}": float(np.sqrt(np.mean(arr ** 2))), "n": float(arr.size)}


def _fingerprint_to_dict(fp: DatasetFingerprint) -> Dict[str, Any]:
    return {
        "file_count": fp.file_count,
        "total_bytes": fp.total_bytes,
        "sha256": fp.sha256,
        "issues": [asdict(i) for i in fp.issues],
        "stats": fp.stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task", choices=["cathode", "bms", "recycling", "all"], default="all")
    parser.add_argument("--profile", default="smoke")
    args = parser.parse_args()
    pipeline = IndustrialTrainingPipeline(Path(args.project_root), task=args.task, profile=args.profile)
    payload = pipeline.run()
    print(json.dumps({"out_dir": str(pipeline.out_dir), "tasks": list(payload["contracts"].keys())}, indent=2))


if __name__ == "__main__":
    main()
