import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AuditFinding:
    check_id: str
    severity: str
    message: str
    value: Optional[float] = None
    limit: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    subject: str
    generated_at: str
    score: float
    passed: bool
    findings: List[AuditFinding] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "generated_at": self.generated_at,
            "score": self.score,
            "passed": self.passed,
            "findings": [asdict(f) for f in self.findings],
            "metrics": self.metrics,
        }


class PhysicsAuditSuite:
    def __init__(self, strict: bool = False):
        self.strict = strict

    def audit_cathode_screening(self, results: Sequence[Dict[str, Any]]) -> AuditReport:
        findings: List[AuditFinding] = []
        if not results:
            findings.append(AuditFinding("cathode.empty", "critical", "No cathode screening results were supplied."))
            return self._report("cathode_screening", findings, {})
        capacities = np.array([float(r.get("Q0", np.nan)) for r in results], dtype=float)
        fades = np.array([float(r.get("fade_500", r.get("fade", np.nan))) for r in results], dtype=float)
        costs = np.array([float(r.get("cost_usd_kwh", np.nan)) for r in results], dtype=float)
        phase = np.array([float(r.get("phase_stability", np.nan)) for r in results], dtype=float)
        self._check_finite("cathode.Q0.finite", capacities, findings)
        self._check_finite("cathode.fade.finite", fades, findings)
        self._check_range("cathode.Q0.range", capacities, 50.0, 240.0, findings, "mAh/g")
        self._check_range("cathode.fade.range", fades, 0.0, 0.65, findings, "fraction")
        self._check_range("cathode.cost.range", costs, 0.0, 600.0, findings, "USD/kWh")
        self._check_range("cathode.phase.range", phase, 0.0, 1.0, findings, "score")
        for idx, item in enumerate(results[:20]):
            comp = item.get("comp") or {}
            if comp:
                total_transition = float(comp.get("Mn", 0.0)) + float(comp.get("Fe", 0.0)) + float(comp.get("dopant_frac", 0.0))
                if total_transition < 0.45 or total_transition > 1.10:
                    findings.append(
                        AuditFinding(
                            "cathode.stoichiometry.transition_sum",
                            "major",
                            "Transition metal fraction is outside the expected layered oxide design window.",
                            value=total_transition,
                            limit=1.10,
                            context={"index": idx, "composition": comp},
                        )
                    )
        metrics = {
            "n_results": float(len(results)),
            "best_score": float(max(float(r.get("score", 0.0)) for r in results)),
            "mean_Q0": self._nanmean(capacities),
            "mean_fade_500": self._nanmean(fades),
            "median_cost_usd_kwh": self._nanmedian(costs),
        }
        return self._report("cathode_screening", findings, metrics)

    def audit_capacity_curve(self, cycles: Iterable[float], capacity: Iterable[float], subject: str = "capacity_curve") -> AuditReport:
        c = np.asarray(list(cycles), dtype=float)
        q = np.asarray(list(capacity), dtype=float)
        findings: List[AuditFinding] = []
        self._check_finite("curve.cycles.finite", c, findings)
        self._check_finite("curve.capacity.finite", q, findings)
        if len(c) != len(q):
            findings.append(AuditFinding("curve.length", "critical", "Cycles and capacity arrays have different lengths."))
        if len(c) > 1 and np.any(np.diff(c) <= 0):
            findings.append(AuditFinding("curve.cycles.monotonic", "critical", "Cycle index must increase monotonically."))
        if len(q) > 2:
            upward_steps = np.diff(q) > np.maximum(0.04 * np.maximum(q[:-1], 1.0), 3.0)
            if np.any(upward_steps):
                findings.append(
                    AuditFinding(
                        "curve.capacity.large_recovery",
                        "major",
                        "Capacity curve has large upward jumps that look unlike normal cycling data.",
                        value=float(np.max(np.diff(q))),
                        context={"jump_count": int(np.sum(upward_steps))},
                    )
                )
        self._check_range("curve.capacity.range", q, 0.0, 260.0, findings, "mAh/g")
        metrics = {
            "n_points": float(len(q)),
            "initial_capacity": float(q[0]) if len(q) else math.nan,
            "final_capacity": float(q[-1]) if len(q) else math.nan,
            "fade_fraction": float(1.0 - q[-1] / max(q[0], 1e-9)) if len(q) else math.nan,
        }
        return self._report(subject, findings, metrics)

    def audit_bms_history(self, history: Dict[str, Any], alerts: Optional[Sequence[Dict[str, Any]]] = None) -> AuditReport:
        findings: List[AuditFinding] = []
        risk = np.asarray(history.get("risk", []), dtype=float)
        temp = np.asarray(history.get("T_cells", history.get("temperature", [])), dtype=float)
        voltage = np.asarray(history.get("V_cells", history.get("voltage", [])), dtype=float)
        soc = np.asarray(history.get("SOC_cells", history.get("soc", [])), dtype=float)
        self._check_finite("bms.risk.finite", risk, findings)
        self._check_finite("bms.temperature.finite", temp, findings)
        self._check_finite("bms.voltage.finite", voltage, findings)
        self._check_range("bms.risk.range", risk, 0.0, 1.0, findings, "risk")
        self._check_range("bms.temperature.range", temp, 240.0, 460.0, findings, "K")
        self._check_range("bms.voltage.range", voltage, 1.5, 5.2, findings, "V")
        if soc.size:
            self._check_range("bms.soc.range", soc, -0.05, 1.05, findings, "fraction")
        if risk.size and temp.size and risk.shape[0] == temp.shape[0]:
            mean_temp = temp.mean(axis=-1) if temp.ndim > 1 else temp
            high_risk = np.where(np.max(risk, axis=-1) > 0.75 if risk.ndim > 1 else risk > 0.75)[0]
            high_temp = np.where(mean_temp > 333.0)[0]
            if len(high_risk) and len(high_temp):
                lead = float((high_temp[0] - high_risk[0]))
                if lead < 0:
                    findings.append(AuditFinding("bms.alert.late", "critical", "Risk crosses threshold after thermal stress appears.", value=lead))
        alert_count = len(alerts or [])
        metrics = {
            "risk_max": float(np.nanmax(risk)) if risk.size else math.nan,
            "temperature_max_K": float(np.nanmax(temp)) if temp.size else math.nan,
            "voltage_min_V": float(np.nanmin(voltage)) if voltage.size else math.nan,
            "alert_count": float(alert_count),
        }
        return self._report("bms_history", findings, metrics)

    def audit_recycling_solution(self, solution: Dict[str, Any], trajectory: Optional[np.ndarray] = None) -> AuditReport:
        findings: List[AuditFinding] = []
        values = {
            "T": (float(solution.get("T", solution.get("optimal_T", math.nan))), 303.0, 383.0),
            "pH": (float(solution.get("pH", solution.get("optimal_pH", math.nan))), 0.0, 4.5),
            "conc": (float(solution.get("conc", solution.get("optimal_conc", math.nan))), 0.0, 5.0),
            "t": (float(solution.get("t", solution.get("optimal_t", math.nan))), 1.0, 300.0),
            "alpha_Mn": (float(solution.get("alpha_Mn", math.nan)), 0.0, 1.0),
            "alpha_Fe": (float(solution.get("alpha_Fe", math.nan)), 0.0, 1.0),
            "alpha_Na": (float(solution.get("alpha_Na", math.nan)), 0.0, 1.0),
        }
        for key, (value, lo, hi) in values.items():
            self._check_scalar_range(f"recycling.{key}.range", value, lo, hi, findings)
        if trajectory is not None:
            arr = np.asarray(trajectory, dtype=float)
            self._check_finite("recycling.trajectory.finite", arr, findings)
            self._check_range("recycling.trajectory.range", arr, 0.0, 1.0, findings, "fraction")
            if arr.ndim >= 2:
                diffs = np.diff(arr, axis=-1)
                if np.nanmin(diffs) < -0.03:
                    findings.append(
                        AuditFinding(
                            "recycling.trajectory.monotonic",
                            "major",
                            "Extraction fraction should not materially decrease during a leaching run.",
                            value=float(np.nanmin(diffs)),
                            limit=-0.03,
                        )
                    )
        recovery = 0.5 * values["alpha_Mn"][0] + 0.3 * values["alpha_Fe"][0] + 0.2 * values["alpha_Na"][0]
        metrics = {"weighted_recovery": float(recovery), **{k: float(v[0]) for k, v in values.items()}}
        return self._report("recycling_solution", findings, metrics)

    def merge_reports(self, reports: Sequence[AuditReport], subject: str = "v2_readiness") -> AuditReport:
        findings: List[AuditFinding] = []
        metrics: Dict[str, float] = {}
        for report in reports:
            findings.extend(report.findings)
            metrics[f"{report.subject}.score"] = report.score
            for key, value in report.metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metrics[f"{report.subject}.{key}"] = float(value)
        score = float(np.mean([r.score for r in reports])) if reports else 0.0
        passed = all(r.passed for r in reports)
        return AuditReport(subject=subject, generated_at=_now(), score=score, passed=passed, findings=findings, metrics=metrics)

    def _check_finite(self, check_id: str, arr: np.ndarray, findings: List[AuditFinding]) -> None:
        if arr.size == 0:
            findings.append(AuditFinding(check_id, "major", "Array is empty."))
            return
        nonfinite = int(np.sum(~np.isfinite(arr)))
        if nonfinite:
            findings.append(AuditFinding(check_id, "critical", "Array contains non-finite values.", value=float(nonfinite)))

    def _check_range(
        self,
        check_id: str,
        arr: np.ndarray,
        lo: float,
        hi: float,
        findings: List[AuditFinding],
        unit: str,
    ) -> None:
        if arr.size == 0:
            return
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        below = int(np.sum(finite < lo))
        above = int(np.sum(finite > hi))
        if below or above:
            severity = "critical" if self.strict else "major"
            findings.append(
                AuditFinding(
                    check_id,
                    severity,
                    f"Values fall outside the expected physical range [{lo}, {hi}] {unit}.",
                    value=float(min(np.nanmin(finite), np.nanmax(finite))),
                    limit=float(hi),
                    context={"below": below, "above": above},
                )
            )

    def _check_scalar_range(self, check_id: str, value: float, lo: float, hi: float, findings: List[AuditFinding]) -> None:
        if not math.isfinite(value):
            findings.append(AuditFinding(check_id, "critical", "Scalar value is not finite."))
        elif value < lo or value > hi:
            findings.append(AuditFinding(check_id, "major", f"Scalar value is outside expected range [{lo}, {hi}].", value=value, limit=hi))

    def _nanmean(self, arr: np.ndarray) -> float:
        return float(np.nanmean(arr)) if arr.size else math.nan

    def _nanmedian(self, arr: np.ndarray) -> float:
        return float(np.nanmedian(arr)) if arr.size else math.nan

    def _report(self, subject: str, findings: List[AuditFinding], metrics: Dict[str, float]) -> AuditReport:
        score = self._score(findings)
        passed = not any(f.severity == "critical" for f in findings) and score >= (0.78 if self.strict else 0.62)
        return AuditReport(subject=subject, generated_at=_now(), score=score, passed=passed, findings=findings, metrics=metrics)

    def _score(self, findings: Sequence[AuditFinding]) -> float:
        penalty = 0.0
        for f in findings:
            if f.severity == "critical":
                penalty += 0.35
            elif f.severity == "major":
                penalty += 0.14
            else:
                penalty += 0.04
        return float(max(0.0, 1.0 - penalty))


def audit_cache(project_root: Path) -> AuditReport:
    suite = PhysicsAuditSuite()
    cache = project_root / "data" / "cache"
    reports: List[AuditReport] = []
    cathode_path = cache / "cathode_screening.npz"
    if cathode_path.exists():
        data = np.load(cathode_path, allow_pickle=True)
        rankings = list(data["rankings"])
        reports.append(suite.audit_cathode_screening(rankings))
        if "fade_curves" in data and "cycles" in data:
            reports.append(suite.audit_capacity_curve(data["cycles"], data["fade_curves"][0], subject="cached_fade_curve"))
    recycling_path = cache / "recycling_optimization.npz"
    if recycling_path.exists():
        data = np.load(recycling_path, allow_pickle=True)
        solution = {k: float(data[k]) for k in data.files if np.asarray(data[k]).shape == ()}
        trajectory = data["alpha_trajectory"] if "alpha_trajectory" in data else None
        reports.append(suite.audit_recycling_solution(solution, trajectory))
    for bms_file in sorted(cache.glob("bms_drive_*.npz"))[:2]:
        data = np.load(bms_file, allow_pickle=True)
        history = {k: data[k] for k in data.files}
        reports.append(suite.audit_bms_history(history))
    return suite.merge_reports(reports) if reports else AuditReport("cache", _now(), 0.0, False, [AuditFinding("cache.empty", "critical", "No cache files found.")], {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="data/cache/physics_audit_v2.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    report = audit_cache(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"passed": report.passed, "score": report.score, "findings": len(report.findings), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
