import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "kineticsforge.evidence.v2"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, prefix: str = "ev") -> str:
    payload = "|".join(str(p) for p in parts)
    return f"{prefix}_{sha256_text(payload)[:16]}"


@dataclass
class EvidenceSource:
    source_id: str
    title: str
    source_type: str
    uri: str = ""
    citation: str = ""
    license: str = "unknown"
    reliability: float = 0.5
    collected_at: str = field(default_factory=_utc_now)
    sha256: str = ""
    notes: str = ""

    def normalized_reliability(self) -> float:
        return float(max(0.0, min(1.0, self.reliability)))


@dataclass
class EvidenceRecord:
    record_id: str
    source_id: str
    claim_type: str
    material_system: str
    metric: str
    value: float
    unit: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    provenance_path: str = ""
    extraction_method: str = "manual"
    confidence: float = 0.5
    notes: str = ""

    def interval(self) -> Tuple[float, float]:
        lo = self.value if self.lower is None else self.lower
        hi = self.value if self.upper is None else self.upper
        return (float(min(lo, hi)), float(max(lo, hi)))

    def normalized_confidence(self) -> float:
        return float(max(0.0, min(1.0, self.confidence)))


@dataclass
class ClaimAssessment:
    claim_id: str
    claim: str
    metric: str
    proposed_value: float
    unit: str
    support_score: float
    contradiction_score: float
    evidence_count: int
    verdict: str
    rationale: str
    supporting_records: List[str] = field(default_factory=list)
    contradicting_records: List[str] = field(default_factory=list)


class EvidenceRegistry:
    def __init__(self) -> None:
        self.sources: Dict[str, EvidenceSource] = {}
        self.records: Dict[str, EvidenceRecord] = {}

    def add_source(self, source: EvidenceSource) -> EvidenceSource:
        self.sources[source.source_id] = source
        return source

    def add_record(self, record: EvidenceRecord) -> EvidenceRecord:
        if record.source_id not in self.sources:
            self.add_source(
                EvidenceSource(
                    source_id=record.source_id,
                    title=record.source_id,
                    source_type="unregistered",
                    reliability=0.35,
                )
            )
        self.records[record.record_id] = record
        return record

    def source(self, source_id: str) -> EvidenceSource:
        return self.sources[source_id]

    def query(
        self,
        metric: Optional[str] = None,
        material_system: Optional[str] = None,
        claim_type: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceRecord]:
        out: List[EvidenceRecord] = []
        for record in self.records.values():
            if metric and record.metric != metric:
                continue
            if material_system and material_system.lower() not in record.material_system.lower():
                continue
            if claim_type and record.claim_type != claim_type:
                continue
            if conditions and not self._conditions_match(record.conditions, conditions):
                continue
            out.append(record)
        return out

    def _conditions_match(self, observed: Dict[str, Any], requested: Dict[str, Any]) -> bool:
        for key, expected in requested.items():
            if key not in observed:
                return False
            actual = observed[key]
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                try:
                    value = float(actual)
                except Exception:
                    return False
                if value < float(expected[0]) or value > float(expected[1]):
                    return False
            elif str(actual).lower() != str(expected).lower():
                return False
        return True

    def weighted_metric_summary(self, metric: str, material_system: str = "") -> Dict[str, float]:
        records = self.query(metric=metric, material_system=material_system or None)
        if not records:
            return {"count": 0, "mean": math.nan, "weighted_mean": math.nan, "spread": math.nan}
        values = [r.value for r in records]
        weights = [self._record_weight(r) for r in records]
        total_w = sum(weights) or 1.0
        weighted = sum(v * w for v, w in zip(values, weights)) / total_w
        spread = math.sqrt(sum(w * (v - weighted) ** 2 for v, w in zip(values, weights)) / total_w)
        return {"count": len(records), "mean": float(mean(values)), "weighted_mean": float(weighted), "spread": float(spread)}

    def assess_claim(
        self,
        claim: str,
        metric: str,
        proposed_value: float,
        unit: str,
        material_system: str = "",
        tolerance_fraction: float = 0.15,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> ClaimAssessment:
        records = self.query(metric=metric, material_system=material_system or None, conditions=conditions)
        supporting: List[str] = []
        contradicting: List[str] = []
        support = 0.0
        contradiction = 0.0
        for record in records:
            weight = self._record_weight(record)
            lo, hi = record.interval()
            tolerance = max(abs(proposed_value) * tolerance_fraction, 1e-12)
            expanded_lo = lo - tolerance
            expanded_hi = hi + tolerance
            if expanded_lo <= proposed_value <= expanded_hi:
                support += weight
                supporting.append(record.record_id)
            else:
                distance = min(abs(proposed_value - expanded_lo), abs(proposed_value - expanded_hi))
                contradiction += weight * min(1.5, 1.0 + distance / tolerance)
                contradicting.append(record.record_id)
        denom = support + contradiction + 1e-9
        support_score = support / denom if records else 0.0
        contradiction_score = contradiction / denom if records else 0.0
        verdict = self._verdict(support_score, contradiction_score, len(records))
        rationale = self._rationale(verdict, support_score, contradiction_score, records)
        return ClaimAssessment(
            claim_id=stable_id(claim, metric, proposed_value, unit, prefix="claim"),
            claim=claim,
            metric=metric,
            proposed_value=float(proposed_value),
            unit=unit,
            support_score=float(support_score),
            contradiction_score=float(contradiction_score),
            evidence_count=len(records),
            verdict=verdict,
            rationale=rationale,
            supporting_records=supporting,
            contradicting_records=contradicting,
        )

    def _record_weight(self, record: EvidenceRecord) -> float:
        source = self.sources.get(record.source_id)
        source_rel = source.normalized_reliability() if source else 0.35
        return float(source_rel * record.normalized_confidence())

    def _verdict(self, support: float, contradiction: float, count: int) -> str:
        if count == 0:
            return "unsupported"
        if support >= 0.72 and contradiction <= 0.28:
            return "defensible"
        if support >= 0.50 and contradiction <= 0.50:
            return "plausible_needs_validation"
        if contradiction > support:
            return "contested"
        return "weakly_supported"

    def _rationale(self, verdict: str, support: float, contradiction: float, records: Sequence[EvidenceRecord]) -> str:
        if not records:
            return "No matching evidence records are registered for this metric and material system."
        return (
            f"{verdict}: {len(records)} records matched; "
            f"weighted support={support:.2f}, weighted contradiction={contradiction:.2f}."
        )

    def materialize_trace(self, record_ids: Iterable[str]) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        for rid in record_ids:
            record = self.records.get(rid)
            if not record:
                continue
            source = self.sources.get(record.source_id)
            item = asdict(record)
            item["source"] = asdict(source) if source else None
            trace.append(item)
        return trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "sources": [asdict(s) for s in self.sources.values()],
            "records": [asdict(r) for r in self.records.values()],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceRegistry":
        reg = cls()
        for src in payload.get("sources", []):
            reg.add_source(EvidenceSource(**src))
        for rec in payload.get("records", []):
            reg.add_record(EvidenceRecord(**rec))
        return reg

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "EvidenceRegistry":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for src in self.sources.values():
                f.write(json.dumps({"kind": "source", **asdict(src)}) + "\n")
            for rec in self.records.values():
                f.write(json.dumps({"kind": "record", **asdict(rec)}) + "\n")

    def save_claim_assessment_csv(self, path: Path, assessments: Sequence[ClaimAssessment]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(asdict(assessments[0]).keys()) if assessments else list(ClaimAssessment.__dataclass_fields__.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for assessment in assessments:
                row = asdict(assessment)
                row["supporting_records"] = ";".join(row["supporting_records"])
                row["contradicting_records"] = ";".join(row["contradicting_records"])
                writer.writerow(row)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_registry_from_project(root: Path) -> EvidenceRegistry:
    reg = EvidenceRegistry()
    root = root.resolve()
    manifest = _load_json_if_exists(root / "data" / "real" / "assembled" / "real_dataset_manifest.json")
    quality = _load_json_if_exists(root / "data" / "cache" / "data_quality_report_foundation.json")
    curation = _load_json_if_exists(root / "data" / "real" / "scraped" / "curation_report.json")
    real_source = EvidenceSource(
        source_id="real_assembled_dataset",
        title="Assembled open Na-ion battery dataset",
        source_type="dataset_manifest",
        uri=str(root / "data" / "real" / "assembled"),
        reliability=0.78,
        sha256=sha256_text(json.dumps(manifest, sort_keys=True)) if manifest else "",
        notes="Local manifest-derived evidence. Review raw citations before investor-grade external claims.",
    )
    reg.add_source(real_source)
    if manifest:
        for key, metric in [
            ("cycle_summary_rows", "real_cycle_summary_rows"),
            ("timeseries_sample_rows", "real_timeseries_sample_rows"),
            ("literature_measurement_rows", "literature_measurement_rows"),
        ]:
            value = manifest.get(key)
            if isinstance(value, (int, float)):
                reg.add_record(
                    EvidenceRecord(
                        record_id=stable_id("manifest", key, value),
                        source_id=real_source.source_id,
                        claim_type="data_volume",
                        material_system="Na-ion battery",
                        metric=metric,
                        value=float(value),
                        unit="rows",
                        confidence=0.88,
                        provenance_path="data/real/assembled/real_dataset_manifest.json",
                    )
                )
    if quality:
        q_source = EvidenceSource(
            source_id="synthetic_foundation_quality",
            title="KineticsForge synthetic foundation quality report",
            source_type="quality_report",
            uri=str(root / "data" / "cache" / "data_quality_report_foundation.json"),
            reliability=0.62,
            sha256=sha256_text(json.dumps(quality, sort_keys=True)),
        )
        reg.add_source(q_source)
        for key, value in _flatten_numeric(quality).items():
            reg.add_record(
                EvidenceRecord(
                    record_id=stable_id("quality", key, value),
                    source_id=q_source.source_id,
                    claim_type="synthetic_quality",
                    material_system="KineticsForge synthetic foundation",
                    metric=key,
                    value=float(value),
                    unit="report_value",
                    confidence=0.58,
                    provenance_path="data/cache/data_quality_report_foundation.json",
                )
            )
    if curation:
        lit_source = EvidenceSource(
            source_id="curated_literature_measurements",
            title="Curated open literature measurements",
            source_type="literature_curation",
            uri=str(root / "data" / "real" / "scraped"),
            reliability=0.74,
            sha256=sha256_text(json.dumps(curation, sort_keys=True)),
        )
        reg.add_source(lit_source)
        for key, value in _flatten_numeric(curation).items():
            reg.add_record(
                EvidenceRecord(
                    record_id=stable_id("curation", key, value),
                    source_id=lit_source.source_id,
                    claim_type="literature_curation",
                    material_system="battery literature",
                    metric=key,
                    value=float(value),
                    unit="report_value",
                    confidence=0.68,
                    provenance_path="data/real/scraped/curation_report.json",
                )
            )

    citations_csv = root / "data" / "real" / "scraped" / "literature_citations.csv"
    if citations_csv.exists():
        csv_source = EvidenceSource(
            source_id="manual_literature_citations",
            title="Manual Literature Citations Table",
            source_type="literature_csv",
            uri=str(citations_csv),
            reliability=0.85,
            sha256=sha256_file(citations_csv),
            notes="Manually compiled high-confidence literature values."
        )
        reg.add_source(csv_source)
        try:
            with citations_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    reg.add_record(
                        EvidenceRecord(
                            record_id=stable_id("lit_csv", row.get("doi", str(i)), row.get("metric", "")),
                            source_id=csv_source.source_id,
                            claim_type="literature_measurement",
                            material_system=row.get("material_system", "unknown"),
                            metric=row.get("metric", ""),
                            value=float(row.get("value", 0.0)),
                            unit=row.get("unit", ""),
                            conditions={"condition_str": row.get("condition", "")},
                            confidence=float(row.get("extraction_confidence", 0.8)),
                            provenance_path="data/real/scraped/literature_citations.csv",
                        )
                    )
        except Exception as e:
            pass

    return reg


def _flatten_numeric(payload: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(_flatten_numeric(value, name))
    return out


def default_claims(reg: EvidenceRegistry) -> List[ClaimAssessment]:
    return [
        reg.assess_claim(
            "The assembled project contains more than 10000 real Na-ion cycle summary rows.",
            metric="real_cycle_summary_rows",
            proposed_value=10000.0,
            unit="rows",
            material_system="Na-ion",
            tolerance_fraction=0.40,
        ),
        reg.assess_claim(
            "The assembled project contains more than 200000 real Na-ion time-series sample rows.",
            metric="real_timeseries_sample_rows",
            proposed_value=200000.0,
            unit="rows",
            material_system="Na-ion",
            tolerance_fraction=0.35,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="data/cache/evidence_registry_v2.json")
    parser.add_argument("--claims-csv", default="data/cache/evidence_claims_v2.csv")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    registry = build_registry_from_project(root)
    registry.save_json(root / args.out)
    assessments = default_claims(registry)
    if assessments:
        registry.save_claim_assessment_csv(root / args.claims_csv, assessments)
    print(json.dumps({"sources": len(registry.sources), "records": len(registry.records), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
