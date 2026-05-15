import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.dataset_contracts import describe_array, ensure_dir, utc_now, write_json


FORMULA_RE = re.compile(
    r"\b(?:Na|Li)(?:[\d.xδ]+)?(?:(?:[A-Z][a-z]?|\([A-Za-z0-9.]+\))(?:[\d.xδ]+)?){1,12}(?:O(?:[\d.xδ]+)?)?(?:\b|[.,;:)])"
)


@dataclass
class CurationRule:
    rule_id: str
    description: str


CURATION_RULES = [
    CurationRule("preserve_raw", "Raw extracted candidates are never overwritten; curated outputs are written as new files."),
    CurationRule("require_context", "Curated electrochemical rows require sodium/lithium/battery/cathode context in title or extraction context."),
    CurationRule("cathode_capacity_range", "Specific cathode capacity rows are kept in 40-260 mAh/g unless the context clearly says high-capacity anode/conversion material, which is excluded."),
    CurationRule("retention_range", "Capacity retention rows are kept in 20-100.5 percent and must have cycle context or battery/cathode context."),
    CurationRule("recycling_range", "Metal recovery rows are kept in 1-100.5 percent and tagged as recycling evidence."),
    CurationRule("provenance_required", "Every curated row keeps DOI, title, source URL, extraction method, confidence, context hash, and context snippet."),
]


class LiteratureCurator:
    def __init__(self, root: Path):
        self.root = root
        self.scraped_dir = ensure_dir(root / "real" / "scraped")
        self.raw_path = self.scraped_dir / "literature_measurements.parquet"
        self.curated_path = self.scraped_dir / "curated_literature_measurements.parquet"
        self.prior_path = self.scraped_dir / "calibration_priors.json"
        self.report_path = self.scraped_dir / "curation_report.json"

    def load(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.raw_path)

    def reextract_formula(self, row: pd.Series) -> str:
        existing = str(row.get("formula") or "").strip()
        if existing and existing.lower() != "unknown" and len(existing) >= 4:
            return self.clean_formula(existing)
        text = " ".join([str(row.get("title") or ""), str(row.get("context") or "")])
        matches = []
        for m in FORMULA_RE.finditer(text):
            token = self.clean_formula(m.group(0))
            if self.formula_is_battery_like(token):
                matches.append(token)
        if not matches:
            return "unknown"
        return sorted(matches, key=lambda x: (not x.startswith("Na"), len(x)))[0]

    @staticmethod
    def clean_formula(formula: str) -> str:
        formula = formula.replace("δ", "x").replace("−", "-")
        formula = re.sub(r"[^A-Za-z0-9().x+-]", "", formula)
        return formula.strip(".,;:()[]")

    @staticmethod
    def formula_is_battery_like(formula: str) -> bool:
        if len(formula) < 5:
            return False
        if not formula.startswith(("Na", "Li")):
            return False
        return any(el in formula for el in ("Mn", "Fe", "Ni", "Co", "V", "P", "O"))

    @staticmethod
    def context_class(row: pd.Series) -> str:
        text = " ".join([str(row.get("title") or ""), str(row.get("context") or "")]).lower()
        if any(w in text for w in ("leaching", "recycling", "recovery", "extraction", "dissolution")):
            return "recycling"
        if any(w in text for w in ("thermal runaway", "bms", "battery pack", "precursor", "internal resistance")):
            return "bms_safety"
        if any(w in text for w in ("cathode", "layered oxide", "sodium-ion", "sodium ion", "na-ion", "electrochemical performance")):
            return "cathode"
        if "battery" in text:
            return "battery_general"
        return "unknown"

    @staticmethod
    def is_bad_context(row: pd.Series) -> bool:
        text = " ".join([str(row.get("title") or ""), str(row.get("context") or "")]).lower()
        bad = ("supercapacitor", "capacitor", "metal-organic framework", "mof", "oxygen evolution", "hydrogen evolution")
        if any(w in text for w in bad):
            return True
        if "anode" in text and "cathode" not in text:
            return True
        return False

    def curate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = row.to_dict()
            item["formula_curated"] = self.reextract_formula(row)
            item["domain_curated"] = self.context_class(row)
            item["curation_status"] = "rejected"
            item["curation_reason"] = ""
            prop = str(row.get("property_name") or "")
            value = float(row.get("value")) if pd.notna(row.get("value")) else np.nan
            confidence = float(row.get("confidence") or 0.0)
            if self.is_bad_context(row):
                item["curation_reason"] = "excluded_bad_context"
            elif item["domain_curated"] == "unknown":
                item["curation_reason"] = "unknown_domain"
            elif confidence < 0.42:
                item["curation_reason"] = "low_confidence"
            elif prop == "specific_capacity":
                if 40.0 <= value <= 260.0:
                    item["curation_status"] = "accepted"
                    item["curation_reason"] = "accepted_cathode_capacity"
                else:
                    item["curation_reason"] = "specific_capacity_outside_cathode_range"
            elif prop == "capacity_retention":
                if 20.0 <= value <= 100.5:
                    item["curation_status"] = "accepted"
                    item["curation_reason"] = "accepted_retention"
                else:
                    item["curation_reason"] = "retention_outside_range"
            elif prop == "metal_recovery":
                if 1.0 <= value <= 100.5:
                    item["curation_status"] = "accepted"
                    item["curation_reason"] = "accepted_recovery"
                else:
                    item["curation_reason"] = "recovery_outside_range"
            else:
                item["curation_reason"] = "unsupported_property"
            if item["formula_curated"] == "unknown" and item["curation_status"] == "accepted":
                item["confidence"] = max(0.0, confidence - 0.08)
            rows.append(item)
        curated = pd.DataFrame(rows)
        return curated

    def priors(self, curated: pd.DataFrame) -> Dict[str, Any]:
        accepted = curated[curated["curation_status"] == "accepted"].copy() if not curated.empty else pd.DataFrame()
        cathode = accepted[accepted["domain_curated"].isin(["cathode", "battery_general"])] if not accepted.empty else pd.DataFrame()
        retention = cathode[cathode["property_name"] == "capacity_retention"] if not cathode.empty else pd.DataFrame()
        capacity = cathode[cathode["property_name"] == "specific_capacity"] if not cathode.empty else pd.DataFrame()
        recovery = accepted[accepted["property_name"] == "metal_recovery"] if not accepted.empty else pd.DataFrame()
        capacity_fallback = capacity.empty
        retention_fallback = retention.empty
        recovery_fallback = recovery.empty
        capacity_values = capacity["value"].to_numpy(dtype=float) if not capacity.empty else np.array([125.0, 145.0, 160.0, 175.0])
        retention_values = retention["value"].to_numpy(dtype=float) if not retention.empty else np.array([78.0, 83.0, 88.0, 92.0])
        recovery_values = recovery["value"].to_numpy(dtype=float) if not recovery.empty else np.array([58.0, 73.0, 87.0])
        priors = {
            "created_at": utc_now(),
            "raw_rows": int(len(curated)),
            "accepted_rows": int(len(accepted)),
            "fallback_used": {
                "specific_capacity_mAh_g": bool(capacity_fallback),
                "capacity_retention_percent": bool(retention_fallback),
                "metal_recovery_percent": bool(recovery_fallback),
            },
            "fallback_note": "Fallback priors are engineering defaults used only when accepted literature rows are absent; they are not scraped evidence.",
            "accepted_by_property": accepted["property_name"].value_counts().to_dict() if not accepted.empty else {},
            "accepted_by_domain": accepted["domain_curated"].value_counts().to_dict() if not accepted.empty else {},
            "specific_capacity_mAh_g": describe_array(capacity_values),
            "capacity_retention_percent": describe_array(retention_values),
            "metal_recovery_percent": describe_array(recovery_values),
            "q0_center_mAh_g": float(np.clip(np.nanmedian(capacity_values), 105.0, 185.0)),
            "q0_spread_mAh_g": float(np.clip(np.nanstd(capacity_values), 8.0, 36.0)),
            "retention_center_percent": float(np.clip(np.nanmedian(retention_values), 65.0, 96.0)),
            "retention_spread_percent": float(np.clip(np.nanstd(retention_values), 3.0, 18.0)),
            "recovery_center_percent": float(np.clip(np.nanmedian(recovery_values), 50.0, 95.0)),
            "source": str(self.curated_path),
            "provenance_fields": ["doi", "title", "source_url", "context_hash", "extraction_method", "confidence", "context"],
        }
        return priors

    def report(self, curated: pd.DataFrame, priors: Dict[str, Any]) -> Dict[str, Any]:
        accepted = curated[curated["curation_status"] == "accepted"] if not curated.empty else pd.DataFrame()
        rejected = curated[curated["curation_status"] != "accepted"] if not curated.empty else pd.DataFrame()
        return {
            "created_at": utc_now(),
            "rules": [asdict(rule) for rule in CURATION_RULES],
            "raw_rows": int(len(curated)),
            "accepted_rows": int(len(accepted)),
            "rejected_rows": int(len(rejected)),
            "rejection_reasons": rejected["curation_reason"].value_counts().to_dict() if not rejected.empty else {},
            "accepted_properties": accepted["property_name"].value_counts().to_dict() if not accepted.empty else {},
            "accepted_domains": accepted["domain_curated"].value_counts().to_dict() if not accepted.empty else {},
            "priors": priors,
        }

    def run(self) -> Dict[str, Any]:
        raw = self.load()
        curated = self.curate(raw)
        priors = self.priors(curated)
        if not curated.empty:
            curated.to_parquet(self.curated_path, index=False)
        write_json(self.prior_path, priors)
        report = self.report(curated, priors)
        write_json(self.report_path, report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate scraped literature evidence into trusted calibration priors.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = LiteratureCurator(Path(args.root).resolve()).run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
