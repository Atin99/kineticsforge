"""Feature Store — V4 Architecture: Versioned, schema-validated computed features.

Every feature has provenance (DOI or cell_id or synthetic_seed).
Computed features include: Arrhenius params, SEI growth rates, leaching kinetics,
capacity fade rates, regional climate stress indices.

This is a local parquet-backed feature store. Production upgrade: replace with
a proper feature store (Feast, Tecton, or Delta Lake).
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


FEATURE_STORE_DIR = "data/feature_store"
FEATURE_CATALOG_FILE = "feature_catalog.json"


@dataclass
class FeatureSchema:
    name: str
    dtype: str
    unit: str
    description: str
    provenance_type: str  # "doi", "cell_id", "synthetic_seed", "computed"
    physics_domain: str  # "cathode", "bms", "recycling", "climate", "cross-domain"


@dataclass
class FeatureVersion:
    version: str
    created_at: str
    n_records: int
    schema: List[FeatureSchema]
    source_hash: str
    file_path: str
    notes: str = ""


@dataclass
class FeatureCatalog:
    """Master catalog of all feature sets in the store."""
    feature_sets: Dict[str, List[FeatureVersion]] = field(default_factory=dict)


class FeatureStore:
    """Local feature store backed by versioned numpy/JSON files.

    Structure:
        data/feature_store/
            feature_catalog.json      -- master catalog
            cathode_fade_rates/
                v1.npz
                v1_provenance.json
            bms_resistance_features/
                v1.npz
            ...
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.store_dir = self.root / FEATURE_STORE_DIR
        self.catalog_path = self.store_dir / FEATURE_CATALOG_FILE
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> FeatureCatalog:
        if self.catalog_path.exists():
            try:
                raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
                cat = FeatureCatalog()
                for name, versions in raw.get("feature_sets", {}).items():
                    cat.feature_sets[name] = [
                        FeatureVersion(
                            version=v["version"],
                            created_at=v["created_at"],
                            n_records=v["n_records"],
                            schema=[FeatureSchema(**s) for s in v["schema"]],
                            source_hash=v["source_hash"],
                            file_path=v["file_path"],
                            notes=v.get("notes", ""),
                        )
                        for v in versions
                    ]
                return cat
            except Exception:
                pass
        return FeatureCatalog()

    def _save_catalog(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        serializable: Dict[str, Any] = {"feature_sets": {}}
        for name, versions in self.catalog.feature_sets.items():
            serializable["feature_sets"][name] = [
                {
                    "version": v.version,
                    "created_at": v.created_at,
                    "n_records": v.n_records,
                    "schema": [asdict(s) for s in v.schema],
                    "source_hash": v.source_hash,
                    "file_path": v.file_path,
                    "notes": v.notes,
                }
                for v in versions
            ]
        self.catalog_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def register_feature_set(
        self,
        name: str,
        version: str,
        data: Dict[str, np.ndarray],
        schema: List[FeatureSchema],
        provenance: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> FeatureVersion:
        """Register a new feature set version.

        Args:
            name: Feature set name (e.g., "cathode_fade_rates")
            version: Semantic version string (e.g., "v1")
            data: Dict of column_name -> numpy array
            schema: List of FeatureSchema describing each column
            provenance: Optional provenance metadata
            notes: Free-text notes
        """
        feature_dir = self.store_dir / name
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = feature_dir / f"{version}.npz"
        np.savez_compressed(str(data_path), **data)

        # Save provenance
        if provenance:
            prov_path = feature_dir / f"{version}_provenance.json"
            prov_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

        # Compute source hash
        source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]

        n_records = max((arr.shape[0] for arr in data.values()), default=0)

        fv = FeatureVersion(
            version=version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            n_records=n_records,
            schema=schema,
            source_hash=source_hash,
            file_path=str(data_path.relative_to(self.root)),
            notes=notes,
        )

        if name not in self.catalog.feature_sets:
            self.catalog.feature_sets[name] = []
        self.catalog.feature_sets[name].append(fv)
        self._save_catalog()
        return fv

    def load(self, name: str, version: str = "latest") -> Optional[Dict[str, np.ndarray]]:
        """Load a feature set by name and version."""
        versions = self.catalog.feature_sets.get(name, [])
        if not versions:
            return None
        if version == "latest":
            target = max(versions, key=lambda v: v.created_at)
        else:
            matches = [v for v in versions if v.version == version]
            if not matches:
                return None
            target = matches[-1]
        data_path = self.root / target.file_path
        if not data_path.exists():
            return None
        loaded = np.load(str(data_path), allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}

    def list_feature_sets(self) -> Dict[str, int]:
        """List all feature set names and number of versions."""
        return {name: len(versions) for name, versions in self.catalog.feature_sets.items()}

    def summary(self) -> List[Dict[str, Any]]:
        """Summary suitable for API display."""
        rows = []
        for name, versions in self.catalog.feature_sets.items():
            for v in versions:
                rows.append({
                    "name": name,
                    "version": v.version,
                    "n_records": v.n_records,
                    "source_hash": v.source_hash,
                    "created_at": v.created_at,
                    "n_columns": len(v.schema),
                    "physics_domains": list(set(s.physics_domain for s in v.schema)),
                })
        return rows


# Predefined schemas for common feature sets
CATHODE_FADE_SCHEMA = [
    FeatureSchema("cell_id", "str", "", "Cell identifier", "cell_id", "cathode"),
    FeatureSchema("fade_rate_per_cycle", "float64", "1/cycle", "Capacity fade rate", "computed", "cathode"),
    FeatureSchema("Ea_effective_eV", "float64", "eV", "Effective activation energy from Arrhenius fit", "computed", "cathode"),
    FeatureSchema("k0_sei", "float64", "1/s", "SEI growth pre-exponential factor", "computed", "cathode"),
    FeatureSchema("temperature_K", "float64", "K", "Operating temperature", "cell_id", "cathode"),
    FeatureSchema("dataset_key", "str", "", "Source dataset identifier", "doi", "cathode"),
]

BMS_RESISTANCE_SCHEMA = [
    FeatureSchema("cell_id", "str", "", "Cell identifier", "cell_id", "bms"),
    FeatureSchema("R_ohm", "float64", "Ohm", "Ohmic resistance from EIS", "cell_id", "bms"),
    FeatureSchema("R_ct", "float64", "Ohm", "Charge transfer resistance", "cell_id", "bms"),
    FeatureSchema("R_sei", "float64", "Ohm", "SEI resistance", "cell_id", "bms"),
    FeatureSchema("sigma_warburg", "float64", "Ohm/sqrt(Hz)", "Warburg coefficient", "cell_id", "bms"),
    FeatureSchema("SOH_pct", "float64", "%", "State of health", "cell_id", "bms"),
]

CLIMATE_STRESS_SCHEMA = [
    FeatureSchema("region", "str", "", "Indian region name", "computed", "climate"),
    FeatureSchema("mean_T_C", "float64", "degC", "Annual mean temperature", "computed", "climate"),
    FeatureSchema("heat_stress_hours", "int64", "hours", "Hours with heat stress index > 0.1", "computed", "climate"),
    FeatureSchema("cold_plating_hours", "int64", "hours", "Hours with cold plating risk > 0.1", "computed", "climate"),
    FeatureSchema("source", "str", "", "IMD climate normals or NASA POWER", "computed", "climate"),
]
