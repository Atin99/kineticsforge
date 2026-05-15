import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ENERGY_EV_TO_J_PER_MOL = 96485.33212
GAS_CONSTANT = 8.314462618
FARADAY = 96485.33212


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    source_type: str
    url: str
    license_note: str
    access_note: str
    expected_scale: str
    priority: int
    chemistry_scope: Tuple[str, ...] = field(default_factory=tuple)
    default_files: Tuple[str, ...] = field(default_factory=tuple)
    citation: str = ""


@dataclass
class DatasetShard:
    shard_id: str
    domain: str
    path: str
    format: str
    rows: int
    bytes: int
    sha256: str
    schema_version: str
    generated_at: str
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    project: str
    schema_version: str
    created_at: str
    profile: str
    seed: int
    root: str
    shards: List[DatasetShard] = field(default_factory=list)
    sources: List[SourceSpec] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_shard(self, shard: DatasetShard) -> None:
        self.shards.append(shard)

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def to_json(self) -> Dict[str, Any]:
        out = asdict(self)
        out["total_rows"] = int(sum(s.rows for s in self.shards))
        out["total_bytes"] = int(sum(s.bytes for s in self.shards))
        out["domains"] = sorted({s.domain for s in self.shards})
        return out


@dataclass
class ValidationIssue:
    severity: str
    domain: str
    field: str
    message: str
    observed: Any = None
    expected: Any = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def infer_rows_from_npz(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    best = 0
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and arr.shape:
            n = int(np.prod(arr.shape))
            best = max(best, n)
    data.close()
    return best


def register_file(
    manifest: DatasetManifest,
    path: Path,
    domain: str,
    rows: int,
    schema_version: str,
    notes: Optional[Dict[str, Any]] = None,
) -> DatasetShard:
    shard = DatasetShard(
        shard_id=f"{domain}:{path.stem}",
        domain=domain,
        path=str(path),
        format=path.suffix.lower().lstrip("."),
        rows=int(rows),
        bytes=file_size(path),
        sha256=sha256_file(path),
        schema_version=schema_version,
        generated_at=utc_now(),
        notes=notes or {},
    )
    manifest.add_shard(shard)
    return shard


def bounded_sigmoid(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return low + (high - low) / (1.0 + np.exp(-x))


def arrhenius(k0: float, ea_ev: float, temp_k: np.ndarray | float) -> np.ndarray:
    return k0 * np.exp(-(ea_ev * ENERGY_EV_TO_J_PER_MOL) / (GAS_CONSTANT * np.asarray(temp_k)))


def stable_noise(rng: np.random.Generator, n: int, sigma: float, phi: float = 0.82) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n, dtype=np.float32)
    prev = 0.0
    for i in range(n):
        prev = phi * prev + eps[i]
        out[i] = prev
    return out


def finite_slope(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return 0.0
    x = np.asarray(x, dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def monotonicity_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 1.0
    return float(np.mean(np.diff(values) <= 1e-8))


def describe_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": math.nan, "p05": math.nan, "mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def validate_range(
    issues: List[ValidationIssue],
    domain: str,
    field: str,
    values: np.ndarray,
    low: float,
    high: float,
    severity: str = "error",
) -> None:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        issues.append(ValidationIssue(severity, domain, field, "field contains no finite values", None, [low, high]))
        return
    observed_low = float(np.min(finite))
    observed_high = float(np.max(finite))
    if observed_low < low or observed_high > high:
        issues.append(
            ValidationIssue(
                severity,
                domain,
                field,
                "field outside expected physical range",
                {"min": observed_low, "max": observed_high},
                {"min": low, "max": high},
            )
        )


def validate_finite(
    issues: List[ValidationIssue],
    domain: str,
    field: str,
    values: np.ndarray,
    severity: str = "error",
) -> None:
    arr = np.asarray(values)
    total = int(arr.size)
    if total == 0:
        issues.append(ValidationIssue(severity, domain, field, "field is empty", 0, ">0 values"))
        return
    finite = int(np.isfinite(arr).sum())
    if finite != total:
        issues.append(
            ValidationIssue(
                severity,
                domain,
                field,
                "field contains NaN or infinite values",
                {"finite": finite, "total": total, "nonfinite": total - finite},
                "all finite",
            )
        )


def write_validation_report(path: Path, issues: Sequence[ValidationIssue], metrics: Dict[str, Any]) -> None:
    payload = {
        "created_at": utc_now(),
        "status": "pass" if not any(i.severity == "error" for i in issues) else "fail",
        "issue_count": len(issues),
        "errors": sum(1 for i in issues if i.severity == "error"),
        "warnings": sum(1 for i in issues if i.severity == "warning"),
        "issues": [asdict(i) for i in issues],
        "metrics": metrics,
    }
    write_json(path, payload)


def source_catalog() -> List[SourceSpec]:
    return [
        SourceSpec(
            source_id="openalex_literature_scrape",
            title="OpenAlex open-access scholarly works scrape",
            source_type="metadata_and_open_access_text_scrape",
            url="https://api.openalex.org/works",
            license_note="OpenAlex metadata is open; downloaded full text must follow each publisher/repository license.",
            access_note="Free API. Query open-access papers, store DOI, landing URL, PDF URL, extraction context hash, and confidence for every measurement.",
            expected_scale="Hundreds to thousands of papers per query family; extraction quality depends on available OA full text and table formatting.",
            priority=0,
            chemistry_scope=("Na-ion", "Li-ion", "battery recycling", "battery safety"),
            default_files=(),
            citation="OpenAlex API and each original paper DOI.",
        ),
        SourceSpec(
            source_id="crossref_metadata_scrape",
            title="Crossref bibliographic metadata scrape",
            source_type="metadata_scrape",
            url="https://api.crossref.org/works",
            license_note="Crossref bibliographic metadata is factual/open; full text links retain original publisher terms.",
            access_note="Free REST API. Used as metadata fallback and DOI provenance layer.",
            expected_scale="Large scholarly metadata index with title, DOI, venue, abstract, license, and links where deposited.",
            priority=0,
            chemistry_scope=("Na-ion", "Li-ion", "battery materials", "recycling"),
            default_files=(),
            citation="Crossref REST API and each original paper DOI.",
        ),
        SourceSpec(
            source_id="batterylife_processed_v10",
            title="BatteryLife Processed",
            source_type="processed_open_dataset",
            url="https://zenodo.org/records/18646655",
            license_note="Open Zenodo dataset; cite BatteryLife and original source datasets.",
            access_note="No paid API required. Default recommended files are NA-ion.zip, Life labels.zip, READMEs.zip, SNL.zip, HNEI.zip, CALCE.zip.",
            expected_scale="990 batteries, four dataset families, 30.9 GB if fully downloaded; NA-ion subset is about 289 MB.",
            priority=1,
            chemistry_scope=("Na-ion", "Li-ion", "Zn-ion", "CALB"),
            default_files=("NA-ion.zip", "Life labels.zip", "READMEs.zip"),
            citation="Tan et al., BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction, KDD 2025.",
        ),
        SourceSpec(
            source_id="nasa_pcoe_battery_aging",
            title="NASA PCoE Li-ion Battery Aging Datasets",
            source_type="raw_open_dataset",
            url="https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
            license_note="NASA public portal; verify downstream license terms before redistribution.",
            access_note="Free direct download from NASA PHM S3 mirror; MATLAB structures require parsing.",
            expected_scale="Commercial 18650 Li-ion cells with charge, discharge, impedance, temperature, voltage, current, and capacity.",
            priority=2,
            chemistry_scope=("Li-ion",),
            default_files=("NASA_5_Battery_Data_Set.zip",),
            citation="NASA Ames Prognostics Center of Excellence battery aging datasets.",
        ),
        SourceSpec(
            source_id="isu_ilcc_battery_aging",
            title="ISU-ILCC Battery Aging Dataset",
            source_type="raw_open_dataset",
            url="https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234",
            license_note="Figshare record lists CC BY 4.0; cite the dataset and upstream investigators.",
            access_note="Free direct Figshare downloads. Start with capacity_fade and Q_interpolated; RPT_json is larger and should be normalized as a separate shard.",
            expected_scale="Hundreds of Li-ion polymer cells cycled under dozens of charge/discharge/depth-of-discharge stress conditions.",
            priority=2,
            chemistry_scope=("Li-ion polymer", "capacity fade", "cycle aging"),
            default_files=("Valid_cells.csv", "README_V2.0.pdf", "capacity_fade.zip", "Q_interpolated.zip"),
            citation="ISU-ILCC Battery Aging Dataset, Iowa State University / Iowa Lakes Community College / UConn REIL.",
        ),
        SourceSpec(
            source_id="oxford_battery_degradation_1",
            title="Oxford Battery Degradation Dataset 1",
            source_type="raw_open_dataset",
            url="https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac",
            license_note="Oxford ORA dataset; follow ORA record terms.",
            access_note="Free ORA download; useful for degradation curve validation.",
            expected_scale="Longitudinal commercial pouch-cell degradation records.",
            priority=3,
            chemistry_scope=("Li-ion",),
            default_files=(),
            citation="Oxford Battery Degradation Dataset 1.",
        ),
        SourceSpec(
            source_id="battery_archive",
            title="BatteryArchive.org",
            source_type="repository",
            url="https://www.batteryarchive.org/",
            license_note="Repository-specific terms; some complete CSV files require email request.",
            access_note="Free public repository; manually request full CSV bundles when needed.",
            expected_scale="Standardized performance, safety, and degradation datasets from multiple labs.",
            priority=4,
            chemistry_scope=("Li-ion", "Na-ion", "abuse_testing"),
            default_files=(),
            citation="Battery Archive public battery data repository.",
        ),
        SourceSpec(
            source_id="nasa_power_hourly_climate",
            title="NASA POWER hourly climate API",
            source_type="open_climate_api",
            url="https://power.larc.nasa.gov/api/temporal/hourly/point",
            license_note="Open NASA data service; cite NASA POWER and record request parameters.",
            access_note="No paid API required. Pull hourly temperature, humidity, irradiance, and wind for deployment coordinates.",
            expected_scale="Hourly climate series for selected point locations; use for regional battery thermal stress profiles.",
            priority=5,
            chemistry_scope=("climate", "thermal_degradation", "India"),
            default_files=(),
            citation="NASA POWER Project hourly API.",
        ),
        SourceSpec(
            source_id="copernicus_era5_hourly",
            title="ERA5 hourly single-level climate reanalysis",
            source_type="free_registration_climate_api",
            url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
            license_note="Copernicus/ECMWF license applies; preserve request metadata and citation.",
            access_note="Free CDS account required. Pull hourly 2 m temperature, dew point, and radiation fields for regional stress testing.",
            expected_scale="Hourly climate reanalysis from 1940 onward on a global grid.",
            priority=6,
            chemistry_scope=("climate", "thermal_degradation", "India"),
            default_files=(),
            citation="Copernicus Climate Change Service ERA5.",
        ),
        SourceSpec(
            source_id="imd_api",
            title="India Meteorological Department API platform",
            source_type="official_weather_api",
            url="https://api.imd.gov.in/",
            license_note="Follow IMD API access terms and attribution requirements.",
            access_note="Official Indian weather observations, forecasts, warnings, and specialized bulletins; API access may require registration.",
            expected_scale="Real-time and historical India weather products depending on approved API access.",
            priority=7,
            chemistry_scope=("climate", "India", "thermal_safety"),
            default_files=(),
            citation="India Meteorological Department API.",
        ),
        SourceSpec(
            source_id="materials_project",
            title="Materials Project",
            source_type="free_academic_api",
            url="https://materialsproject.org/api",
            license_note="Free API key required; respect Materials Project terms.",
            access_note="Use MP_API_KEY. Pull Na-Mn-Fe-O thermodynamic priors, voltages, hull energies, and composition features.",
            expected_scale="Tens of thousands of computed structures depending on query scope.",
            priority=5,
            chemistry_scope=("Na-Mn-Fe-O", "Li-ion", "solid_state_materials"),
            default_files=(),
            citation="Materials Project database.",
        ),
    ]


def write_source_catalog(path: Path) -> None:
    write_json(path, {"created_at": utc_now(), "sources": [asdict(s) for s in source_catalog()]})
