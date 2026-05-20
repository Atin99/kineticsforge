"""KineticsForge production CPU server.

The default runtime is dependency-light for public deployment, but it
opportunistically runs trained PyTorch checkpoints when torch is installed.

Run: python serve.py
"""
import os
import sys
import time
import math
import uuid
import json
import csv
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PYDEPS = ROOT / "_pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("pip install fastapi uvicorn numpy pydantic python-multipart", file=sys.stderr)
    raise

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kineticsforge")

try:
    from api.chat_assistant import answer_chat
except Exception as exc:  # pragma: no cover - keeps the production server bootable
    logger.warning("chat assistant unavailable: %s", exc)
    answer_chat = None

try:
    from data.byod_pipeline import analyze_upload
except Exception as exc:  # pragma: no cover - keeps the production server bootable
    logger.warning("BYOD analyzer unavailable: %s", exc)
    analyze_upload = None

PRODUCTION = os.environ.get("KF_ENV", "").lower() == "production" or os.environ.get("RENDER", "").lower() == "true"
app = FastAPI(
    title="KineticsForge",
    docs_url=None if PRODUCTION else "/docs",
    redoc_url=None if PRODUCTION else "/redoc",
    openapi_url=None if PRODUCTION else "/openapi.json",
)


def _csv_env(name: str, default: str = "") -> List[str]:
    value = os.environ.get(name, default)
    return [part.strip() for part in value.split(",") if part.strip()]


allowed_origins = _csv_env("KF_CORS_ORIGINS", "*")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])
allowed_hosts = _csv_env("KF_ALLOWED_HOSTS")
if allowed_hosts and "*" not in allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# Diffusion-limited SEI: capacity loss follows sqrt(N) cumulative growth.
# dQ/dN_SEI = Q * sei_arrhenius(T) * SEI_GROWTH_COEFF * (sqrt(N) - sqrt(N-1)) * stress
# Arrhenius factor is normalized to reference T=318.15K so the coefficient stays physical.
SEI_GROWTH_COEFF = 0.048  # Calibrated diffusion-limited SEI coefficient (replaces old dual-term 1e10 + sqrt model)
SEI_REF_EA = 0.56  # SEI activation energy [eV] — used for temperature scaling
SEI_REF_TEMP = 318.15  # Reference temperature [K] for Arrhenius normalization
GLOBAL_DEGRADATION_SCALE = 0.052
JT_LOSS_COEFF = 6.5e-3
DESOLV_LOSS_COEFF = 2.5e-4
BV_RATE_LOSS_COEFF = 1.2e-4
RESIDUAL_LOSS_COEFF = 1.0e-5
RECYCLING_MC_SAMPLES = 200

# In-memory rate limiter for the deployed server
_request_counts: Dict[str, list] = {}
RATE_LIMIT_RPM = 120
RATE_LIMIT_MAX_KEYS = int(os.environ.get("KF_RATE_LIMIT_MAX_KEYS", "5000"))
RATE_LIMIT_PRUNE_INTERVAL = float(os.environ.get("KF_RATE_LIMIT_PRUNE_INTERVAL_SECONDS", "60"))
_last_rate_prune = 0.0
MAX_UPLOAD_BYTES = int(os.environ.get("KF_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_BATCH_FILES = int(os.environ.get("KF_BATCH_MAX_FILES", "50"))
MAX_BATCH_TOTAL_BYTES = int(os.environ.get("KF_BATCH_MAX_MB", "80")) * 1024 * 1024
SESSION_TTL_SECONDS = int(os.environ.get("KF_SESSION_TTL_SECONDS", str(24 * 60 * 60)))
_byod_sessions: Dict[str, Dict[str, Any]] = {}
CHECKPOINT_DIR = ROOT / "checkpoints" / "trained"
CHECKPOINT_MANIFEST_PATH = CHECKPOINT_DIR / "checkpoint_manifest.json"
_MODEL_HUB: Any = None
_MODEL_HUB_ERROR: Optional[str] = None


def _checkpoint_bundle_needed() -> bool:
    if os.environ.get("KF_CHECKPOINT_BUNDLE_FORCE", "").lower() in {"1", "true", "yes"}:
        return True
    return not CHECKPOINT_MANIFEST_PATH.exists() or not any(CHECKPOINT_DIR.glob("*.pt"))


def _extract_checkpoint_bundle(path: Path) -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    allowed_json = {"checkpoint_manifest.json", "tracker.json"}
    extracted = 0
    max_unpacked = int(os.environ.get("KF_CHECKPOINT_BUNDLE_UNPACKED_MB", "256")) * 1024 * 1024
    unpacked = 0
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            base = os.path.basename(info.filename)
            if not base:
                continue
            is_weight = base.endswith(("_best.pt", "_final.pt", "_resume.pt"))
            is_manifest = base in allowed_json
            if not (is_weight or is_manifest):
                continue
            unpacked += int(info.file_size)
            if unpacked > max_unpacked:
                raise RuntimeError("Checkpoint bundle exceeds safe unpack limit")
            target = CHECKPOINT_DIR / base
            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1
    return extracted


def _ensure_checkpoint_bundle() -> None:
    url = os.environ.get("KF_CHECKPOINT_BUNDLE_URL", "").strip()
    if not url or not _checkpoint_bundle_needed():
        return
    max_bytes = int(os.environ.get("KF_CHECKPOINT_BUNDLE_MB", "128")) * 1024 * 1024
    bundle_path = ROOT / "checkpoints" / "_checkpoint_bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=float(os.environ.get("KF_CHECKPOINT_BUNDLE_TIMEOUT", "30"))) as response:
            length = int(response.headers.get("Content-Length") or 0)
            if length and length > max_bytes:
                raise RuntimeError("Checkpoint bundle is larger than KF_CHECKPOINT_BUNDLE_MB")
            total = 0
            with bundle_path.open("wb") as dst:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError("Checkpoint bundle download exceeded KF_CHECKPOINT_BUNDLE_MB")
                    dst.write(chunk)
        extracted = _extract_checkpoint_bundle(bundle_path)
        logger.info("checkpoint bundle restored %s artifacts", extracted)
    except Exception as exc:
        logger.warning("checkpoint bundle restore failed: %s", exc)
    finally:
        try:
            bundle_path.unlink(missing_ok=True)
        except OSError:
            pass


_ensure_checkpoint_bundle()


MODEL_REGISTRY = [
    {"id": "M1_CathodeUDE", "name": "Cathode UDE", "type": "UDE physics plus residual", "status": "trained_checkpoint", "params": 45909, "checkpoint": "cathode_ude"},
    {"id": "M2_SOH", "name": "SOH Estimator", "type": "MLP regression", "status": "trained_checkpoint", "params": 32737, "checkpoint": "soh"},
    {"id": "M3_CycleLife", "name": "Cycle Life", "type": "Classifier", "status": "trained_checkpoint", "params": 37156, "checkpoint": "cycle_life"},
    {"id": "M4_FadeRate", "name": "Fade Rate", "type": "MLP regression", "status": "trained_checkpoint", "params": 7329, "checkpoint": "fade_rate"},
    {"id": "M5_BMS_TGN", "name": "BMS Pack Graph", "type": "Thermal graph risk", "status": "label_gate", "params": 11020, "checkpoint": "bms_tgn"},
    {"id": "M6_RUL", "name": "RUL Predictor", "type": "MLP regression", "status": "trained_checkpoint", "params": 34753, "checkpoint": "rul"},
    {"id": "M7_Anomaly", "name": "Anomaly AE", "type": "Autoencoder", "status": "trained_checkpoint", "params": 31551, "checkpoint": "anomaly_ae"},
    {"id": "M8_Joint_SOH_RUL", "name": "Joint SOH+RUL", "type": "Multi-task MLP", "status": "trained_checkpoint", "params": 141907, "checkpoint": "joint_soh_rul"},
    {"id": "M9_KneeDetect", "name": "Knee Detector", "type": "Conv1D detector", "status": "trained_checkpoint", "params": 219233, "checkpoint": "knee_detect"},
    {"id": "M10_ChemRank", "name": "Chem Ranker", "type": "Embedding ranker", "status": "trained_checkpoint", "params": 7809, "checkpoint": "chem_rank"},
    {"id": "M11_ElectrolyteHealth", "name": "Electrolyte Health", "type": "EIS diagnostics", "status": "trained_checkpoint", "params": 8819, "checkpoint": "electrolyte_health"},
    {"id": "M12_Replenishability", "name": "Replenishability", "type": "Recovery preview", "status": "research_preview", "params": 9554, "checkpoint": "replenishability"},
    {"id": "M13_ChemIdentifier", "name": "Chem Identifier", "type": "Early-cycle classifier", "status": "trained_checkpoint", "params": 80552, "checkpoint": "chem_identifier"},
    {"id": "M14_FormationProtocol", "name": "Formation Protocol", "type": "Formation quality", "status": "research_preview", "params": 19747, "checkpoint": "formation_protocol"},
]


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _prune_rate_limit(now: float) -> None:
    global _last_rate_prune
    if now - _last_rate_prune < RATE_LIMIT_PRUNE_INTERVAL:
        return
    cutoff = now - 60.0
    for key in list(_request_counts.keys()):
        events = _request_counts.get(key, [])
        while events and events[0] < cutoff:
            events.pop(0)
        if not events:
            _request_counts.pop(key, None)
    if len(_request_counts) > RATE_LIMIT_MAX_KEYS:
        for key in sorted(_request_counts, key=lambda k: _request_counts[k][-1] if _request_counts[k] else 0.0)[: len(_request_counts) - RATE_LIMIT_MAX_KEYS]:
            _request_counts.pop(key, None)
    _last_rate_prune = now


def _check_rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.time()
    _prune_rate_limit(now)
    events = _request_counts.setdefault(key, [])
    cutoff = now - 60.0
    while events and events[0] < cutoff:
        events.pop(0)
    if len(events) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    events.append(now)


@app.middleware("http")
async def production_product_guards(request: Request, call_next):
    if request.url.path not in {"/", "/health"} and not request.url.path.startswith("/app"):
        try:
            _check_rate_limit(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Periodically prune expired BYOD sessions to prevent memory leaks
    now = time.time()
    if now - _last_rate_prune < 5.0:  # piggyback on rate limiter timing
        pass
    elif _byod_sessions:
        _prune_sessions()
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


def _checkpoint_path(base: str) -> Optional[Path]:
    for root in (ROOT / "checkpoints" / "trained", ROOT / "checkpoints"):
        for suffix in ("best", "final", "resume"):
            path = root / f"{base}_{suffix}.pt"
            if path.exists():
                return path
    return None


def _load_checkpoint_manifest() -> Dict[str, Any]:
    if not CHECKPOINT_MANIFEST_PATH.exists():
        return {"files": [], "source_zips": []}
    try:
        data = json.loads(CHECKPOINT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("checkpoint manifest unreadable: %s", CHECKPOINT_MANIFEST_PATH)
        return {"files": [], "source_zips": []}
    if not isinstance(data, dict):
        return {"files": [], "source_zips": []}
    data.setdefault("files", [])
    data.setdefault("source_zips", [])
    return data


def _manifest_by_file() -> Dict[str, Dict[str, Any]]:
    manifest = _load_checkpoint_manifest()
    out: Dict[str, Dict[str, Any]] = {}
    for item in manifest.get("files", []):
        if isinstance(item, dict) and item.get("file"):
            out[str(item["file"])] = item
    return out


def _public_source_zips(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifact_counts: Dict[str, int] = {}
    for file_meta in manifest.get("files", []):
        if isinstance(file_meta, dict) and file_meta.get("source_name"):
            name = str(file_meta["source_name"])
            artifact_counts[name] = artifact_counts.get(name, 0) + 1
    out = []
    for item in manifest.get("source_zips", []):
        if isinstance(item, dict):
            name = item.get("name")
            out.append({"name": name, "mtime": item.get("mtime"), "artifacts": artifact_counts.get(str(name), 0)})
    return out


def _model_registry_payload() -> List[Dict[str, Any]]:
    out = []
    provenance = _manifest_by_file()
    for item in MODEL_REGISTRY:
        ckpt = _checkpoint_path(item["checkpoint"])
        row = dict(item)
        row["checkpoint_present"] = ckpt is not None
        row["checkpoint_file"] = str(ckpt.relative_to(ROOT)) if ckpt else None
        row["runtime"] = "production_cpu_checkpoint_optional"
        row["params_basis"] = "architecture parameter count; not a fabricated marketing estimate"
        if ckpt:
            meta = provenance.get(ckpt.name)
            row["checkpoint_sha256"] = meta.get("sha256") if meta else None
            row["checkpoint_source_zip"] = meta.get("source_name") if meta else None
            row["checkpoint_source_member"] = meta.get("source_member") if meta else None
        out.append(row)
    return out


def _checkpoint_inference_enabled() -> bool:
    return os.environ.get("KF_DISABLE_TORCH_INFERENCE", "").lower() not in {"1", "true", "yes"}


def _get_model_hub() -> Any:
    global _MODEL_HUB, _MODEL_HUB_ERROR
    if _MODEL_HUB is not None:
        return _MODEL_HUB
    if not _checkpoint_inference_enabled():
        _MODEL_HUB_ERROR = "disabled by KF_DISABLE_TORCH_INFERENCE"
        return None
    try:
        from inference.engine import ModelHub
        _MODEL_HUB = ModelHub()
        _MODEL_HUB_ERROR = None
        return _MODEL_HUB
    except Exception as exc:
        _MODEL_HUB_ERROR = f"{exc.__class__.__name__}: {exc}"
        logger.warning("checkpoint inference unavailable: %s", _MODEL_HUB_ERROR)
        return None


def _capacity_history(result: Dict[str, Any]) -> List[float]:
    hist = []
    for row in result.get("cycle_summary", []):
        value = row.get("discharge_capacity_ah")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            hist.append(float(value))
    if not hist:
        soh = ((result.get("predictions") or {}).get("soh"))
        hist = [float(soh)] if isinstance(soh, (int, float)) and math.isfinite(float(soh)) else [1.0]
    first = max(abs(hist[0]), 1e-12)
    return [float(v) / first for v in hist]


def _sanitize_checkpoint_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean_models: Dict[str, Any] = {}
    for model_id, item in (payload.get("models") or {}).items():
        if not isinstance(item, dict):
            continue
        clean = {k: v for k, v in item.items() if k != "checkpoint"}
        if item.get("checkpoint"):
            clean["checkpoint_file"] = os.path.basename(str(item["checkpoint"]))
        clean_models[model_id] = clean
    return {
        "status": "ok",
        "runtime": payload.get("runtime"),
        "mask_present": payload.get("mask_present"),
        "models": clean_models,
    }


def _merge_checkpoint_outputs(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    model_outputs = result.setdefault("predictions", {}).setdefault("model_outputs", {})
    for model_id, item in (payload.get("models") or {}).items():
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        target = model_outputs.setdefault(model_id, {})
        target["checkpoint_source"] = "trained_forward"
        if item.get("checkpoint"):
            target["checkpoint_file"] = os.path.basename(str(item["checkpoint"]))
        for key, value in item.items():
            if key in {"status", "checkpoint"}:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                target[f"checkpoint_{key}"] = value
    result.setdefault("predictions", {})["inference_mode"] = "checkpoint_plus_rules"


def _run_checkpoint_inference(result: Dict[str, Any], require_checkpoint: bool = False) -> Dict[str, Any]:
    hub = _get_model_hub()
    if hub is None:
        detail = _MODEL_HUB_ERROR or "checkpoint runtime unavailable"
        if require_checkpoint:
            raise HTTPException(status_code=503, detail=f"Checkpoint inference unavailable: {detail}")
        result["checkpoint_inference"] = {"status": "unavailable", "detail": detail}
        result.setdefault("predictions", {})["inference_mode"] = "rules_only"
        return result
    try:
        from inference.engine import predict_byod_features
        payload = predict_byod_features(
            hub,
            result.get("feature_vector") or [],
            result.get("feature_mask") or [],
            capacity_history=_capacity_history(result),
        )
        _merge_checkpoint_outputs(result, payload)
        result["checkpoint_inference"] = _sanitize_checkpoint_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("checkpoint inference failed")
        if require_checkpoint:
            raise HTTPException(status_code=500, detail=f"Checkpoint inference failed: {exc}") from exc
        result["checkpoint_inference"] = {"status": "error", "detail": f"{exc.__class__.__name__}: {exc}"}
        result.setdefault("predictions", {})["inference_mode"] = "rules_only"
    return result


def _store_byod_result(result: Dict[str, Any]) -> str:
    _prune_sessions()
    session_id = str(uuid.uuid4())
    _byod_sessions[session_id] = {"created_at": time.time(), "result": result}
    return session_id


def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    pred = result.get("predictions") or {}
    models = pred.get("model_outputs") or {}
    m7 = models.get("M7_Anomaly") if isinstance(models.get("M7_Anomaly"), dict) else {}
    m11 = models.get("M11_ElectrolyteHealth") if isinstance(models.get("M11_ElectrolyteHealth"), dict) else {}
    m13 = models.get("M13_ChemIdentifier") if isinstance(models.get("M13_ChemIdentifier"), dict) else {}
    return {
        "filename": result.get("filename"),
        "rows_read": result.get("rows_read"),
        "rows_available": result.get("rows_available"),
        "schema_format": (result.get("schema") or {}).get("format"),
        "schema_score": (result.get("schema") or {}).get("score"),
        "features_present": int(sum(1 for x in result.get("feature_mask", []) if x)),
        "soh": pred.get("soh"),
        "confidence": pred.get("confidence"),
        "cycle_80_estimate": pred.get("cycle_80_estimate"),
        "fade_fraction_per_cycle": pred.get("fade_fraction_per_cycle"),
        "inference_mode": pred.get("inference_mode", "rules_only"),
        "checkpoint_status": (result.get("checkpoint_inference") or {}).get("status"),
        "anomaly_score": m7.get("checkpoint_anomaly_score", m7.get("anomaly_score")),
        "sodium_plating_probability": m11.get("checkpoint_sodium_plating_probability", m11.get("sodium_plating_probability")),
        "chemistry": m13.get("predicted_family", m13.get("checkpoint_class_id", "unknown")),
        "warnings": result.get("warnings", []),
    }


def _feature_delta(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    fa = a.get("features") or {}
    fb = b.get("features") or {}
    rows = []
    for key in sorted(set(fa) & set(fb)):
        va, vb = fa.get(key), fb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and math.isfinite(float(va)) and math.isfinite(float(vb)):
            rows.append({
                "feature": key,
                "a": round(float(va), 8),
                "b": round(float(vb), 8),
                "delta": round(float(vb) - float(va), 8),
                "relative_delta": round((float(vb) - float(va)) / max(abs(float(va)), 1e-12), 6),
            })
    rows.sort(key=lambda row: abs(row["relative_delta"]), reverse=True)
    return rows


def _compare_byod_results(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    sa, sb = _result_summary(a), _result_summary(b)
    soh_delta = None
    if isinstance(sa.get("soh"), (int, float)) and isinstance(sb.get("soh"), (int, float)):
        soh_delta = round(float(sb["soh"]) - float(sa["soh"]), 5)
    fade_delta = None
    if isinstance(sa.get("fade_fraction_per_cycle"), (int, float)) and isinstance(sb.get("fade_fraction_per_cycle"), (int, float)):
        fade_delta = round(float(sb["fade_fraction_per_cycle"]) - float(sa["fade_fraction_per_cycle"]), 8)
    cycle80_delta = None
    if isinstance(sa.get("cycle_80_estimate"), (int, float)) and isinstance(sb.get("cycle_80_estimate"), (int, float)):
        cycle80_delta = round(float(sb["cycle_80_estimate"]) - float(sa["cycle_80_estimate"]), 1)
    decision = "no clear winner"
    if soh_delta is not None:
        if soh_delta > 0.015:
            decision = "file_b looks healthier on SOH"
        elif soh_delta < -0.015:
            decision = "file_a looks healthier on SOH"
    if fade_delta is not None:
        if fade_delta > 0 and decision == "file_b looks healthier on SOH":
            decision = "mixed: file_b has higher SOH but worse fade slope"
        elif fade_delta < 0 and decision == "file_a looks healthier on SOH":
            decision = "mixed: file_a has higher SOH but worse fade slope"
    return {
        "a": sa,
        "b": sb,
        "delta": {
            "soh": soh_delta,
            "fade_fraction_per_cycle": fade_delta,
            "cycle_80_estimate": cycle80_delta,
        },
        "top_feature_deltas": _feature_delta(a, b)[:20],
        "decision": decision,
        "use": "A/B triage only. Confirm with matched test protocols before treating this as a cell-quality conclusion.",
    }


def _batch_report(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [_result_summary(item["result"]) for item in items]
    soh_vals = np.array([s["soh"] for s in summaries if isinstance(s.get("soh"), (int, float))], dtype=float)
    fade_vals = np.array([s["fade_fraction_per_cycle"] for s in summaries if isinstance(s.get("fade_fraction_per_cycle"), (int, float))], dtype=float)
    stats = {
        "cells": len(summaries),
        "soh_mean": round(float(np.mean(soh_vals)), 5) if soh_vals.size else None,
        "soh_std": round(float(np.std(soh_vals)), 5) if soh_vals.size else None,
        "fade_mean": round(float(np.mean(fade_vals)), 8) if fade_vals.size else None,
        "fade_std": round(float(np.std(fade_vals)), 8) if fade_vals.size else None,
    }
    soh_mean = float(np.mean(soh_vals)) if soh_vals.size else None
    soh_std = float(np.std(soh_vals)) if soh_vals.size else 0.0
    outliers = []
    for item, summary in zip(items, summaries):
        reasons = []
        soh = summary.get("soh")
        if isinstance(soh, (int, float)) and soh_mean is not None and soh_std > 1e-9 and abs(float(soh) - soh_mean) > 2.0 * soh_std:
            reasons.append("SOH outside 2 sigma")
        anomaly = summary.get("anomaly_score")
        if isinstance(anomaly, (int, float)) and float(anomaly) >= 0.60:
            reasons.append("high anomaly score")
        conf = summary.get("confidence")
        if isinstance(conf, (int, float)) and float(conf) < 0.45:
            reasons.append("low confidence")
        if summary.get("warnings"):
            reasons.append("parser warnings")
        if reasons:
            outliers.append({"session_id": item["session_id"], "filename": summary.get("filename"), "reasons": reasons, "summary": summary})
    return {
        "stats": stats,
        "summaries": [{"session_id": item["session_id"], **summary} for item, summary in zip(items, summaries)],
        "outliers": outliers,
        "decision": "investigate outliers before continuing the batch" if outliers else "batch looks internally consistent for first-pass triage",
    }


def _finite_series(values: Optional[List[float]]) -> List[float]:
    if not values:
        return []
    out = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _webhook_cycle_decision(req: "CyclerWebhookRequest") -> Dict[str, Any]:
    voltages = _finite_series(req.voltage_V)
    temps = _finite_series(req.temperature_C)
    currents = _finite_series(req.current_A)
    soh = req.soh
    if isinstance(soh, (int, float)) and float(soh) > 1.2:
        soh = float(soh) / 100.0
    if soh is None and req.discharge_capacity_ah and req.nominal_capacity_ah:
        soh = float(req.discharge_capacity_ah) / max(abs(float(req.nominal_capacity_ah)), 1e-12)
    ce = req.coulombic_efficiency
    if ce is None and req.charge_capacity_ah and req.discharge_capacity_ah:
        ce = float(req.discharge_capacity_ah) / max(abs(float(req.charge_capacity_ah)), 1e-12)

    hard_stop = []
    investigate = []
    if isinstance(soh, (int, float)):
        if float(soh) < clamp(req.stop_soh, 0.1, 1.1):
            hard_stop.append("SOH below stop threshold")
        elif float(soh) < clamp(req.warn_soh, 0.1, 1.1):
            investigate.append("SOH below warning threshold")
    else:
        investigate.append("SOH unavailable")
    if ce is not None:
        ce_val = float(ce) / 100.0 if float(ce) > 1.2 else float(ce)
        if ce_val < 0.96:
            investigate.append("low coulombic efficiency")
    if temps:
        t_max = max(temps)
        if t_max > clamp(req.max_temperature_C, 30.0, 120.0):
            hard_stop.append("temperature above safety limit")
        elif t_max > clamp(req.max_temperature_C, 30.0, 120.0) - 5.0:
            investigate.append("temperature close to limit")
    if voltages:
        v_min, v_max = min(voltages), max(voltages)
        if v_min < clamp(req.min_voltage_V, 0.0, 5.0) or v_max > clamp(req.max_voltage_V, 2.0, 6.0):
            hard_stop.append("voltage outside configured limits")

    recommendation = "continue"
    reasons = []
    if hard_stop:
        recommendation = "stop"
        reasons = hard_stop
    elif investigate:
        recommendation = "investigate"
        reasons = investigate
    return {
        "station_id": req.station_id,
        "channel_id": req.channel_id,
        "cell_id": req.cell_id,
        "cycle": req.cycle,
        "recommendation": recommendation,
        "reasons": reasons,
        "soh": round(float(soh), 5) if isinstance(soh, (int, float)) and math.isfinite(float(soh)) else None,
        "coulombic_efficiency": round(float(ce) / 100.0 if isinstance(ce, (int, float)) and float(ce) > 1.2 else float(ce), 6) if isinstance(ce, (int, float)) and math.isfinite(float(ce)) else None,
        "max_temperature_C": round(max(temps), 3) if temps else None,
        "voltage_window_V": [round(min(voltages), 4), round(max(voltages), 4)] if voltages else None,
        "mean_abs_current_A": round(float(np.mean(np.abs(np.array(currents, dtype=float)))), 5) if currents else None,
        "policy": {
            "stop_soh": req.stop_soh,
            "warn_soh": req.warn_soh,
            "max_temperature_C": req.max_temperature_C,
            "voltage_V": [req.min_voltage_V, req.max_voltage_V],
        },
    }


def _session_export_payload(session_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if "batch" in result:
        return {
            "format": "kineticsforge_batch_v1",
            "session_id": session_id,
            "created_for": "batch_upload_qc",
            "batch": result.get("batch") or {},
        }
    pred = result.get("predictions") or {}
    return {
        "format": "kineticsforge_canonical_v1",
        "session_id": session_id,
        "filename": result.get("filename"),
        "rows": {
            "read": result.get("rows_read"),
            "available": result.get("rows_available"),
            "truncated": result.get("truncated"),
        },
        "schema": result.get("schema") or {},
        "features": result.get("features") or {},
        "feature_vector": result.get("feature_vector") or [],
        "feature_mask": result.get("feature_mask") or [],
        "cycle_summary": result.get("cycle_summary") or [],
        "dqdv": result.get("dqdv") or {},
        "predictions": pred,
        "checkpoint_inference": result.get("checkpoint_inference") or {},
        "warnings": result.get("warnings") or [],
        "privacy": "Raw upload rows are not stored in this session export.",
    }


def _prune_sessions() -> None:
    now = time.time()
    expired = [sid for sid, rec in _byod_sessions.items() if now - rec.get("created_at", now) > SESSION_TTL_SECONDS]
    for sid in expired:
        _byod_sessions.pop(sid, None)


class DegradationRequest(BaseModel):
    temperature_C: float = 45.0
    c_rate: float = 1.0
    cycles: int = 500
    enable_p2o2: bool = True
    enable_jt: bool = True
    enable_sei: bool = True
    enable_neural: bool = True
    sei_k_scale: float = 1.0
    sei_ea_ev: float = 0.56
    p2_rate: float = 0.0028
    p2_soc_crit: float = 0.78
    jt_scale: float = 1.0
    bv_scale: float = 1.0
    stress_exponent: float = 0.55
    residual_scale: float = 1.0
    na: float = 1.02
    mn: float = 0.52
    fe: float = 0.43
    dopant_frac: float = 0.05


class BMSRequest(BaseModel):
    n_cells: int = 8
    duration_seconds: int = 120
    inject_fault: bool = True
    enable_eis: bool = True
    asymmetric_alert: bool = True
    cth_j_per_k: float = 95.0
    edge_k: float = 0.18
    cooling_h: float = 0.045
    load_scale: float = 1.0
    rct_gate: float = 0.043
    risk_threshold: float = 0.42
    ambient_C: float = 45.0
    seed: Optional[int] = 42


class RecyclingRequest(BaseModel):
    mass_kg: float = 100.0
    acid_molarity: float = 2.0
    temperature_C: float = 80.0
    monte_carlo: bool = True
    bayesian_update: bool = True
    leach_time_min: float = 120.0
    particle_um: float = 50.0
    acid_order: float = 0.95
    mn_ea_j_mol: float = 27000.0


class ScreenRequest(BaseModel):
    na: float = 1.0
    mn: float = 0.5
    fe: float = 0.5
    al_doped: bool = False
    ti_doped: bool = False
    upper_voltage: float = 4.10
    ehull_slope: float = 20.0
    w_capacity: float = 0.32
    w_stability: float = 0.32
    w_fade: float = 0.22
    w_cost: float = 0.14


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1200)
    section: str = "general"
    state: Optional[Dict[str, Any]] = None


class CathodeBatchRequest(BaseModel):
    n: int = 100
    temperature_K: float = 318.15


class CyclerWebhookRequest(BaseModel):
    station_id: Optional[str] = None
    channel_id: Optional[str] = None
    cell_id: Optional[str] = None
    cycle: Optional[int] = None
    nominal_capacity_ah: Optional[float] = None
    discharge_capacity_ah: Optional[float] = None
    charge_capacity_ah: Optional[float] = None
    soh: Optional[float] = None
    coulombic_efficiency: Optional[float] = None
    voltage_V: Optional[List[float]] = None
    current_A: Optional[List[float]] = None
    temperature_C: Optional[List[float]] = None
    stop_soh: float = 0.80
    warn_soh: float = 0.85
    max_temperature_C: float = 60.0
    min_voltage_V: float = 1.5
    max_voltage_V: float = 4.5


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def na_ion_terms(q: float, soc: float, comp: Dict[str, float], temp_K: float, req: Optional[DegradationRequest] = None) -> Dict[str, float]:
    k_b = 8.617e-5
    sei_scale = clamp(getattr(req, "sei_k_scale", 1.0), 0.0, 10.0)
    sei_ea = clamp(getattr(req, "sei_ea_ev", 0.56), 0.30, 0.95)
    p2_base = clamp(getattr(req, "p2_rate", 2.8e-3), 0.0, 0.025)
    p2_soc_base = clamp(getattr(req, "p2_soc_crit", 0.78), 0.45, 1.05)
    jt_scale = clamp(getattr(req, "jt_scale", 1.0), 0.0, 4.0)
    mn = clamp(comp.get("Mn", 0.52), 0.0, 1.5)
    fe = clamp(comp.get("Fe", 0.43), 0.0, 1.5)
    dop = clamp(comp.get("dopant_frac", 0.05), 0.0, 0.25)
    jt = clamp(
        mn
        * clamp(1.15 - soc, 0.0, 1.0)
        * math.exp(clamp((temp_K - 298.15) * 0.018, -4.0, 4.0))
        * math.exp(-0.45 * fe - 0.70 * dop),
        0.0,
        4.0,
    ) * jt_scale
    soc_crit = clamp(p2_soc_base - 0.09 * mn + 0.06 * fe + 0.18 * dop, 0.55, 0.95)
    p2_gate = sigmoid((soc - soc_crit) / 0.045)
    p2o2_rate = clamp(
        p2_base
        * p2_gate
        * math.exp(clamp((temp_K - 298.15) * 0.024 / 25.0, -3.0, 3.0))
        * (1.0 + 0.35 * jt),
        0.0,
        0.08,
    )
    # Na+ desolvation barrier: 0.4-0.6 eV in carbonate electrolytes (Jian et al., Komaba et al.)
    # 0.18 eV is the SEI migration barrier, not desolvation. Corrected to physical range.
    barrier = 0.50 + 0.025 * mn - 0.014 * fe - 0.050 * dop
    desolv = clamp(
        math.exp(clamp(barrier / (k_b * temp_K + 1e-10), -2.0, 4.0))
        * (1.0 + 0.25 * max(0.0, soc - 0.85)),
        0.2,
        30.0,
    )
    beta = clamp(0.48 - 0.035 * math.log1p(desolv) + 0.025 * clamp(soc - 0.5, -0.5, 0.5), 0.25, 0.75)
    # Unified SEI: Arrhenius factor normalized to reference T so downstream coefficient stays small
    sei_ref = math.exp(-sei_ea / (k_b * SEI_REF_TEMP))
    sei_arrhenius = sei_scale * math.exp(-sei_ea / (k_b * temp_K)) / max(sei_ref, 1e-30)
    return {"jt": jt, "p2o2_rate": p2o2_rate, "desolv": desolv, "beta": beta, "sei_rate": sei_arrhenius, "soc_crit": soc_crit}


def simulate_degradation(req: DegradationRequest) -> Dict:
    temp_K = req.temperature_C + 273.15
    cycles = int(clamp(req.cycles, 50, 3000))
    comp = {
        "Na": clamp(req.na, 0.60, 1.20),
        "Mn": clamp(req.mn, 0.05, 0.95),
        "Fe": clamp(req.fe, 0.05, 0.95),
        "dopant_frac": clamp(req.dopant_frac, 0.0, 0.25),
    }
    q = 1.0
    stress = 0.6 + req.c_rate ** clamp(req.stress_exponent, 0.25, 3.0)
    curve = [q]
    voltage = [3.34]
    mechanisms = {"p2o2": 0.0, "jt": 0.0, "sei_desolv": 0.0, "rate_polarization": 0.0, "residual": 0.0}
    for i in range(1, cycles + 1):
        soh_window = clamp(q, 0.50, 1.0)
        soc_base = 0.78 + 0.04 * min(1.0, req.c_rate / 2.4)
        usable_soc = 0.62 + 0.38 * soh_window
        soc = clamp(0.55 + (soc_base - 0.55) * usable_soc + 0.022 * math.sin(i * 0.17) * usable_soc, 0.55, 0.98)
        terms = na_ion_terms(q, soc, comp, temp_K, req)
        scale = GLOBAL_DEGRADATION_SCALE * stress
        sqrt_increment = math.sqrt(i) - math.sqrt(i - 1)
        sei_loss = q * terms["sei_rate"] * SEI_GROWTH_COEFF * sqrt_increment * scale if req.enable_sei else 0.0
        p2_loss = q * 0.65 * terms["p2o2_rate"] * scale if req.enable_p2o2 else 0.0
        jt_loss = q * JT_LOSS_COEFF * terms["jt"] * scale if req.enable_jt else 0.0
        desolv_loss = q * DESOLV_LOSS_COEFF * math.log1p(terms["desolv"]) * scale
        exchange_proxy = clamp(0.34 + 0.18 * comp["Fe"] - 0.08 * math.log1p(terms["desolv"]) + 0.04 * (1.0 - terms["beta"]), 0.08, 0.90)
        eta = math.asinh(req.c_rate / (2.0 * exchange_proxy))
        rate_stress = 1.0 + 0.20 * max(0.0, req.c_rate - 1.5) ** 2
        rate_loss = q * clamp(req.bv_scale, 0.0, 5.0) * BV_RATE_LOSS_COEFF * eta * eta * rate_stress * scale
        residual_loss = q * RESIDUAL_LOSS_COEFF * clamp(req.residual_scale, 0.0, 5.0) * sigmoid((i / cycles - 0.62) / 0.16) * (0.8 + 0.35 * req.c_rate) if req.enable_neural else 0.0
        q = clamp(q - sei_loss - p2_loss - jt_loss - desolv_loss - rate_loss - residual_loss, 0.25, 1.02)
        mechanisms["p2o2"] += p2_loss
        mechanisms["jt"] += jt_loss
        mechanisms["sei_desolv"] += sei_loss + desolv_loss
        mechanisms["rate_polarization"] += rate_loss
        mechanisms["residual"] += residual_loss
        v_degradation = mechanisms["p2o2"] * 0.15 + mechanisms["jt"] * 0.08 + mechanisms["sei_desolv"] * 0.05 + mechanisms["rate_polarization"] * 0.04
        voltage.append(clamp(3.34 - v_degradation, 2.40, 3.50))
        curve.append(q)
    knee = None
    for i in range(2, len(curve)):
        if curve[i] - 2 * curve[i - 1] + curve[i - 2] < -1.6e-5:
            knee = i
            break
    rul80 = next((i for i, v in enumerate(curve) if v < 0.8), None)
    step = max(1, len(curve) // 160)
    return {
        "capacity_start": 1.0,
        "capacity_end": round(curve[-1], 5),
        "fade_pct": round(1.0 - curve[-1], 5),
        "knee_point": knee,
        "rul_at_80pct": rul80,
        "cycles": cycles,
        "composition": comp,
        "curve_sampled": [round(curve[i], 5) for i in range(0, len(curve), step)],
        "voltage_sampled": [round(voltage[i], 5) for i in range(0, len(voltage), step)],
        "mechanisms": {k: round(v, 5) for k, v in mechanisms.items()},
    }


def build_neighbors(n: int) -> List[List[int]]:
    cols = n if n <= 8 else int(math.ceil(math.sqrt(n * 1.4)))
    rows = int(math.ceil(n / cols))
    pos = [(i % cols, i // cols) for i in range(n)]
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pos[i][0] - pos[j][0])
            dy = abs(pos[i][1] - pos[j][1])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def _hist_back(hist: List[float], window: int) -> float:
    part = hist[max(0, len(hist) - window):]
    return float(np.mean(part)) if part else 0.0


def simulate_bms(req: BMSRequest) -> Dict:
    n = int(clamp(req.n_cells, 4, 32))
    duration = int(clamp(req.duration_seconds, 30, 600))
    steps = int(clamp(duration, 60, 240))
    dt = duration / steps
    ambient = clamp(req.ambient_C, 15.0, 70.0) + 273.15
    seed = req.seed if req.seed is not None else 42
    rng = np.random.default_rng(seed)
    fault_cell = int(rng.integers(0, n)) if req.inject_fault else -1
    neighbors = build_neighbors(n)
    temp = ambient + rng.normal(0, 0.25, n)
    r0 = 0.033 * (1.0 + rng.normal(0, 0.025, n))
    sei = 0.010 + rng.random(n) * 0.002
    risk = np.zeros(n)
    histories = [[] for _ in range(n)]
    alerts = []
    threshold = clamp(req.risk_threshold if req.asymmetric_alert else max(req.risk_threshold, 0.55), 0.05, 0.95)
    failure_time = duration * 0.84
    for s in range(steps + 1):
        t = s * dt
        prev = temp.copy()
        raw = np.zeros(n)
        for i in range(n):
            fault_drive = sigmoid((t - duration * 0.46) / max(3.0, duration * 0.07)) ** 2 if i == fault_cell else 0.0
            arrh = math.exp(-0.28 / 8.617e-5 * (1.0 / prev[i] - 1.0 / ambient))
            sei[i] += dt * (1.0e-6 * arrh + fault_drive * 7.0e-5)
            r_int = r0[i] + 0.18 * sei[i] + fault_drive * 0.020
            q_ohm = clamp(req.load_scale, 0.05, 4.0) * (34.0 * r_int + fault_drive * 14.0)
            coupling = sum(clamp(req.edge_k, 0.0, 1.2) * (prev[j] - prev[i]) for j in neighbors[i])
            dtdt = (q_ohm + coupling - clamp(req.cooling_h, 0.0, 0.5) * (prev[i] - ambient)) / max(10.0, req.cth_j_per_k)
            temp[i] = clamp(prev[i] + dt * dtdt, 290.0, 390.0)
            r_sei = 0.006 + 0.080 * sei[i] + fault_drive * 0.010
            r_ct = 0.028 * math.exp(1800.0 * (1.0 / temp[i] - 1.0 / ambient)) * (1.0 + 3.5 * sei[i] + fault_drive * 1.6)
            temp_score = sigmoid((temp[i] - 333.15) / 4.5)
            slope_score = sigmoid((dtdt * 60.0 - 1.2) / 0.7)
            eis_score = sigmoid((r_ct + r_sei - clamp(req.rct_gate, 0.005, 0.20)) / 0.009) if req.enable_eis else 0.25 * temp_score
            # Use prev temps for neighbor scoring to avoid cell-ordering artifacts
            neigh_score = np.mean([sigmoid((prev[j] - 333.15) / 5.0) for j in neighbors[i]]) if neighbors[i] else 0.0
            raw[i] = clamp(0.34 * temp_score + 0.21 * slope_score + 0.27 * eis_score + 0.18 * neigh_score, 0.0, 1.0)
            histories[i].append(float(raw[i]))
            hist = histories[i]
            lookback = 0.40 * _hist_back(hist, 30) + 0.28 * _hist_back(hist, 60) + 0.20 * _hist_back(hist, 120) + 0.12 * _hist_back(hist, 240)
            risk[i] = clamp(0.78 * risk[i] + 0.22 * lookback, 0.0, 1.0)
        if s % max(4, steps // 16) == 0 or float(np.max(risk)) > threshold:
            max_cell = int(np.argmax(risk))
            if risk[max_cell] > threshold:
                alerts.append({"t": round(t, 1), "cell": max_cell, "risk": round(float(risk[max_cell]), 4), "lead_seconds": max(0, round(failure_time - t, 1))})
    return {
        "cells": n,
        "fault_cell": fault_cell,
        "seed": seed,
        "thermal_equation": "Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)",
        "final_risks": {f"C{i}": round(float(risk[i]), 4) for i in range(n)},
        "final_temperature_C": {f"C{i}": round(float(temp[i] - 273.15), 2) for i in range(n)},
        "alerts": alerts[-30:],
        "max_risk": round(float(np.max(risk)), 4),
        "ambient_C": round(ambient - 273.15, 2),
    }


def score_composition(comp: Dict[str, float], temp_K: float = 318.15, knobs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    knobs = knobs or comp
    upper_voltage = clamp(float(knobs.get("upper_voltage", 4.10)), 3.60, 4.50)
    ehull_slope = max(1.0, float(knobs.get("ehull_slope", 20.0)))
    w_capacity = float(knobs.get("w_capacity", 0.32))
    w_stability = float(knobs.get("w_stability", 0.32))
    w_fade = float(knobs.get("w_fade", 0.22))
    w_cost = float(knobs.get("w_cost", 0.14))
    al = bool(comp.get("al_doped", False))
    ti = bool(comp.get("ti_doped", False))
    fade_mult, life_mult, cap_mult, vol_mult, rate_mult = (0.90, 1.10, 0.99, 0.90, 1.08) if ti else ((0.82, 1.18, 0.97, 0.85, 1.05) if al else (1, 1, 1, 1, 1))
    dop_frac = (0.04 if al else 0.0) + (0.03 if ti else 0.0)
    na, mn, fe = comp["Na"], comp["Mn"], comp["Fe"]
    q0 = (120.0 + 40.0 * mn - 20.0 * fe) * cap_mult * (1.0 - 0.5 * abs(na - 1.0))
    ea = 0.55 + 0.1 * mn - 0.03 * fe
    k_fade = 1e-4 * (1.0 + 0.2 * fe) * math.exp(-ea * 96485.0 / (8.314 * temp_K))
    jt = 1.0 + 0.3 * max(0.0, mn - 0.5)
    ss = 1.0 / (1.0 + math.exp(-8.0 * (0.5 - mn)))
    fe_stab = 0.9 + 0.2 * fe
    voltage_stress = 1.0 + 1.8 * sigmoid((upper_voltage - 4.05) / 0.08)
    fade_500 = clamp(1.0 - math.exp(-(k_fade * jt * fade_mult / fe_stab) * voltage_stress * 500.0 ** 1.15), 0.02, 0.48)
    cycle_life = 400.0 * life_mult / jt * (0.85 if mn > 0.6 else 1.0)
    # Voltage from weighted redox couples: Fe3+/4+ ~3.20V, Mn3+/4+ ~3.75V in P2-type
    # (Yabuuchi & Komaba 2014, Clément et al. 2015)
    fe_w = fe / max(mn + fe, 0.01)
    mn_w = mn / max(mn + fe, 0.01)
    avg_voltage = mn_w * 3.75 + fe_w * 3.20 + 0.15 * (1.0 - na)
    energy_density = q0 * avg_voltage
    e_form = -4.2 - 0.6 * mn - 0.35 * fe - 0.4 * na - (0.048 if al else 0.0) - (0.054 if ti else 0.0)
    ehull = max(0.0, e_form - (-4.0 - 0.3 * mn - 0.2 * fe) + 0.05)
    phase_stab = 1.0 / (1.0 + math.exp(ehull_slope * (ehull - 0.05)))
    thermal_abuse = clamp((250.0 - 30.0 * max(0.0, mn - 0.5) + 15.0 * fe + (8.0 if al else 0.0) + (7.5 if ti else 0.0) - 180.0) / 120.0, 0.0, 1.0)
    oxygen_risk = clamp(0.22 + max(0.0, mn - 0.55) + max(0.0, 1.0 - na) * 0.8 + 0.24 * sigmoid((upper_voltage - 4.15) / 0.07) - (0.06 if al else 0.0) - (0.08 if ti else 0.0), 0.0, 1.0)
    mixing_risk = clamp(0.18 + abs(mn - fe) * 0.35 + max(0.0, 0.98 - na) * 1.2 + (0.03 if ti else -0.02 if al else 0.0), 0.0, 1.0)
    moisture = clamp(0.20 + max(0.0, na - 0.98) * 0.9 + max(0.0, 1.0 - na) * 2.2, 0.0, 1.0)
    jt_index = clamp((mn - 0.48) * 1.8 - (0.18 if ti else 0.0), 0.0, 1.0)
    defect_score = clamp(1.0 - (0.24 * oxygen_risk + 0.22 * mixing_risk + 0.20 * moisture + 0.24 * jt_index), 0.0, 1.0)
    cost_kg = na * 3.1 * 0.23 + mn * 2.4 * 0.55 + fe * 0.45 * 0.56 + dop_frac * (11.0 * 0.479 if ti else 2.7 * 0.27 if al else 0.0) + 2.5
    cost_kwh = cost_kg / max(energy_density / 1000.0, 0.01)
    stability = clamp(0.28 * (1.0 - fade_500) + 0.18 * ss * fe_stab + 0.18 * phase_stab + 0.16 * thermal_abuse + 0.20 * defect_score, 0.0, 1.0)
    # Mn valence from charge balance: Na + Mn*v_Mn + Fe*3 + dop*v_dop = 4 (for AMO2)
    dop_charge = (0.04 * 3.0 if al else 0.0) + (0.03 * 4.0 if ti else 0.0)
    mn_ox = clamp((4.0 - na - fe * 3.0 - dop_charge) / max(mn, 0.01), 3.0, 4.0)
    fe_ox = 3.0
    total_charge = na + mn * mn_ox + fe * fe_ox + dop_charge
    charge_balance_risk = clamp(abs(total_charge - 4.0) / 1.4, 0.0, 1.0)
    score = w_capacity * (q0 / 180.0) + w_stability * stability + w_fade * (1.0 - fade_500) + w_cost * max(0.0, 1.0 - cost_kwh / 200.0) - 0.08 * charge_balance_risk
    return {
        "capacity": round(q0, 3),
        "capacity_500": round(q0 * (1.0 - fade_500), 3),
        "fade_500": round(fade_500, 5),
        "cycle_life": round(cycle_life, 1),
        "voltage": round(avg_voltage, 4),
        "stability": round(stability, 4),
        "jt_index": round(jt_index, 4),
        "energy_density": round(energy_density, 3),
        "cost_usd_kwh": round(cost_kwh, 2),
        "oxygen_risk": round(oxygen_risk, 4),
        "charge_balance_risk": round(charge_balance_risk, 4),
        "score": round(score, 5),
    }


def screen_batch(n: int = 100, temp_K: float = 318.15, knobs: Optional[Dict[str, float]] = None) -> List[Dict]:
    out = []
    for na in np.linspace(0.84, 1.12, 8):
        for mn in np.linspace(0.20, 0.82, 16):
            for dop in ("none", "al", "ti"):
                fe = clamp(1.0 - mn - (0.04 if dop == "al" else 0.03 if dop == "ti" else 0.0), 0.12, 0.82)
                comp = {"Na": float(na), "Mn": float(mn), "Fe": float(fe), "al_doped": dop == "al", "ti_doped": dop == "ti"}
                prop = score_composition(comp, temp_K=temp_K, knobs=knobs)
                score = prop["score"]
                out.append({"composition": comp, "properties": prop, "score": round(score, 5)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[: max(1, min(n, len(out)))]


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def shrinking_core(k: float, t_min: float) -> float:
    # Reaction-controlled: X = 1 - (1 - kt)^3. Clamp final conversion, not intermediate.
    kt = k * t_min
    if kt >= 1.0:
        return 0.995
    return clamp(1.0 - (1.0 - kt) ** 3, 0.0, 0.995)


def recycling_result(req: RecyclingRequest) -> Dict:
    acid_order = clamp(req.acid_order, 0.1, 2.5)
    particle_um = clamp(req.particle_um, 5.0, 500.0)
    elements = [
        {"n": "Mn", "wt": 0.22, "k0": 0.0038, "Ea": clamp(req.mn_ea_j_mol, 12000.0, 62000.0), "order": acid_order, "particle": particle_um, "prior": (8.8, 1.2)},
        {"n": "Fe", "wt": 0.11, "k0": 0.0029, "Ea": 30000.0, "order": max(0.1, acid_order - 0.10), "particle": particle_um * 1.10, "prior": (7.2, 2.8)},
        {"n": "Na", "wt": 0.05, "k0": 0.0062, "Ea": 19000.0, "order": max(0.1, acid_order - 0.40), "particle": particle_um * 0.90, "prior": (6.5, 3.5)},
        {"n": "Al", "wt": 0.04, "k0": 0.0011, "Ea": 36000.0, "order": acid_order + 0.10, "particle": particle_um * 1.30, "prior": None},
        {"n": "Cu", "wt": 0.015, "k0": 0.0007, "Ea": 34000.0, "order": max(0.1, acid_order - 0.15), "particle": particle_um * 1.40, "prior": None},
    ]
    temp_K = req.temperature_C + 273.15
    t_min = clamp(req.leach_time_min, 5.0, 360.0)
    recoveries = {}
    total = 0.0
    for el in elements:
        temp_factor = math.exp(-el["Ea"] / 8.314 * (1.0 / temp_K - 1.0 / 353.15))
        k = el["k0"] * req.acid_molarity ** el["order"] * temp_factor * (50.0 / el["particle"]) ** 0.35
        x = shrinking_core(k, t_min)
        if req.bayesian_update and el["prior"]:
            x = clamp(0.75 * x + 0.25 * beta_mean(*el["prior"]), 0.0, 0.995)
        mass = req.mass_kg * el["wt"] * x
        total += mass
        recoveries[el["n"]] = {"recovery_rate": round(x, 5), "mass_kg": round(mass, 3)}
    interval = None
    if req.monte_carlo:
        rng = np.random.default_rng(42)
        samples = []
        rates = np.array([recoveries[el["n"]]["recovery_rate"] for el in elements])
        wts = np.array([el["wt"] for el in elements])
        for _ in range(RECYCLING_MC_SAMPLES):
            feed_noise = np.clip(rng.normal(1.0, 0.08, len(elements)), 0.75, 1.25)
            assay_noise = np.clip(rng.normal(1.0, 0.025, len(elements)), 0.92, 1.08)
            samples.append(float(np.sum(req.mass_kg * wts * feed_noise * rates * assay_noise)))
        interval = {"p05_kg": round(float(np.percentile(samples, 5)), 3), "p95_kg": round(float(np.percentile(samples, 95)), 3)}
    acid_kg = req.acid_molarity * 0.098 * req.mass_kg * (t_min / 120.0) ** 0.12
    heat_kwh = max(0.0, req.temperature_C - 25.0) * req.mass_kg * 0.00116
    impurity_penalty = clamp(recoveries["Al"]["recovery_rate"] * 0.28 + recoveries["Cu"]["recovery_rate"] * 0.36, 0.0, 0.8)
    purity_proxy = clamp(0.94 - impurity_penalty * 0.18 + (recoveries["Mn"]["recovery_rate"] + recoveries["Fe"]["recovery_rate"] + recoveries["Na"]["recovery_rate"]) * 0.012, 0.70, 0.98)
    cost = acid_kg * 8.5 + heat_kwh * 8.0 + req.mass_kg * 150.0
    margin_proxy = total * 620.0 * purity_proxy - cost
    return {
        "feedstock_kg": req.mass_kg,
        "kinetics": "shrinking-core leaching",
        "recipe": {"time_min": round(t_min, 2), "particle_um": round(particle_um, 2), "acid_order": round(acid_order, 3)},
        "recoveries": recoveries,
        "total_recovered_kg": round(total, 3),
        "uncertainty_interval": interval,
        "product_purity_proxy": round(purity_proxy, 4),
        "margin_proxy_inr": round(margin_proxy, 2),
        "cost_estimate_inr": round(cost, 2),
        "priors": {"Mn": "Beta(8.8,1.2)", "Fe": "Beta(7.2,2.8)", "Na": "Beta(6.5,3.5)"},
    }


@app.get("/health")
def health():
    return {
        "status": "operational",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "production_cpu",
        "checkpoint_inference": "enabled" if _checkpoint_inference_enabled() else "disabled",
        "models": len(MODEL_REGISTRY),
        "sessions": len(_byod_sessions),
    }


@app.get("/api/models")
def api_models():
    manifest = _load_checkpoint_manifest()
    return {
        "models": _model_registry_payload(),
        "total": len(MODEL_REGISTRY),
        "checkpoint_manifest": {
            "path": str(CHECKPOINT_MANIFEST_PATH.relative_to(ROOT)),
            "generated_at": manifest.get("generated_at"),
            "files": len(manifest.get("files", [])),
        },
        "source_zips": _public_source_zips(manifest),
        "note": "The production CPU server runs physics endpoints and attempts trained PyTorch checkpoint inference when torch is installed.",
    }


@app.post("/api/predict/degradation")
def api_degradation(req: DegradationRequest):
    return {"result": simulate_degradation(req), "provenance": {"model": "Na-ion UDE physics mirror", "claim": "simulation-backed"}}


@app.post("/api/simulate/bms")
def api_bms(req: BMSRequest):
    return simulate_bms(req)


@app.post("/api/optimize/recycling")
def api_recycling(req: RecyclingRequest):
    return recycling_result(req)


@app.post("/api/screen/cathode")
def api_screen(req: ScreenRequest):
    comp = {"Na": req.na, "Mn": req.mn, "Fe": req.fe, "al_doped": req.al_doped, "ti_doped": req.ti_doped}
    knobs = {
        "upper_voltage": req.upper_voltage,
        "ehull_slope": req.ehull_slope,
        "w_capacity": req.w_capacity,
        "w_stability": req.w_stability,
        "w_fade": req.w_fade,
        "w_cost": req.w_cost,
    }
    return {"composition": comp, "predicted": score_composition(comp, knobs=knobs), "candidates": screen_batch(24, knobs=knobs)}


async def _analyze_byod_file(file: UploadFile, require_checkpoint: bool = False) -> Dict[str, Any]:
    if analyze_upload is None:
        raise HTTPException(status_code=503, detail="BYOD analyzer is unavailable")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Upload limit is {mb} MB. Use offline/local deployment for larger cycler exports.")
    try:
        result = analyze_upload(file.filename or "upload.csv", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("BYOD analysis failed")
        raise HTTPException(status_code=500, detail=f"Could not analyze upload: {exc}") from exc
    result = _run_checkpoint_inference(result, require_checkpoint=require_checkpoint)
    session_id = _store_byod_result(result)
    return {
        "session_id": session_id,
        "expires_in_seconds": SESSION_TTL_SECONDS,
        "privacy": "Raw uploads are processed in memory and are not written to disk by the server. Analysis sessions auto-expire.",
        **result,
    }


@app.post("/api/byod/analyze")
async def api_byod_analyze(file: UploadFile = File(...)):
    return await _analyze_byod_file(file, require_checkpoint=False)


@app.post("/api/byod/analyze-full")
async def api_byod_analyze_full(file: UploadFile = File(...)):
    return await _analyze_byod_file(file, require_checkpoint=True)


@app.post("/api/byod/compare")
async def api_byod_compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    a = await _analyze_byod_file(file_a, require_checkpoint=False)
    b = await _analyze_byod_file(file_b, require_checkpoint=False)
    return {
        "comparison": _compare_byod_results(a, b),
        "file_a_session_id": a.get("session_id"),
        "file_b_session_id": b.get("session_id"),
    }


@app.post("/api/byod/batch")
async def api_byod_batch(file: UploadFile = File(...)):
    if analyze_upload is None:
        raise HTTPException(status_code=503, detail="BYOD analyzer is unavailable")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded ZIP is empty")
    if len(content) > MAX_BATCH_TOTAL_BYTES:
        mb = MAX_BATCH_TOTAL_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Batch upload limit is {mb} MB")
    try:
        zf = zipfile.ZipFile(io.BytesIO(content), "r")
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Batch upload must be a ZIP file of cycler CSV/TXT/XLSX files") from exc
    allowed = {".csv", ".txt", ".tsv", ".mpt", ".dat", ".xlsx", ".xlsm"}
    items: List[Dict[str, Any]] = []
    total_unpacked = 0
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for info in members:
            suffix = Path(info.filename).suffix.lower()
            if suffix not in allowed:
                continue
            if len(items) >= MAX_BATCH_FILES:
                break
            if info.file_size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{Path(info.filename).name} exceeds per-file upload limit")
            total_unpacked += int(info.file_size)
            if total_unpacked > MAX_BATCH_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Batch unpacked size exceeds limit")
            try:
                result = analyze_upload(Path(info.filename).name, zf.read(info))
                result = _run_checkpoint_inference(result, require_checkpoint=False)
            except Exception as exc:
                logger.warning("batch member failed: %s: %s", info.filename, exc)
                items.append({
                    "session_id": None,
                    "result": {
                        "filename": Path(info.filename).name,
                        "rows_read": 0,
                        "warnings": [f"Could not analyze file: {exc}"],
                        "predictions": {"inference_mode": "failed"},
                    },
                })
                continue
            session_id = _store_byod_result(result)
            items.append({"session_id": session_id, "result": result})
    if not items:
        raise HTTPException(status_code=400, detail="No supported cycler files found in ZIP")
    report = _batch_report(items)
    batch_id = str(uuid.uuid4())
    _byod_sessions[batch_id] = {"created_at": time.time(), "result": {"batch": report}}
    return {
        "batch_session_id": batch_id,
        "files_analyzed": len(items),
        "expires_in_seconds": SESSION_TTL_SECONDS,
        **report,
    }


@app.post("/api/byod/webhook/cycle")
def api_byod_webhook_cycle(req: CyclerWebhookRequest):
    return {
        "status": "accepted",
        "result": _webhook_cycle_decision(req),
        "use": "Cycle-level triage for automated cycler integration. For full diagnostics, post the complete file to /api/byod/analyze.",
    }


@app.get("/api/byod/session/{session_id}")
def api_byod_session(session_id: str):
    _prune_sessions()
    rec = _byod_sessions.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return _session_export_payload(session_id, rec["result"])


@app.get("/api/byod/session/{session_id}/export-json")
def api_byod_export_json(session_id: str):
    _prune_sessions()
    rec = _byod_sessions.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return JSONResponse(_session_export_payload(session_id, rec["result"]))


@app.get("/api/byod/session/{session_id}/export")
def api_byod_export(session_id: str):
    _prune_sessions()
    rec = _byod_sessions.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    result = rec["result"]
    if "batch" in result:
        batch = result.get("batch") or {}
        out = io.StringIO()
        keys = ["kind", "session_id", "filename", "soh", "confidence", "cycle_80_estimate", "fade_fraction_per_cycle", "anomaly_score", "warnings"]
        writer = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in batch.get("summaries", []):
            writer.writerow({
                "kind": "batch_cell",
                "session_id": row.get("session_id"),
                "filename": row.get("filename"),
                "soh": row.get("soh"),
                "confidence": row.get("confidence"),
                "cycle_80_estimate": row.get("cycle_80_estimate"),
                "fade_fraction_per_cycle": row.get("fade_fraction_per_cycle"),
                "anomaly_score": row.get("anomaly_score"),
                "warnings": "; ".join(str(w) for w in row.get("warnings", [])),
            })
        return PlainTextResponse(out.getvalue(), media_type="text/csv")
    rows: List[Dict[str, Any]] = []
    for row in result.get("cycle_summary", []):
        rows.append({
            "kind": "cycle",
            "name": "",
            "cycle": row.get("cycle"),
            "discharge_capacity_ah": row.get("discharge_capacity_ah"),
            "charge_capacity_ah": row.get("charge_capacity_ah"),
            "ce": row.get("ce"),
            "value": "",
            "source": "",
        })
    for name, value in sorted((result.get("features") or {}).items()):
        rows.append({
            "kind": "feature",
            "name": name,
            "cycle": "",
            "discharge_capacity_ah": "",
            "charge_capacity_ah": "",
            "ce": "",
            "value": value,
            "source": "tier1_extractor",
        })
    model_outputs = ((result.get("predictions") or {}).get("model_outputs") or {})
    for model_id, payload in sorted(model_outputs.items()):
        if not isinstance(payload, dict):
            continue
        source = payload.get("source", "")
        for key, value in sorted(payload.items()):
            if key == "source":
                continue
            rows.append({
                "kind": "model_output",
                "name": f"{model_id}.{key}",
                "cycle": "",
                "discharge_capacity_ah": "",
                "charge_capacity_ah": "",
                "ce": "",
                "value": json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value,
                "source": source,
            })
    out = io.StringIO()
    keys = ["kind", "name", "cycle", "discharge_capacity_ah", "charge_capacity_ah", "ce", "value", "source"]
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return PlainTextResponse(out.getvalue(), media_type="text/csv")


@app.post("/predict/lifetime")
def predict_lifetime(req: DegradationRequest):
    return api_degradation(req)


@app.post("/alert/bms")
def alert_bms(req: BMSRequest):
    result = simulate_bms(req)
    return {"result": {"alert_fired": bool(result["alerts"]), "alerts": result["alerts"], "max_risk": result["max_risk"], "fault_cell": result["fault_cell"]}, "provenance": {"model": "thermal graph plus EIS risk mirror"}}


@app.post("/optimize/recycling")
def optimize_recycling(req: RecyclingRequest):
    return {"result": recycling_result(req), "provenance": {"model": "Bayesian shrinking-core recycling mirror"}}


@app.post("/cathode/screen")
def cathode_screen(req: CathodeBatchRequest):
    return {"count": min(req.n, 100), "results": screen_batch(req.n, req.temperature_K)}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if answer_chat is None:
        return {
            "answer": "Assistant module is unavailable, but the physics API is still running.",
            "source": "local_fallback",
            "memory": "off",
            "setup_required": True,
        }
    return answer_chat(req.question, section=req.section, state=req.state)


WEBAPP = ROOT / "webapp"
if WEBAPP.exists():
    @app.get("/")
    async def index():
        return FileResponse(str(WEBAPP / "index.html"))

    app.mount("/", StaticFiles(directory=str(WEBAPP)), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("serve_lite:app", host="0.0.0.0", port=port, log_level="info")
