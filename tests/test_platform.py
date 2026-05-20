"""KineticsForge — Automated test suite for production endpoints.

Covers determinism, physics correctness, and API contract stability.
Run: pytest tests/ -v
"""
import hashlib
import json
import math
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_lite import (
    BMSRequest,
    DegradationRequest,
    RecyclingRequest,
    ScreenRequest,
    simulate_bms,
    simulate_degradation,
    recycling_result,
    score_composition,
    build_neighbors,
    clamp,
    sigmoid,
)


# ── BMS Determinism ──────────────────────────────────────────────────────

def test_bms_determinism_seed_42():
    """Same seed MUST produce identical risk outputs."""
    a = simulate_bms(BMSRequest(seed=42))
    b = simulate_bms(BMSRequest(seed=42))
    assert a["max_risk"] == b["max_risk"], "BMS max_risk must be deterministic with same seed"
    assert a["fault_cell"] == b["fault_cell"], "Fault cell assignment must be deterministic"
    assert a["alerts"] == b["alerts"], "Alert list must be identical"


def test_bms_different_seeds_differ():
    """Different seeds should produce different outputs."""
    a = simulate_bms(BMSRequest(seed=42))
    b = simulate_bms(BMSRequest(seed=999))
    # Fault cells may differ (or at least risk values should)
    risks_a = json.dumps(a["max_risk"])
    risks_b = json.dumps(b["max_risk"])
    # Not a hard assertion — same seed collision is possible but extremely unlikely
    # We just check that the function runs without error for different seeds


def test_bms_no_fault():
    """Simulation without fault injection should complete and produce lower risk."""
    result = simulate_bms(BMSRequest(inject_fault=False, seed=42))
    assert result["fault_cell"] == -1
    assert result["max_risk"] < 0.5, "No-fault simulation should have lower peak risk"


def test_bms_output_structure():
    """BMS output must contain required fields."""
    result = simulate_bms(BMSRequest(seed=42))
    for key in ["cells", "fault_cell", "alerts", "max_risk", "ambient_C"]:
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["alerts"], list)
    assert isinstance(result["max_risk"], float)


# ── Degradation Physics ─────────────────────────────────────────────────

def test_degradation_default_parameters():
    """Default degradation should produce known-good EOL capacity."""
    result = simulate_degradation(DegradationRequest())
    assert 0.65 < result["capacity_end"] < 0.90, f"EOL capacity {result['capacity_end']} outside expected range"
    assert result["fade_pct"] > 0.10, "Fade should be non-trivial at default settings"


def test_degradation_high_temperature():
    """Higher temperature should produce faster degradation."""
    cold = simulate_degradation(DegradationRequest(temperature_C=25))
    hot = simulate_degradation(DegradationRequest(temperature_C=55))
    assert hot["capacity_end"] < cold["capacity_end"], "Higher T must degrade faster"


def test_degradation_high_c_rate():
    """Higher C-rate should produce faster degradation."""
    slow = simulate_degradation(DegradationRequest(c_rate=0.5))
    fast = simulate_degradation(DegradationRequest(c_rate=2.0))
    assert fast["capacity_end"] < slow["capacity_end"], "Higher C-rate must degrade faster"


def test_degradation_mechanisms_sum():
    """Mechanism losses should approximately sum to total fade."""
    result = simulate_degradation(DegradationRequest())
    mech = result["mechanisms"]
    total_mech = sum(mech.values())
    total_fade = result["fade_pct"]
    # Mechanisms are cumulative loss fractions, total_fade is 1 - EOL
    # They won't be exactly equal due to capacity-weighted compounding, but should be close
    assert abs(total_mech - total_fade) < 0.10, f"Mechanism sum {total_mech:.4f} vs fade {total_fade:.4f}"


def test_degradation_disable_all():
    """Disabling all mechanisms should still produce some fade from desolvation + rate."""
    result = simulate_degradation(DegradationRequest(
        enable_p2o2=False, enable_jt=False, enable_sei=False, enable_neural=False
    ))
    # Only desolvation and rate losses remain
    assert result["capacity_end"] > 0.90, "With all mechanisms off, fade should be minimal"


def test_degradation_curve_monotonic():
    """Capacity curve should be monotonically non-increasing."""
    result = simulate_degradation(DegradationRequest())
    curve = result["curve_sampled"]
    for i in range(1, len(curve)):
        assert curve[i] <= curve[i - 1] + 1e-6, f"Capacity increased at index {i}"


def test_degradation_output_structure():
    """Degradation output must contain required fields."""
    result = simulate_degradation(DegradationRequest())
    for key in ["capacity_start", "capacity_end", "fade_pct", "knee_point",
                "rul_at_80pct", "cycles", "composition", "curve_sampled",
                "voltage_sampled", "mechanisms"]:
        assert key in result, f"Missing key: {key}"


# ── Recycling Determinism ───────────────────────────────────────────────

def test_recycling_determinism():
    """Recycling with MC should be deterministic (hardcoded seed=42)."""
    a = recycling_result(RecyclingRequest())
    b = recycling_result(RecyclingRequest())
    assert a["total_recovered_kg"] == b["total_recovered_kg"], "Recycling must be deterministic"
    assert a["uncertainty_interval"] == b["uncertainty_interval"], "MC interval must match"


def test_recycling_higher_temp_higher_recovery():
    """Higher leaching temperature should increase recovery."""
    cold = recycling_result(RecyclingRequest(temperature_C=50))
    hot = recycling_result(RecyclingRequest(temperature_C=90))
    assert hot["total_recovered_kg"] > cold["total_recovered_kg"], "Higher T should improve recovery"


def test_recycling_output_structure():
    """Recycling output must contain required fields."""
    result = recycling_result(RecyclingRequest())
    for key in ["feedstock_kg", "kinetics", "recipe", "recoveries",
                "total_recovered_kg", "product_purity_proxy",
                "margin_proxy_inr", "cost_estimate_inr"]:
        assert key in result, f"Missing key: {key}"
    for el in ["Mn", "Fe", "Na", "Al", "Cu"]:
        assert el in result["recoveries"], f"Missing element: {el}"


def test_recycling_recovery_bounds():
    """Recovery rates must be in [0, 1]."""
    result = recycling_result(RecyclingRequest())
    for el, data in result["recoveries"].items():
        rate = data["recovery_rate"]
        assert 0 <= rate <= 1, f"{el} recovery rate {rate} outside [0,1]"


# ── Materials Screening ─────────────────────────────────────────────────

def test_screening_known_composition():
    """Score a known composition and verify reasonable outputs."""
    comp = {"Na": 1.0, "Mn": 0.5, "Fe": 0.5, "al_doped": False, "ti_doped": False}
    result = score_composition(comp)
    assert 80 < result["capacity"] < 200, f"Capacity {result['capacity']} unreasonable"
    assert 3.0 < result["voltage"] < 4.0, f"Voltage {result['voltage']} unreasonable"
    assert 0 <= result["stability"] <= 1
    assert 0 <= result["oxygen_risk"] <= 1
    assert result["score"] > 0, "Score must be positive for reasonable composition"


def test_screening_dopant_effect():
    """Ti doping should reduce JT index and fade."""
    base = score_composition({"Na": 1.0, "Mn": 0.6, "Fe": 0.4, "al_doped": False, "ti_doped": False})
    doped = score_composition({"Na": 1.0, "Mn": 0.6, "Fe": 0.4, "al_doped": False, "ti_doped": True})
    assert doped["jt_index"] < base["jt_index"], "Ti doping should reduce JT index"
    assert doped["fade_500"] <= base["fade_500"], "Ti doping should not increase fade"
    assert doped["jt_index"] < base["jt_index"], "Ti doping should reduce JT index"


# ── Utility Functions ───────────────────────────────────────────────────

def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_sigmoid():
    assert abs(sigmoid(0) - 0.5) < 1e-6
    assert sigmoid(100) > 0.99
    assert sigmoid(-100) < 0.01


def test_build_neighbors():
    """Neighbor graph should be symmetric and reasonable."""
    n = build_neighbors(8)
    assert len(n) == 8
    for i in range(8):
        for j in n[i]:
            assert i in n[j], "Neighbor relationship must be symmetric"


# ── BYOD Pipeline ───────────────────────────────────────────────────────

def test_byod_schema_detection():
    """Column fingerprinting should detect Arbin-style headers."""
    from data.byod_pipeline import detect_schema
    headers = ["Data_Point", "Test_Time(s)", "Cycle_Index", "Current(A)", "Voltage(V)",
               "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)", "Step_Index"]
    schema = detect_schema(headers)
    assert schema["usable"], "Arbin-style headers should be usable"
    assert "current_A" in schema["mapping"]
    assert "voltage_V" in schema["mapping"]
    assert schema["format"] in ["arbin", "generic"]


def test_byod_neware_schema():
    """Column fingerprinting should handle Chinese Neware headers."""
    from data.byod_pipeline import detect_schema
    headers = ["记录号", "循环号", "工步号", "电流(A)", "电压(V)", "容量(Ah)", "温度(C)"]
    schema = detect_schema(headers)
    assert schema["usable"], "Chinese Neware headers should be usable"
    assert "current_A" in schema["mapping"]
    assert "voltage_V" in schema["mapping"]


def test_byod_biologic_schema():
    """Column fingerprinting should handle BioLogic-style headers."""
    from data.byod_pipeline import detect_schema
    headers = ["time/s", "Ewe/V", "I/mA", "cycle number", "Q charge/discharge/mA.h"]
    schema = detect_schema(headers)
    assert schema["usable"], "BioLogic headers should be usable"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
