"""KineticsForge — Automated test suite for new analytics features.

Covers scenario planner and regional climate API logic.
"""
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scenario_planner import run_scenarios, simulate_bms, compute_bms_confidence
from core.regional_climate import RegionalClimateEngine, REGIONS


def test_scenario_planner_bms():
    """Ensure scenario planner runs multiple BMS setups properly."""
    scenarios = [
        {"ambient_C": 30, "duration_seconds": 60, "seed": 10},
        {"ambient_C": 50, "duration_seconds": 60, "seed": 20},
    ]
    results = run_scenarios(scenarios)
    assert len(results) == 2
    for r in results:
        assert "max_risk" in r
        assert "fault_cell" in r
        assert "confidence" in r
        assert "final_temps_C" in r


def test_climate_regions():
    """Ensure all regions in India context are present and return reasonable profiles."""
    engine = RegionalClimateEngine()
    for name in REGIONS:
        profile = engine.profile(name, days=7)
        assert profile.region == name
        assert len(profile.temperature_C) == 7 * 24
        assert len(profile.relative_humidity_percent) == 7 * 24
        assert len(profile.heat_stress_index) == 7 * 24
        assert len(profile.cold_plating_index) == 7 * 24
