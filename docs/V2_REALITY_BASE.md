# KineticsForge V2 Reality Base

V2 is not judged by how many files exist. It is judged by whether a skeptical materials professor, a battery R&D lead, or an Indian grant panel can ask "how do you know?" and get a concrete answer.

## What Changed

- INR-first economics with `core/india_context.py`.
- India ambient assumptions: 38 C city, 45 C hot operating point, 50 C abuse reference.
- Claim evidence registry with explicit support and contradiction scores.
- Physics audit suite for bounds, monotonicity, alert timing, and recovery feasibility.
- Dimensional/unit sanity checks so parameters are not silently mixed.
- Defect chemistry model for charge balance, oxygen-redox risk, transition-metal mixing, moisture sensitivity, and Jahn-Teller risk.
- Inverse-design cathode planner that returns candidates, constraints, next measurements, and kill criteria.
- Synthesis protocol generator with precursor masses, process setpoints, QC gates, and failure responses.
- Digital twin telemetry assimilation for BMS instead of only precomputed visual risk.
- Closed-loop recycling planner connecting black-mass recovery to cathode feedstock feasibility.
- Pilot contract module with data requirements, exclusions, IP terms, milestone payment structure, and ROI case.
- Kaggle input zip builder and one-cell runner.
- Technical report builder and Streamlit readiness page.

## What Is Still Not Proven

- Synthetic data does not prove real electrochemical performance.
- Surrogate phase stability is not a replacement for DFT, CALPHAD, or XRD validation.
- Defect chemistry heuristics are triage tools, not atomistic calculations.
- BMS risk simulation is not AIS certification.
- Recycling recovery simulation is not wet-lab recovery.
- INR economics are normalized assumptions and must be replaced with supplier quotes during a real pilot.

## How To Use V2 Without Overclaiming

Run:

```bash
python -m validation.v2_readiness_report --project-root .
python -m reports.technical_report_builder --input data/cache/v2_readiness_report.json
python -m training.colab_kaggle.industrial_training_pipeline --project-root . --task all --profile smoke
python scripts/build_kaggle_input_zip.py
```

For a pitch:

- Say "simulation-backed shortlist", not "validated cathode".
- Say "INR-normalized cost proxy", not "quoted manufacturing cost".
- Say "physics-audited risk trajectory", not "certified safety system".
- Say "lab-ready protocol", not "guaranteed synthesis".
