# KineticsForge

A battery engineering workbench I built for Na-ion cells. The whole idea behind this is pretty simple actually - instead of giving you some final answer about your battery, it helps you figure out what experiment to run next. You plug in your cathode composition, cycling conditions, or pack layout, and it tells you which degradation mechanism is probably winning, which cell in your pack looks sketchy, whether a cathode recipe is even worth synthesizing, and if your recycling batch will actually make money.

It runs in the browser. No GPU needed for the webapp itself, all the physics runs in javascript. The heavier Python modules are there for when you want to do real training or run the full API backend.

## How to run it

```bash
python serve.py
```

Then go to `http://localhost:8000`. Thats it really. If you want the chatbot to use a cloud model, put your key in `.env`:

```
OPENROUTER_API_KEY=your_key_here
```

Without the key the assistant still works, it just uses built-in answers instead of calling an LLM. Honestly for most questions the local answers are more than fine.

You can also just use `launch.bat` on windows if you dont want to type stuff.

## What the panels do

**Diagnostics** - Set temperature, C-rate, cycles, and composition. It simulates capacity fade cycle by cycle, breaking it down into SEI growth, P2-O2 phase transition, Jahn-Teller distortion, rate stress from overpotential, and a bounded residual that catches whatever the explicit physics missed. There's a mechanism attribution donut chart that shows exactly what percentage each mechanism is responsible for. The confidence bar tells you how much of the fade is explained by real physics vs the residual term — if the residual dominates, the confidence drops and the UI tells you to calibrate against experimental data. Uncertainty bands show you the range of plausible outcomes. You can fit the model against your own experimental capacity data using the calibration tool.

**Upload (BYOD)** - Bring a cycler CSV/TXT/XLSX file from any major cycler: Arbin, Neware, Maccor, BioLogic, Basytec, or generic formats including Chinese-header Neware files. The pipeline fingerprints messy column names with 200+ header variants, maps them to the canonical battery schema, extracts tier-1 cycling features, and computes a dQ/dV fingerprint with annotated peaks (Fe³⁺/²⁺, Mn³⁺/⁴⁺, Na ordering, O²⁻ redox) plus a d²V/dQ² second derivative overlay for deeper mechanism fingerprinting. Runs M1-M14 product outputs and uses trained checkpoints when the PyTorch runtime is available. Missing features are carried as a mask instead of silently zero-filled. Formation efficiency scoring shows SEI quality, lifetime index, and robustness gauges when formation data is present.

**BMS Pack Monitoring** - Models your pack as a graph where cells are nodes and thermal connections are edges. Each cell gets a risk score based on temperature, heat-rise slope, impedance drift, and what its neighbors are doing. The simulation is fully deterministic — same seed gives the same result every time. You can run a seed sweep across multiple seeds to check if your alert is robust or just a stochastic fluke; the BMS confidence reflects cross-seed stability. You can inject faults and toggle EIS diagnostics to see how the risk model responds.

**Materials Screening** - Sweeps over Na/Mn/Fe composition space and scores each candidate on capacity, structural stability, fade at 500 cycles, cost per kWh, oxygen evolution risk, and charge balance. Generates a Pareto front so you can actually see the tradeoffs instead of optimizing one thing at a time. Its not going to replace DFT obviously but its fast enough to narrow down your candidates before you go to the lab.

**Recycling** - Takes black mass feedstock and estimates metal recovery using shrinking-core leaching kinetics. Runs Monte Carlo over feedstock variation and assay noise to give you a recovery interval instead of a single number that you cant really trust. Then does a cost check to see if the batch is actually worth running. Uses static Bayesian priors on element recovery rates (Beta distributions) that weight the kinetic model output. A feedback loop to update priors from real lab recoveries is on the roadmap but not implemented yet.

**Decision Console** - Aggregates actionable items from every panel into a single queue. Each ticket has a severity gate (critical/high/medium/low), an owner assignment (researcher/engineer/operator), the evidence source, and a suggested next experiment. This is meant to be the "what do I do next" page that a researcher or QC engineer checks first thing in the morning. Exports to Markdown or JSON for lab notebooks and LIMS integration.

## Export and reporting

Every panel has three export options:

- **Download CSV** — tabular data for OriginLab, Excel, or pandas
- **Download JSON** — structured output with full parameters, results, and metadata for MATLAB/Python post-processing
- **📄 Generate Report** — opens a print-ready report window with metrics, charts (captured from canvas), decision text, and provenance footer. Print to PDF from the browser.

## Test suite

```bash
python -m pytest tests/ -v
```

23 tests covering BMS determinism, degradation physics invariants (monotonicity, temperature/C-rate sensitivity, mechanism sums), recycling reproducibility and bounds, materials screening dopant effects, BYOD schema detection for Arbin/Neware/BioLogic formats, and utility functions. No mocks — tests run against the real `serve_lite.py` functions.

## Folder structure

This is the actual layout of the project right now:

```
kineticsforge/
|-- serve_lite.py               production CPU server implementation; legacy filename
|-- serve.py                    recommended run entrypoint
|-- Dockerfile
|-- Procfile                    render deployment config
|-- render.yaml
|-- requirements.txt
|-- requirements-deploy.txt     just the deploy deps, no pytorch
|-- launch.bat                  windows launcher
|-- launch.ps1                  powershell launcher
|-- setup.py
|
|-- webapp/                     the browser UI
|   |-- app.js                  all the simulation logic, canvas rendering, everything
|   |-- index.html              layout and structure
|   |-- index.css               styles
|
|-- tests/                      pytest test suite
|   |-- test_platform.py        23 tests: determinism, physics, schema, bounds
|
|-- api/                        backend stuff
|   |-- server.py               full API with auth and rate limiting
|   |-- chat_assistant.py       the in-app chatbot with dynamic model routing
|   |-- auth.py                 bearer token auth
|   |-- rate_limiter.py
|
|-- core/                       physics engine, this is where the real math lives
|   |-- phase_transition.py     P2 to O2 transition and JT coupling
|   |-- ude_wrapper.py          universal differential equation model
|   |-- neural_ode.py           neural ODE integration
|   |-- neural_ode_extended.py  extended version with more physics
|   |-- composition_embedder.py composition vector to embedding
|   |-- evidence_registry.py    tracks where claims come from so nothing is unverifiable
|   |-- regional_climate.py     temperature profiles for indian cities
|   |-- india_context.py        INR costs, indian ambient conditions
|   |-- feature_store.py        computed features with schema validation
|   |-- model_registry.py       loads models by name not by path
|   |-- calibration_against_published.py   calibrates coefficients vs published data
|   |-- dimensional_analysis.py
|   |-- physics_audit.py
|   |-- physics_constraints.py
|   |-- physics_constraints_extended.py
|   |-- stochastic_calculus.py
|   |-- uncertainty.py
|   |-- koopman.py              koopman operator stuff
|   |-- lyapunov.py             stability analysis
|   |-- sindy_discovery.py      sparse identification of dynamics
|   |-- utils.py
|
|-- modules/
|   |-- cathode/                UDE degradation model + bayesian screener
|   |   |-- degradation_ode.py  the actual Na-ion degradation physics + neural residual
|   |   |-- screener.py         pareto screening with GP surrogate
|   |   |-- defect_chemistry.py charge balance, oxygen redox, JT index
|   |   |-- composition_sampler.py
|   |   |-- inverse_design.py
|   |   |-- maml_trainer.py     meta-learning for few-shot cathode adaptation
|   |   |-- synthesis_protocol.py
|   |   |-- uncertainty_quantification.py
|   |
|   |-- bms/                    temporal graph network for pack monitoring
|   |   |-- graph_node.py       TGN with asymmetric alert loss
|   |   |-- precursor_detector.py   multi-scale lookback (30/60/120/240 steps)
|   |   |-- eis_feature_extractor.py   randles circuit impedance features
|   |   |-- drive_cycle_sim.py
|   |   |-- sei_kinetics.py
|   |   |-- topology_variants.py
|   |   |-- digital_twin_assimilation.py
|   |
|   |-- recycling/              shrinking-core leaching + bayesian optimization
|   |   |-- leaching_ode.py
|   |   |-- bayesian_optimizer.py
|   |   |-- closed_loop_optimizer.py
|   |   |-- stochastic_blender.py
|   |
|   |-- digital_twin/
|   |   |-- full_cell_model.py
|   |
|   |-- full_cell/              full cell design with uncertainty propagation
|       |-- cell_architect.py
|
|-- inference/                  model loading and inference engine
|   |-- engine.py
|   |-- models.py               M1-M10 architecture definitions
|   |-- models_extended.py      M11-M14 architecture definitions
|
|-- data/                       data pipelines
|   |-- byod_pipeline.py        BYOD upload processing, schema detection, feature extraction
|   |-- assemble_real_dataset.py
|   |-- normalize_real_data.py
|   |-- normalize_nasa_pcoe_data.py
|   |-- normalize_isu_ilcc_data.py
|   |-- real_data_catalog.py
|   |-- real_data_pipeline.py
|   |-- advanced_data_pipeline.py
|   |-- hyper_data_pipeline.py
|   |-- calibration_engine.py
|   |-- crystal_structure_harvester.py
|   |-- crystal_structure_finetuner.py
|   |-- literature_scraper.py
|   |-- curate_literature_data.py
|   |-- synthetic_data_pipeline.py
|   |-- dataset_contracts.py
|   |-- real/                   raw + normalized real battery data
|   |-- synthetic/              generated data for sparse conditions
|   |-- cache/                  precomputed stuff
|
|-- training/                   training scripts, designed for kaggle T4 not local machines
|   |-- train_cathode.py
|   |-- train_bms.py
|   |-- train_recycling.py
|   |-- GPU_UPGRADE_QUEUE.md
|   |-- colab_kaggle/           the whole kaggle/colab training infrastructure
|       |-- industrial_training_pipeline.py
|       |-- kaggle_colab_train.py
|       |-- kaggle_colab_data_bootstrap.py
|       |-- kaggle_input_builder.py
|       |-- configs/
|       |-- notebooks/
|       |-- runs/
|
|-- kaggle_deploy/              kaggle notebook cells, real data zips, prep scripts
|-- kaggle_deploy_2/            second-round training cells
|-- kaggle_deploy_3/            fused mega-cell training (M1-M14 sequential)
|
|-- validation/                 holdout benchmarks and readiness checks
|   |-- holdout_benchmarks.py
|   |-- real_holdout_benchmarks.py
|   |-- v2_readiness_report.py
|
|-- checkpoints/                saved model weights from kaggle training
|   |-- trained/                all the .pt files from training runs
|
|-- business/                   pilot contracts, lab budgets, customer discovery docs
|-- reports/                    generated technical reports
|-- scripts/                    utility scripts
|-- dashboard/                  streamlit debug tool, not the product UI
|-- docs/                       status documents and notes
```

## API endpoints

Run `python serve.py` and you get these:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | /api/predict/degradation | capacity fade simulation with mechanism breakdown |
| POST | /api/simulate/bms | thermal graph pack simulation |
| POST | /api/optimize/recycling | leaching kinetics + bayesian recovery |
| POST | /api/screen/cathode | composition scoring and pareto candidates |
| POST | /api/byod/analyze | upload cycler data, map columns, extract tier-1 features, dQ/dV, M1-M14 rules, and checkpoint outputs when available |
| POST | /api/byod/analyze-full | same upload path, but fails if trained PyTorch checkpoint inference is unavailable |
| POST | /api/byod/compare | compare two uploaded cycler files for A/B cell, protocol, or formation studies |
| POST | /api/byod/batch | upload a ZIP of cycler files and get batch statistics plus outlier flags |
| POST | /api/byod/webhook/cycle | cycler station hook for cycle-level continue/investigate/stop triage |
| GET | /api/byod/session/{session_id} | canonical JSON view of an in-memory BYOD analysis session |
| GET | /api/byod/session/{session_id}/export-json | export canonical BYOD JSON for downstream notebooks, QC systems, or archives |
| GET | /api/byod/session/{session_id}/export | export parsed cycle summaries, extracted features, and M1-M14 readouts from the in-memory session |
| GET | /api/models | M1-M14 model registry with checkpoint presence and zip provenance |
| POST | /api/chat | chatbot with OpenRouter free-model routing, deterministic fallback, and browser-supplied recent-turn context |
| GET | /health | is the server alive |

## Deployment

I deploy on Render free tier with the production CPU server. It runs the physics and BYOD product path without requiring PyTorch, then uses trained checkpoints automatically when the runtime has torch installed. Training happens on Kaggle T4 GPUs, I split the work across three accounts because of the 12 hour runtime limit per session. The kaggle cells and data prep scripts are all in `kaggle_deploy/`, `kaggle_deploy_2/`, and `kaggle_deploy_3/`.

The checkpoints folder has trained weights from the Kaggle runs - M1-M10 in `checkpoints/trained/` and M11-M14 imported from the latest result zips. Run `python scripts/extract_checkpoints.py` after downloading new Kaggle outputs; it writes `checkpoints/trained/checkpoint_manifest.json` so `/api/models` and the UI can show which result zip each checkpoint came from.

Weights are ignored by git on purpose. For local Docker builds the Dockerfile copies `checkpoints/` if the files are present. For Render's git-based Python runtime, create a private artifact with `python scripts/bundle_checkpoints.py`, upload `artifacts/kineticsforge-checkpoints.zip` to private object storage, and set `KF_CHECKPOINT_BUNDLE_URL`. On startup the server restores only checkpoint files and manifest JSON from that bundle.

## Data - how I collected it

This part was honestly kind of painful. I spent a lot of time hunting down publicly available battery cycling datasets from research labs that actually published their data. None of this is proprietary, its all open-access stuff that research groups have put out there.

Here's what I found and assembled:

- **NASA PCoE** - The Prognostics Center of Excellence battery dataset from NASA Ames. Lithium-ion cells cycled under different conditions with charge/discharge curves and impedance measurements. This is probably the most commonly used public battery dataset, pretty much everyone in the field has used it at some point.

- **ISU ILCC** - Iowa State University's battery aging dataset. Has capacity fade curves, RPT (reference performance test) data, and interpolated Q curves. Really good for long-term degradation trend analysis.

- **SNL** - Sandia National Labs cycling data. Different chemistries and cycling protocols. I mainly used this for cross-chemistry benchmarking to make sure the models werent just memorizing one type of cell.

- **XJTU** - Xi'an Jiaotong University dataset. High quality cycling data with multiple cells and conditions. The data formatting was... interesting to deal with.

- **Michigan (MICH)** - University of Michigan battery data. Includes experimental cycling with various stress conditions.

- **UL-PUR** - UL/Purdue dataset. Smaller but well-documented cycling experiments.

I wrote scripts to download, normalize, and assemble all of these into a unified format (see `data/assemble_real_dataset.py` and `data/normalize_real_data.py`). Each dataset needed its own normalization script because they all use completely different file formats, column names, units, and conventions. Some use seconds, some use hours, some have voltage in mV instead of V... you get the idea. The assembled dataset covers 555 unique cells across 11 dataset keys.

I also wrote a literature scraper (`data/literature_scraper.py`) and crystal structure harvester (`data/crystal_structure_harvester.py`) to pull published Na-ion cathode data from papers and materials databases. This feeds into the materials screening module so the composition scores are grounded in actual experimental observations, not just made up numbers.

The raw zip files are too big for git (~56MB each after processing), so I keep them locally and in Kaggle datasets. The processing scripts and catalog are all in `data/` if you want to rebuild everything from scratch.

I also generate synthetic cycling data for training when real data is sparse for specific conditions like very high temperatures or unusual C-rates. The synthetic generation uses the same physics models that the webapp runs, so its at least internally consistent even if it cant replace real experimental data.

## Whats done and whats not

Being honest here. The browser simulations work, all panels run, and the upload path now accepts real cycler exports for first-pass diagnostics. The physics modules are implemented and they compile. The API serves predictions. I have trained models sitting in checkpoints from the kaggle runs, and `/api/models` exposes whether those files are actually present at runtime.

The platform has a full export layer (CSV, JSON, PDF reports), a mechanism attribution system that shows _why_ degradation is happening not just _how much_, dQ/dV fingerprinting with electrochemical peak labeling, formation efficiency scoring, and a decision console that collects actionable items across all panels. Every prediction carries a confidence bar that reflects real signal quality — physics/residual ratio for degradation, cross-seed stability for BMS, feature coverage for uploads — not a fixed number.

But the physics coefficients still need proper calibration against published experimental data - right now some of them are literature estimates that havent been fine-tuned. The degradation predictions are reasonable but could be tighter with better coefficient fitting. I have a calibration engine (`data/calibration_engine.py`) and a calibration-against-published module (`core/calibration_against_published.py`) set up for this, its just ongoing work.

I dont claim hard performance numbers until I've actually run proper holdout benchmarks on real data and passed the validation gates I set up in `validation/`. Simulation-backed is not the same as experimentally validated, and I'm not going to pretend otherwise.
