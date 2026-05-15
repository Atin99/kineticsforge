# KineticsForge

A battery engineering workbench I built for Na-ion cells. The whole idea behind this is pretty simple actually - instead of giving you some final answer about your battery, it helps you figure out what experiment to run next. You plug in your cathode composition, cycling conditions, or pack layout, and it tells you which degradation mechanism is probably winning, which cell in your pack looks sketchy, whether a cathode recipe is even worth synthesizing, and if your recycling batch will actually make money.

It runs in the browser. No GPU needed for the webapp itself, all the physics runs in javascript. The heavier Python modules are there for when you want to do real training or run the full API backend.

## How to run it

```bash
python serve_lite.py
```

Then go to `http://localhost:8000`. Thats it really. If you want the chatbot to use a cloud model, put your key in `.env`:

```
OPENROUTER_API_KEY=your_key_here
```

Without the key the assistant still works, it just uses built-in answers instead of calling an LLM. Honestly for most questions the local answers are more than fine.

You can also just use `launch.bat` on windows if you dont want to type stuff.

## What the four panels do

**Diagnostics** - Set temperature, C-rate, cycles, and composition. It simulates capacity fade cycle by cycle, breaking it down into SEI growth, P2-O2 phase transition, Jahn-Teller distortion, rate stress from overpotential, and a neural residual that catches whatever the explicit physics missed. Theres a mechanism state map that shows you when and why the dominant loss shifts over time. Also has uncertainty bands so you know how much you should trust the numbers.

**BMS Pack Monitoring** - Models your pack as a graph where cells are nodes and thermal connections are edges. Each cell gets a risk score based on temperature, heat-rise slope, impedance drift, and what its neighbors are doing. The whole point is to catch the bad cell before it causes problems, not after the damage is done. You can inject faults and toggle EIS diagnostics to see how the risk model responds.

**Materials Screening** - Sweeps over Na/Mn/Fe composition space and scores each candidate on capacity, structural stability, fade at 500 cycles, cost per kWh, oxygen evolution risk, and charge balance. Generates a Pareto front so you can actually see the tradeoffs instead of optimizing one thing at a time. Its not going to replace DFT obviously but its fast enough to narrow down your candidates before you go to the lab.

**Recycling** - Takes black mass feedstock and estimates metal recovery using shrinking-core leaching kinetics. Runs Monte Carlo over feedstock variation and assay noise to give you a recovery interval instead of a single number that you cant really trust. Then does a cost check to see if the batch is actually worth running. The Bayesian priors update as you feed in real recovery outcomes.

## Folder structure

This is the actual layout of the project right now:

```
kineticsforge/
|-- serve_lite.py               lightweight server, numpy only, this is what render runs
|-- serve.py                    full server with pytorch loaded
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
|   |-- models.py
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
|   |-- acct1_cathode_cell.py
|   |-- acct2_bms_cell.py
|   |-- acct3_recycling_cell.py
|   |-- phase2_acct1_mega.py
|   |-- phase2_acct2_mega.py
|   |-- phase2_acct3_mega.py
|   |-- prepare_real_data.py
|   |-- prepare_real_data_v2.py
|   |-- (data zip files and split folders)
|
|-- data/                       data pipelines
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

Run `python serve_lite.py` and you get these:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | /api/predict/degradation | capacity fade simulation with mechanism breakdown |
| POST | /api/simulate/bms | thermal graph pack simulation |
| POST | /api/optimize/recycling | leaching kinetics + bayesian recovery |
| POST | /api/screen/cathode | composition scoring and pareto candidates |
| POST | /api/chat | chatbot with dynamic model routing |
| GET | /health | is the server alive |

## Deployment

I deploy on Render free tier. `serve_lite.py` only needs numpy, no pytorch, so it fits in the memory limits without any issues. Training happens on Kaggle T4 GPUs, I split the work across three accounts because of the 12 hour runtime limit per session. The kaggle cells and data prep scripts are all in `kaggle_deploy/`.

The checkpoints folder has all the trained weights from the kaggle runs - things like the physics-informed neural ODE, SOH estimator, RUL predictor, knee detection model, recycling model, and a couple of joint models. They're all `.pt` files sitting in `checkpoints/trained/`.

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

Being honest here. The browser simulations work, all four panels run and give numbers that are in the right ballpark for Na-ion cathode behavior. The physics modules are implemented and they compile. The API serves predictions. I have trained models sitting in checkpoints from the kaggle runs.

But the physics coefficients still need proper calibration against published experimental data - right now some of them are literature estimates that havent been fine-tuned. The degradation predictions are reasonable but could be tighter with better coefficient fitting. I have a calibration engine (`data/calibration_engine.py`) and a calibration-against-published module (`core/calibration_against_published.py`) set up for this, its just ongoing work.

I dont claim hard performance numbers until I've actually run proper holdout benchmarks on real data and passed the validation gates I set up in `validation/`. Simulation-backed is not the same as experimentally validated, and I'm not going to pretend otherwise.
