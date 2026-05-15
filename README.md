# KineticsForge

This is a battery engineering workbench I built for Na-ion cells. The whole point is to help you figure out what experiment to run next, not to give you a final answer. You plug in your cathode composition, cycling conditions, or pack layout, and it tells you which degradation mechanism is probably winning, which cell in your pack looks sketchy, whether a cathode recipe is even worth synthesizing, and if your recycling batch will actually make money.

It runs in the browser. No GPU needed for the webapp, the physics runs in javascript. The heavier Python modules are there for when you want to do real training or run the full API.

## How to run it

```bash
python serve_lite.py
```

Then go to `http://localhost:8000`. Thats it. If you want the chatbot to use a cloud model, put your key in `.env`:

```
OPENROUTER_API_KEY=your_key_here
```

Without it the assistant still works, it just uses built-in answers instead of calling an LLM. Honestly for most questions the local answers are fine.

You can also use `launch.bat` on windows if you dont want to type.

## What the four panels do

**Diagnostics** - You set temperature, C-rate, cycles, and composition. It simulates capacity fade cycle by cycle, breaking it down into SEI growth, P2-O2 phase transition, Jahn-Teller distortion, rate stress from overpotential, and a neural residual that catches whatever the explicit physics missed. The mechanism state map shows you when and why the dominant loss shifts. There's also uncertainty bands so you know how much to trust the numbers.

**BMS Pack Monitoring** - Models your pack as a graph where cells are nodes and thermal connections are edges. Each cell gets a risk score from temperature, heat-rise slope, impedance drift, and neighbor effects. The idea is to catch the bad cell before it causes problems, not after. You can inject faults and toggle EIS diagnostics to see how the risk model responds.

**Materials Screening** - Sweeps over Na/Mn/Fe composition space and scores each candidate on capacity, structural stability, fade at 500 cycles, cost per kWh, oxygen evolution risk, and charge balance. Generates a Pareto front so you can see the tradeoffs. Its not going to replace DFT but its fast enough to narrow down your candidates before you go to the lab.

**Recycling** - Takes black mass feedstock and estimates metal recovery using shrinking-core leaching kinetics. Runs Monte Carlo over feedstock variation and assay noise to give you a recovery interval instead of a single number. Then does a cost check to see if the batch is actually worth running. The Bayesian priors update as you feed in real recovery outcomes.

## Folder structure

```
kineticsforge/
|-- webapp/                     browser UI, all the javascript physics + HTML + CSS
|   |-- app.js                  simulation logic and canvas rendering
|   |-- index.html              layout
|   |-- index.css               styles
|-- api/                        backend
|   |-- server.py               full API with auth and rate limiting
|   |-- chat_assistant.py       the in-app chatbot
|   |-- auth.py                 bearer token stuff
|   |-- rate_limiter.py
|-- core/                       physics engine
|   |-- phase_transition.py     P2 to O2 transition and JT coupling
|   |-- ude_wrapper.py          universal differential equation model
|   |-- neural_ode.py           neural ODE integration
|   |-- composition_embedder.py composition to embedding
|   |-- evidence_registry.py    tracks where claims come from
|   |-- regional_climate.py     temperature profiles for indian cities
|   |-- india_context.py        INR costs and indian ambient conditions
|   |-- feature_store.py        computed features with schema validation
|   |-- model_registry.py       loads models by name not path
|   |-- and more...             sindy, koopman, lyapunov, physics constraints
|-- modules/
|   |-- cathode/                UDE degradation model + bayesian screener
|   |   |-- degradation_ode.py  the actual Na-ion physics + neural residual
|   |   |-- screener.py         pareto screening with GP surrogate
|   |   |-- defect_chemistry.py charge balance, oxygen redox, JT index
|   |-- bms/                    temporal graph network for pack monitoring
|   |   |-- graph_node.py       TGN with asymmetric alert loss
|   |   |-- precursor_detector.py   multi-scale lookback (30/60/120/240 steps)
|   |   |-- eis_feature_extractor.py   randles circuit features
|   |-- recycling/              shrinking-core leaching + bayesian optimization
|   |   |-- leaching_ode.py
|   |   |-- bayesian_optimizer.py
|   |   |-- closed_loop_optimizer.py
|   |-- full_cell/              full cell design with uncertainty propagation
|       |-- cell_architect.py
|-- inference/                  model loading and inference engine
|-- training/                   training scripts, designed for kaggle T4 not local
|-- kaggle_deploy/              kaggle notebook cells, real data zips, preparation scripts
|-- data/                       data pipelines, ~5M real rows from public datasets
|-- validation/                 holdout benchmarks and readiness checks
|-- business/                   pilot contracts, lab budgets, customer discovery
|-- checkpoints/                saved model weights
|-- docs/                       status documents
|-- reports/                    generated technical reports
|-- scripts/                    utility scripts
|-- dashboard/                  streamlit debug tool, not the product UI
|-- serve_lite.py               lightweight server, numpy only, what render runs
|-- serve.py                    full server with pytorch
|-- Dockerfile
|-- Procfile                    render deployment
|-- render.yaml
|-- requirements.txt
|-- requirements-deploy.txt     just the deploy deps, no pytorch
|-- launch.bat                  windows launcher
|-- launch.ps1                  powershell launcher
|-- setup.py
```

## API endpoints

Run `python serve_lite.py` and you get these:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | /api/predict/degradation | capacity fade simulation with mechanism breakdown |
| POST | /api/simulate/bms | thermal graph pack simulation |
| POST | /api/optimize/recycling | leaching kinetics + bayesian recovery |
| POST | /api/screen/cathode | composition scoring and pareto candidates |
| POST | /api/chat | chatbot |
| GET | /health | is the server alive |

## Deployment

I deploy on Render free tier. `serve_lite.py` only needs numpy, no pytorch, so it fits in the memory limits. Training happens on Kaggle T4 GPUs, I split the work across three accounts because of the 12 hour runtime limit. The kaggle cells and data prep are all in `kaggle_deploy/`.

## Data

I collected about 5 million real battery cycling rows from publicly available research datasets. None of this is proprietary data, its all open-access stuff that research labs have published:

- **NASA PCoE** - The Prognostics Center of Excellence battery dataset from NASA Ames. Lithium-ion cells cycled under different conditions with charge/discharge curves and impedance data. This is probably the most commonly used public battery dataset out there.
- **ISU ILCC** - Iowa State University's battery aging dataset. Has capacity fade curves, RPT (reference performance test) data, and interpolated Q curves. Good for long-term degradation trend analysis.
- **SNL** - Sandia National Labs cycling data. Different chemistries and cycling protocols, useful for cross-chemistry benchmarking.
- **XJTU** - Xi'an Jiaotong University dataset. High-quality cycling data with multiple cells and conditions.
- **Michigan (MICH)** - University of Michigan battery data. Includes experimental cycling with various stress conditions.
- **UL-PUR** - UL/Purdue dataset. Smaller but well-documented cycling experiments.

I wrote scripts to download, normalize, and assemble all of these into a unified format (see `data/assemble_real_dataset.py` and `data/normalize_real_data.py`). Each dataset needed its own normalization script because they all use different file formats and column names, which was honestly kind of painful. The assembled dataset covers 555 unique cells across 11 dataset keys.

The raw zip files are too big for git (~56MB each after processing), so I keep them locally and in Kaggle datasets. The processing scripts and catalog are in `data/` if you want to rebuild from scratch.

I also generate synthetic cycling data for training when real data is sparse for specific conditions like very high temperatures or unusual C-rates. The synthetic generation uses the same physics models that the webapp runs, so its at least internally consistent.

## Whats done and whats not

The browser simulations work, all four panels run and give reasonable numbers. The physics modules are implemented and compile. The API serves predictions. But the models havent been trained on the full real dataset yet, thats the kaggle work thats still in progress. So treat everything as simulation-backed, not experimentally validated. I dont claim performance numbers until I've actually run holdout benchmarks on real data and passed the validation gates.
