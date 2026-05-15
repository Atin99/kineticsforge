# KineticsForge Reality Status - 2026-05-11

This workspace is `C:\project 5\kineticsforge_v2_work`.

## Verified Today

- Python compilation passes.
- Foundation data validation passes.
- Foundation manifest was regenerated in the current workspace.
- Stale manifest paths to `attempt 1(antigravity gemini)` were removed from active generated manifests.
- Real assembled data passes the current floor:
  - 12,604 real cycle rows.
  - 252,080 real time-series sample rows.
  - 264,735 total real indexed rows.
  - 64 cells.
- Literature extraction was legally expanded through OpenAlex/Crossref metadata:
  - 76 raw measurement candidates.
  - 51 accepted curated rows.
  - 25 rejected rows retained by reason.
- Whole-cell real holdout benchmark exists:
  - Holdout split is by whole cell, not random row.
  - Holdout MAE is 0.02485 normalized-capacity fraction.
  - Holdout RMSE is 0.03403 normalized-capacity fraction.
  - 90 percent interval coverage is 0.977.
- Recycling governance no longer fails on the old `trajectories` key mismatch. The pipeline accepts the real schema, `alpha_trajectories`.
- A full-cell architect module exists at `modules/full_cell/cell_architect.py`.
- A regional climate engine exists at `core/regional_climate.py`.
- BatteryLife v10 expansion dry-run exists for 1.86 GB of legal additional files.

## Still Blocked

- Training-quality claims are not allowed.
- Literature scale is still too small for independent model training. Current accepted rows: 51. Gate target: 500 accepted rows.
- BMS safety claim is blocked. Current smoke baseline:
  - Mean lead steps: -157.78.
  - Missed failures: 3.
  A safety claim needs positive lead time and zero missed failures on the smoke set.
- Recycling literature is still weak. Metal-recovery priors still use fallback defaults because curated accepted recovery rows are not enough.
- No wet-lab validation exists. Use "simulation-backed" or "real-data benchmarked", not "experimentally validated".

## Do Not Do

- Do not train a model on the 51 literature rows.
- Do not call synthetic-only performance a real validation result.
- Do not delete V2 physics modules or data artifacts.
- Do not force BMS gates to pass by changing thresholds without reporting false alerts, missed failures, and lead-time distribution.

## Next Manual Work

Run a long legal literature crawl with downloads enabled:

```powershell
python -m data.literature_scraper --root data --max-papers 800 --max-documents 160 --mailto your_email@example.com
python -m data.curate_literature_data --root data
python -m data.assemble_real_dataset --root data
python -m data.hyper_data_pipeline --profile foundation --root data --seed 20260430
python -m validation.real_holdout_benchmarks --project-root .
python -m training.colab_kaggle.industrial_training_pipeline --project-root . --task all --profile smoke
python -m validation.v2_readiness_report --project-root .
```

Download more legal BatteryLife files if storage allows:

```powershell
python -m data.real_data_catalog --root data --source batterylife_processed_v10 --all-files --max-gb 2
```

Then add parsers for the downloaded BatteryLife subsets before using them in training.
