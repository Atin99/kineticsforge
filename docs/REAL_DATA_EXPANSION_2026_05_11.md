# Real Data Expansion Log - 2026-05-11

## Scope

This pass stayed inside legal public data sources. No private systems, access-control bypasses, paywall bypasses, credentialed scraping, or dark-web extraction were used.

## Public Sources Added Or Expanded

- BatteryLife Processed v10, Zenodo: https://zenodo.org/records/18646655
  - Normalized shards: CALB, CALCE, HNEI, MICH, MICH_EXP, NA-ion, SNL, UL_PUR, XJTU.
  - Raw source pointer retained in rows: https://zenodo.org/records/14904364
- NASA PCoE Li-ion Battery Aging Dataset:
  - Portal: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
  - Direct archive used: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- ISU-ILCC Battery Aging Dataset, Figshare:
  - Record: https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234
  - Files downloaded: Valid_cells.csv, process_data.py, README_V2.0.pdf, capacity_fade.zip, Q_interpolated.zip, RPT_json.zip.

## Final Assembled Real Corpus

Manifest: `data/real/assembled/real_dataset_manifest.json`

- Total real rows: 5,091,293
- Cycle or RPT summary rows: 185,221
- Time-series / Q-V sample rows: 4,904,065
- NASA impedance rows: 1,956
- Accepted curated literature rows: 51
- Cells/source files: 555
- Dataset keys: CALB, CALCE, HNEI, ISU_ILCC, MICH, MICH_EXP, NA-ion, NASA_PCoE, SNL, UL_PUR, XJTU

Large outputs are partitioned to avoid memory failures:

- Time-series parts: `data/real/assembled/real_timeseries_sample_parts/`
- Master-index parts: `data/real/assembled/real_master_index_parts/`
- Cycle summary: `data/real/assembled/real_cycle_summary.parquet`
- Impedance summary: `data/real/assembled/real_impedance_summary.parquet`

## Code Changes

- `data/normalize_real_data.py`
  - Generalized BatteryLife normalizer across all downloaded v10 archives.
  - Added per-archive shard output.
  - Added shard-only mode for safe large-source normalization.
  - Fixed nonnumeric protocol labels such as `stepcharge`.
  - Switched zip pickle loading from full-member `z.read()` to streamed `pickle.load()`.
- `data/normalize_isu_ilcc_data.py`
  - Added ISU-ILCC capacity fade and C/5 interpolated Q-V normalizer.
  - Preserves RPT-index notes so RPT measurement index is not misrepresented as true raw cycle count.
- `data/normalize_nasa_pcoe_data.py`
  - Added NASA PCoE MATLAB archive normalizer for cycle, time-series, and impedance rows.
- `data/assemble_real_dataset.py`
  - Reads BatteryLife shards, NASA, ISU, and curated literature.
  - Streams large assembled time-series and master-index outputs into partition directories.
- `data/real_data_catalog.py`, `data/dataset_contracts.py`
  - Added direct public-source mappings for NASA PCoE and ISU-ILCC.

## Verification Run

- `python -m compileall -q data validation training core modules api`: pass
- `python -m data.assemble_real_dataset --root data`: pass
- `python -m validation.real_holdout_benchmarks --project-root .`: pass, quality `needs_model_improvement`
- `python -m validation.v2_readiness_report --project-root .`: pass, but training-quality claims still blocked
- `python -m data.validate_hyper_data --profile foundation --root data`: pass

## Remaining Honest Blockers

- Training-quality claims are still not allowed.
  - Literature accepted rows are 51; the gate asks for 500 before independent literature-based training claims.
  - Real holdout baseline is usable but not strong: holdout MAE is about 0.0849 fraction and quality is `needs_model_improvement`.
  - BMS alert gate still fails with negative mean lead steps and 3 missed failures.
- NASA impedance contains raw outliers in Re/Rct. Keep these rows for provenance, but filter or quality-flag them before using impedance for model claims.
- The ISU RPT_json raw archive is downloaded and retained, but only author-derived capacity_fade and Q_interpolated products were normalized in this pass. Full raw RPT trace normalization should be a separate streaming parser job.
