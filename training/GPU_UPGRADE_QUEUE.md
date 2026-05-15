# KineticsForge GPU Upgrade Queue

Use the remaining Kaggle time for small calibration models that make the physics more useful. Do not spend it on a large generic model unless real pack-level labels become available.

## Train Next

| Priority | Job | GPU Budget | Data | Model | Output |
|---|---:|---:|---|---|---|
| 1 | EIS-to-SOH TinyTCN | 4h | NASA PCoE + impedance summary | 1D TCN over Randles features | SOH/risk head using Rct, RSEI, Warburg |
| 2 | UDE residual calibrator | 6h | BatteryLife + UL_PUR cycle summaries | small MLP residual gated after explicit SEI/P2/JT terms | calibrated residual with mechanism attribution |
| 3 | Thermal pack surrogate | 6h | synthetic pack graphs from topology_variants.py | tiny graph temporal surrogate | fast risk estimate for browser/API |
| 4 | qNEHVI expansion | 4h | composition grid + defect chemistry + cost | GP/qNEHVI loop | ranked synthesis queue with uncertainty |

## Do Not Train Yet

- A large foundation model from scratch. The current data does not justify it.
- A pack-level TGN claiming real safety lead time without real pack failure labels.
- A crystal GNN from scratch. Fine-tune CHGNet/M3GNet only after CIF harvest is working.

## Acceptance Gates

- Every trained artifact must report its input schema, target, metric, and failure cases.
- Residual models must never replace explicit physics terms in the demo/API.
- Safety claims stay blocked until real or instrumented pack tests show positive lead time and no missed failures.
