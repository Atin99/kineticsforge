# KineticsForge V5 — Kaggle Deploy 2

New models (M11-M14) training deployment.

## Structure

```
kaggle_deploy_2/
├── acct1_cell.py                    ← Paste into Kaggle notebook (Account 1)
├── acct2_cell.py                    ← Paste into Kaggle notebook (Account 2)
├── kineticsforge-v5-acct1.zip       ← Upload as Kaggle dataset (Account 1)
├── kineticsforge-v5-acct2.zip       ← Upload as Kaggle dataset (Account 2)
├── acct1_zip/                       ← Source files for acct1 zip
│   ├── config.json
│   └── models_extended.py
└── acct2_zip/                       ← Source files for acct2 zip
    ├── config.json
    └── models_extended.py
```

## How to use

### Account 1 — M11 ElectrolyteHealth + M12 Replenishability
1. Upload `kineticsforge-v5-acct1.zip` as a Kaggle dataset
2. Create new notebook, enable **GPU T4**
3. Paste contents of `acct1_cell.py` into a code cell
4. Run All → checkpoints saved to `/kaggle/working/checkpoints/`
5. Download: `electrolyte_health_best.pt`, `replenishability_best.pt`

### Account 2 — M13 ChemIdentifier + M14 FormationProtocol
1. Upload `kineticsforge-v5-acct2.zip` as a Kaggle dataset
2. Create new notebook, enable **GPU T4**
3. Paste contents of `acct2_cell.py` into a code cell
4. Run All → checkpoints saved to `/kaggle/working/checkpoints/`
5. Download: `chem_identifier_best.pt`, `formation_protocol_best.pt`

## Time estimates
- M11 + M12: ~2-3 hours on T4
- M13 + M14: ~3-4 hours on T4
- Both well within 7-hour limit

## After training
Copy the `*_best.pt` files into `kineticsforge/checkpoints/trained/` and the inference engine will load them automatically.
