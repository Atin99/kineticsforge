# Kaggle Deploy Instructions

## Folder Contents

```
kaggle_deploy/
  kineticsforge-acct1.zip   -> Upload as Kaggle Dataset "kineticsforge-acct1"
  kineticsforge-acct2.zip   -> Upload as Kaggle Dataset "kineticsforge-acct2"
  kineticsforge-acct3.zip   -> Upload as Kaggle Dataset "kineticsforge-acct3"
  acct1_cathode_cell.py     -> Paste into Account 1 notebook
  acct2_bms_cell.py         -> Paste into Account 2 notebook
  acct3_recycling_cell.py   -> Paste into Account 3 notebook
```

## Per-Account Setup

### Account 1: Cathode UDE (600 epochs)

1. Go to kaggle.com/datasets/new
2. Upload `kineticsforge-acct1.zip`
3. Set dataset slug to `kineticsforge-acct1`
4. Create new notebook, add dataset `kineticsforge-acct1`
5. Enable GPU (T4)
6. Paste entire contents of `acct1_cathode_cell.py` into a single cell
7. Run

### Account 2: BMS TGN (400 epochs)

1. Upload `kineticsforge-acct2.zip` as dataset `kineticsforge-acct2`
2. Create notebook, add dataset, enable GPU
3. Paste `acct2_bms_cell.py` into single cell
4. Run

### Account 3: Recycling ODE (500 epochs)

1. Upload `kineticsforge-acct3.zip` as dataset `kineticsforge-acct3`
2. Create notebook, add dataset, enable GPU
3. Paste `acct3_recycling_cell.py` into single cell
4. Run

## Resuming After 12-Hour Timeout

Each cell saves checkpoints to `/kaggle/working/checkpoints/` every N epochs and on timeout.

To resume:

1. After the session ends, download the checkpoint file from the output:
   - Account 1: `cathode_resume.pt`
   - Account 2: `bms_resume.pt`
   - Account 3: `recycling_resume.pt`

2. Upload the checkpoint file into the same Kaggle dataset (update the dataset version)

3. Re-run the notebook. The cell checks both `/kaggle/working/checkpoints/` and `/kaggle/input/<slug>/` for resume files.

## GPU Budget

- Each account has 30 hrs/week GPU
- Each session maxes at 12 hrs
- Time limit in code is set to 11 hrs (1 hr safety margin)
- Worst case: 3 sessions per account = 33 hrs total across all 3

## Output Files

After training completes, download from `/kaggle/working/checkpoints/`:

| Account | Best Model | Resume File |
|---------|-----------|-------------|
| 1 | `cathode_best.pt` | `cathode_resume.pt` |
| 2 | `bms_best.pt` | `bms_resume.pt` |
| 3 | `recycling_best.pt` | `recycling_resume.pt` |

Copy the best model files into `kineticsforge_v2_work/checkpoints/` with these names:
- `cathode_model.pt`
- `bms_graph_node.pt`
- `recycling_ode.pt`

Then run the readiness report:
```
python -m validation.v2_readiness_report --project-root .
```

## Kaggle Path Notes

- Kaggle runs Linux, all paths use forward slashes
- Dataset input: `/kaggle/input/<dataset-slug>/`
- Working output: `/kaggle/working/`
- The cells handle this automatically
- Dataset slug must match exactly (lowercase, hyphens only)
