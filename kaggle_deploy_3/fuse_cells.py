import os, glob, shutil, zipfile

deploy_dir = r'c:\project 5\kineticsforge\kaggle_deploy_3'
data_dir = os.path.join(deploy_dir, 'acct_zip')
out_zip = os.path.join(deploy_dir, 'kf-v5-30hr-data.zip')
mega_script = os.path.join(deploy_dir, 'mega_cell_all.py')

# 1. Read cells
cell_a = open(os.path.join(deploy_dir, 'cell_a_m1m4.py'), encoding='utf-8').read()
cell_b = open(os.path.join(deploy_dir, 'cell_b_m5m7.py'), encoding='utf-8').read()
cell_c = open(os.path.join(deploy_dir, 'cell_c_m8m10.py'), encoding='utf-8').read()
cell_d = open(os.path.join(deploy_dir, 'cell_d_m11m14.py'), encoding='utf-8').read()

# 2. Extract unique parts
header = cell_a.split('# ── M1: CathodeUDE ──')[0]
m1_m4 = '# ── M1: CathodeUDE ──\n' + cell_a.split('# ── M1: CathodeUDE ──')[1].split('# ── RUN ──')[0]
m5_m7 = '# ── M5: BMS Pack Risk (TGN) ──\n' + cell_b.split('# ── M5: BMS Pack Risk (TGN) ──')[1].split('# ── RUN ──')[0]
m8_m10 = '# ── M8: Joint SOH+RUL+Fade ──\n' + cell_c.split('# ── M8: Joint SOH+RUL+Fade ──')[1].split('# ── RUN ──')[0]
m11_m14 = '# ── M11 ElectrolyteHealth ──\n' + cell_d.split('# ── M11 ElectrolyteHealth ──')[1].split('# ── MAIN RUN ──')[0]

# Add missing functions for M11-M14
missing_funcs = """
def load_cells(split_dir):
    cells = []
    for f in sorted(glob.glob(str(split_dir / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cells.append({"capacity": d["capacity"].astype(np.float32), "features": d["features"].astype(np.float32),
                       "conditions": d["conditions"].astype(np.float32), "cycle_life": float(d.get("cycle_life", -1))})
    return cells

def train_loop(name, model, tdl, vdl, lfn, epochs=200):
    tracker = load_tracker()
    if name in tracker["done"]: log(f"SKIP {name} (done)"); return True
    
    model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); sc=torch.amp.GradScaler("cuda")
    best = tracker["best"] if tracker["current"]==name else 1e9
    s0 = tracker["epoch"] if tracker["current"]==name else 0
    res=Path(CKPT)/f"{name}_resume.pt"
    if not res.exists(): res = DATA/f"{name}_resume.pt"
    if res.exists():
        ck=torch.load(res,map_location=DEVICE,weights_only=False); model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); s0=ck.get("epoch",0); best=ck.get("best",1e9)
        if "sched" in ck: sch.load_state_dict(ck["sched"])
        log(f"  resumed {name} from ep {s0}")
    tracker["current"]=name; save_tracker(tracker)
    total_b = len(tdl)
    for ep in range(s0,epochs):
        if not time_ok():
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sch.state_dict(),"epoch":ep,"best":best}, Path(CKPT)/f"{name}_resume.pt")
            tracker["epoch"]=ep; tracker["best"]=best; save_tracker(tracker); return False
        model.train(); tl=0
        for b in tdl:
            b=[x.to(DEVICE) for x in b]; opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"): l=lfn(model,b)
            if not torch.isfinite(l): continue
            sc.scale(l).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            sc.step(opt); sc.update(); tl+=l.item()
        sch.step(); model.eval(); vl=0
        with torch.no_grad():
            for b in vdl:
                b=[x.to(DEVICE) for x in b]
                with torch.amp.autocast("cuda"): vl+=lfn(model,b).item()
        vl/=max(len(vdl),1)
        if vl<best: best=vl; torch.save({"model":model.state_dict(),"epoch":ep,"best":best},Path(CKPT)/f"{name}_best.pt")
        if ep%5==0: torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sch.state_dict(),"epoch":ep+1,"best":best},Path(CKPT)/f"{name}_resume.pt")
        if ep%10==0: log(f"  [{name}] ep {ep}/{epochs} v={vl:.5f} best={best:.5f} [{time.time()-t0:.0f}s]")
        tracker["epoch"]=ep+1; tracker["best"]=best; save_tracker(tracker)
    torch.save({"model":model.state_dict()},Path(CKPT)/f"{name}_final.pt")
    tracker["done"].append(name); tracker["current"]=None; save_tracker(tracker)
    log(f"  ✓ {name} best={best:.5f}"); return True
"""

# Replace some things in m11_m14 to fit the shared code
# m11-m14 defined its own load_cells, build_eis_dataset, etc. We keep those.
m11_m14_run = cell_d.split('# ── MAIN RUN ──')[1]

# Make sure we import torch.nn.functional
header = header.replace('import torch, torch.nn as nn, numpy as np', 'import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd')
header = header.replace('from pathlib import Path\n', '') + '\nfrom pathlib import Path\n'

run_part = '''
# ── MAIN RUN LOOP ──
BS = 8
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BS,shuffle=True,num_workers=2,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BS,shuffle=False,num_workers=0)
log(f"train={len(train_ds)} val={len(val_ds)} b/ep={len(train_loader)}")

tasks = [
    ("M1_CathodeUDE", lambda: CathodeUDE().to(DEVICE), 800, 3e-4, "cathode_ude"),
    ("M2_SOH", lambda: SOHModel().to(DEVICE), 600, 5e-4, "soh"),
    ("M3_CycleLife", lambda: CycleLifeModel().to(DEVICE), 400, 5e-4, "cycle_life"),
    ("M4_FadeRate", lambda: FadeRateModel().to(DEVICE), 400, 5e-4, "fade_rate"),
    ("M5_BMS_TGN", lambda: PackTGN().to(DEVICE), 500, 3e-4, "bms_tgn"),
    ("M6_RUL", lambda: RULModel().to(DEVICE), 600, 5e-4, "rul"),
    ("M7_Anomaly", lambda: AnomalyAE().to(DEVICE), 400, 5e-4, "anomaly_ae"),
    ("M8_Joint_SOH_RUL", lambda: JointModel().to(DEVICE), 1000, 3e-4, "joint_soh_rul"),
    ("M9_KneeDetect", lambda: KneeDetector().to(DEVICE), 600, 5e-4, "knee_detect"),
    ("M10_ChemRank", lambda: ChemRanker().to(DEVICE), 400, 5e-4, "chem_rank"),
]
for name, mk, epochs, lr, sn in tasks:
    if not time_ok(): log(f"TIME LIMIT before {name}"); break
    log(f"\\n{'='*60}\\nSTARTING {name}\\n{'='*60}")
    model = mk()
    if train_model(name, model, train_loader, val_loader, epochs, lr, sn) == False: break
    del model; gc.collect(); torch.cuda.empty_cache()

# M11-M14 uses different datasets, so we run them differently
DATA = Path(INPUT)
BS = 128
''' + missing_funcs + m11_m14_run

full_code = header + m1_m4 + m5_m7 + m8_m10 + m11_m14 + run_part
with open(mega_script, 'w', encoding='utf-8') as f:
    f.write(full_code)

print('Fused all cells into mega_cell_all.py')

# 3. Zip the data
print(f'Zipping {data_dir} to {out_zip}...')
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, data_dir)
            zf.write(file_path, arcname)
print(f'Created {out_zip} with size {os.path.getsize(out_zip)/1024/1024:.2f} MB')
