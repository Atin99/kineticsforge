import torch
import numpy as np
import os
from modules.cathode.screener import screen_compositions
from modules.bms.precursor_detector import run_drive_cycle
from modules.recycling.bayesian_optimizer import optimize_leaching, generate_contour_data, LeachingODESolver

def precompute_cathode_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing cathode screening results for 100 compositions...")
    results = screen_compositions(n=100, T=318)
    rankings = sorted(results, key=lambda r: r['score'], reverse=True)
    top_10 = rankings[:10]
    worst_5 = rankings[-5:]

    fade_curves = {}
    cycles = np.arange(0, 501, 1)
    for i, r in enumerate(rankings):
        q0 = r['Q0']
        fade_rate = r['fade']
        curve = q0 * (1 - fade_rate * (cycles / 500)**1.5)
        curve += np.random.normal(0, 0.03 * curve, len(cycles))
        fade_curves[i] = curve

    np.savez(os.path.join(cache_dir, 'cathode_screening.npz'),
             rankings=rankings,
             top_10_indices=list(range(10)),
             worst_5_indices=list(range(len(rankings)-5, len(rankings))),
             fade_curves=np.array([fade_curves[i] for i in range(min(20, len(rankings)))]),
             cycles=cycles)
    print(f"Cached {len(rankings)} cathode results.")

def precompute_bms_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing BMS drive cycle simulations...")
    for i in range(5):
        inject = i < 3
        result = run_drive_cycle(seed=42 + i, inject_failure=inject, duration_seconds=2880)
        history = result['history']
        alert_time = result['alerts'][0]['step'] if result.get('alerts') else -1
        fail_cell = result['alert_cells'][0] if result.get('alert_cells') else -1
        np.savez(os.path.join(cache_dir, f'bms_drive_{i}.npz'),
                 time=history['time'],
                 risk=history['risk'],
                 R_int=history['R_int'],
                 V_cells=history['V_cells'],
                 T_cells=history['T_cells'],
                 SOC_cells=history['SOC_cells'],
                 L_sei=history['L_sei'],
                 P_dendrite=history['P_dendrite'],
                 I_pack=history['I_pack'],
                 alert_fired=result['alert_fired'],
                 alert_time=alert_time,
                 lead_time_min=result['lead_time_min'],
                 inject_failure=inject,
                 fail_cell=fail_cell,
                 n_alerts=result['n_alerts'])
    drive_paths = sorted([os.path.join(cache_dir, f'bms_drive_{i}.npz') for i in range(5)])
    histories = []
    scenarios = []
    for path in drive_paths:
        d = np.load(path, allow_pickle=True)
        history = {k: d[k] for k in d.files if k in {'time', 'risk', 'R_int', 'V_cells', 'T_cells', 'SOC_cells', 'L_sei', 'P_dendrite', 'I_pack'}}
        scenarios.append({
            'failure_type': 'synthetic_fault' if bool(d['inject_failure']) else 'nominal',
            'alert_fired': bool(d['alert_fired']),
            'lead_time_min': float(d['lead_time_min']),
            'alert_cells': [int(d['fail_cell'])] if int(d['fail_cell']) >= 0 else [],
            'n_alerts': int(d['n_alerts']),
        })
        histories.append(history)
        d.close()
    np.savez(os.path.join(cache_dir, 'bms_simulation.npz'), scenarios=np.array(scenarios, dtype=object), history=np.array(histories, dtype=object))
    print("Cached 5 BMS drive cycle results.")

def precompute_recycling_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing recycling optimization...")
    opt = optimize_leaching(n_trials=200)
    solver = LeachingODESolver()
    T_range, pH_range, recovery_grid = generate_contour_data(solver, grid_resolution=30)

    alpha_traj = np.zeros((3, 180))
    alpha_state = np.zeros(3)
    for step in range(180):
        for s in range(3):
            da = solver.shrinking_core_rate(alpha_state[s], opt['T'], opt['conc'], 50e-6, s)
            alpha_state[s] = min(alpha_state[s] + da, 1.0)
        alpha_traj[:, step] = alpha_state

    np.savez(os.path.join(cache_dir, 'recycling_optimization.npz'),
             pareto_front=np.array([{
                 'T': opt['T'], 'pH': opt['pH'], 'conc': opt['conc'], 't': opt['t'],
                 'recovery': opt.get('recovery', opt.get('score', 0.0)),
                 'cost': opt.get('cost', 0.0),
                 'impurity': opt.get('impurity', 0.0),
                 'alpha_Mn': opt['alpha_Mn'], 'alpha_Fe': opt['alpha_Fe'], 'alpha_Na': opt['alpha_Na'],
             }], dtype=object),
             optimal_T=opt['T'],
             optimal_pH=opt['pH'],
             optimal_conc=opt['conc'],
             optimal_t=opt['t'],
             alpha_Mn=opt['alpha_Mn'],
             alpha_Fe=opt['alpha_Fe'],
             alpha_Na=opt['alpha_Na'],
             T_range=T_range,
             pH_range=pH_range,
             recovery_grid=recovery_grid,
             alpha_trajectory=alpha_traj)
    print("Cached recycling optimization results.")

def precompute_all():
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
    precompute_cathode_cache(cache_dir)
    precompute_bms_cache(cache_dir)
    precompute_recycling_cache(cache_dir)
    print("All pre-computation complete. Dashboard will load instantly.")

if __name__ == '__main__':
    precompute_all()
