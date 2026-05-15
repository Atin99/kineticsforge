import numpy as np

def generate_composition_grid():
    compositions = []
    for Na in np.linspace(0.9, 1.1, 5):
        for Mn_frac in np.linspace(0.2, 0.8, 10):
            Fe_frac = 1.0 - Mn_frac
            for dopant in [None, 'Al', 'Ti', 'Mg']:
                compositions.append({
                    'Na': Na, 'Mn': Mn_frac, 'Fe': Fe_frac,
                    'dopant': dopant, 'dopant_frac': 0.05 if dopant else 0
                })
    return compositions

def farthest_point_sampling(compositions, n_target=100):
    features = []
    for c in compositions:
        dopant_map = {None: 0.0, 'Al': 1.0, 'Ti': 2.0, 'Mg': 3.0}
        features.append([c['Na'], c['Mn'], c['Fe'], dopant_map[c['dopant']], c['dopant_frac']])
    features = np.array(features)
    selected = [0]
    remaining = list(range(1, len(features)))
    for _ in range(n_target - 1):
        if not remaining:
            break
        dists = np.full(len(remaining), np.inf)
        for i, r_idx in enumerate(remaining):
            for s_idx in selected:
                d = np.linalg.norm(features[r_idx] - features[s_idx])
                dists[i] = min(dists[i], d)
        farthest = remaining[np.argmax(dists)]
        selected.append(farthest)
        remaining.remove(farthest)
    return [compositions[i] for i in selected]

def get_100_compositions():
    all_comps = generate_composition_grid()
    return farthest_point_sampling(all_comps, 100)

def initial_capacity_prior(comp):
    q0 = 120 + 40 * comp['Mn'] - 20 * comp['Fe']
    if comp['dopant'] == 'Al':
        q0 += 15
    elif comp['dopant'] == 'Ti':
        q0 += 8
    elif comp['dopant'] == 'Mg':
        q0 += 5
    q0 += np.random.normal(0, 8)
    return max(q0, 60.0)

def cycle_life_prior(comp):
    base = 400
    if comp['dopant'] == 'Al':
        base *= 1.15
    if comp['Mn'] > 0.6:
        base *= 0.85
    return int(base + np.random.normal(0, 50))
