import torch
import numpy as np
from collections import defaultdict

class TaskCurriculumSampler:
    def __init__(self, tasks, n_ways, n_shots, difficulty_metric='loss'):
        self.tasks = tasks
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.difficulty_metric = difficulty_metric
        self.task_difficulties = defaultdict(float)
        self.task_selection_counts = defaultdict(int)
        
    def update_difficulties(self, task_indices, losses):
        for idx, loss in zip(task_indices, losses):
            # Exponential moving average of task difficulty
            self.task_difficulties[idx] = 0.9 * self.task_difficulties[idx] + 0.1 * loss
            self.task_selection_counts[idx] += 1

    def sample_tasks(self, batch_size, strategy='hard_mining', temperature=1.0):
        if len(self.task_difficulties) < len(self.tasks):
            # Initial random exploration phase
            return np.random.choice(len(self.tasks), batch_size, replace=False)
            
        difficulties = np.array([self.task_difficulties[i] for i in range(len(self.tasks))])
        counts = np.array([self.task_selection_counts[i] for i in range(len(self.tasks))])
        
        if strategy == 'hard_mining':
            # Softmax over difficulties scaled by temperature
            probs = np.exp(difficulties / temperature)
            probs /= np.sum(probs)
        elif strategy == 'ucb':
            # Upper Confidence Bound strategy for task exploration/exploitation
            total_pulls = np.sum(counts)
            exploration_term = np.sqrt(2 * np.log(total_pulls + 1) / (counts + 1e-5))
            ucb_scores = difficulties + 0.5 * exploration_term
            probs = np.exp(ucb_scores / temperature)
            probs /= np.sum(probs)
        else:
            probs = np.ones(len(self.tasks)) / len(self.tasks)
            
        selected_indices = np.random.choice(len(self.tasks), batch_size, p=probs, replace=False)
        return selected_indices

class PhaseDiagramTaskGenerator:
    def __init__(self, composition_bounds, phase_regions):
        self.bounds = composition_bounds
        self.regions = phase_regions
        
    def sample_within_phase(self, phase_name, n_samples):
        region = self.regions[phase_name]
        samples = []
        while len(samples) < n_samples:
            comp = {el: np.random.uniform(b[0], b[1]) for el, b in self.bounds.items()}
            # Normalize to 1.0
            total = sum(comp.values())
            comp = {el: v/total for el, v in comp.items()}
            
            if self._check_in_region(comp, region):
                samples.append(comp)
        return samples

    def _check_in_region(self, comp, region):
        # Evaluates hyper-plane inequalities for phase diagram boundaries
        for condition in region['conditions']:
            val = sum(comp[el] * condition['weights'].get(el, 0) for el in comp)
            if val > condition['max'] or val < condition['min']:
                return False
        return True

    def generate_episodic_batch(self, batch_size, n_way, n_support, n_query):
        batch = []
        phase_names = list(self.regions.keys())
        
        for _ in range(batch_size):
            selected_phases = np.random.choice(phase_names, n_way, replace=False)
            support_set = []
            query_set = []
            
            for phase in selected_phases:
                samples = self.sample_within_phase(phase, n_support + n_query)
                support_set.extend([(s, phase) for s in samples[:n_support]])
                query_set.extend([(s, phase) for s in samples[n_support:]])
                
            batch.append({
                'support': support_set,
                'query': query_set
            })
        return batch
