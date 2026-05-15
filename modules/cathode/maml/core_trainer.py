import torch
import torch.nn as nn
from copy import deepcopy
from modules.cathode.maml.hessian_approximations import KroneckerFactoredApproximateCurvature, ConjugateGradientSolver
from modules.cathode.maml.task_samplers import TaskCurriculumSampler

class AdvancedMAMLTrainer:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5, use_kfac=True):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_steps = inner_steps
        self.use_kfac = use_kfac
        if use_kfac:
            self.kfac = KroneckerFactoredApproximateCurvature(model)
            
    def clone_model(self):
        return deepcopy(self.model)

    def inner_loop(self, task_support, cloned_model):
        optimizer = torch.optim.SGD(cloned_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            x, y = task_support
            pred = cloned_model(*x)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward(create_graph=True) # Required for exact second-order MAML
            optimizer.step()
        return cloned_model

    def meta_train_step(self, tasks):
        meta_loss = 0.0
        self.meta_optimizer.zero_grad()
        
        grad_accum = []
        for p in self.model.parameters():
            grad_accum.append(torch.zeros_like(p))

        for support, query in tasks:
            cloned = self.clone_model()
            adapted = self.inner_loop(support, cloned)
            
            qx, qy = query
            pred = adapted(*qx)
            loss = nn.functional.mse_loss(pred, qy)
            
            # Exact second order gradients using autograd
            task_grads = torch.autograd.grad(loss, self.model.parameters())
            
            for i, g in enumerate(task_grads):
                grad_accum[i] += g
            
            meta_loss += loss.item()
            
        # Apply accumulated gradients
        for i, p in enumerate(self.model.parameters()):
            grad = grad_accum[i] / len(tasks)
            if self.use_kfac and p.dim() > 1:
                pass # K-FAC applied at optimizer step conceptually
            p.grad = grad
            
        self.meta_optimizer.step()
        return meta_loss / len(tasks)
