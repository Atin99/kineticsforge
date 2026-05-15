import torch
import torch.nn as nn

class HessianVectorProduct:
    def __init__(self, model, damping=0.1):
        self.model = model
        self.damping = damping

    def exact_hvp(self, loss, v):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        gvp = torch.sum(flat_grads * v)
        hvp = torch.autograd.grad(gvp, self.model.parameters(), retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in hvp]) + self.damping * v

    def finite_difference_hvp(self, loss_fn, x, y, v, eps=1e-3):
        # R-op via finite difference approximation to avoid double backprop memory issues
        # W_plus = W + eps * v
        original_params = [p.clone() for p in self.model.parameters()]
        
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.add_(eps * v[idx:idx+numel].view(p.shape))
            idx += numel
            
        loss_plus = loss_fn(self.model(x), y)
        grad_plus = torch.autograd.grad(loss_plus, self.model.parameters())
        flat_grad_plus = torch.cat([g.contiguous().view(-1) for g in grad_plus])
        
        idx = 0
        for i, p in enumerate(self.model.parameters()):
            numel = p.numel()
            p.data.copy_(original_params[i])
            p.data.sub_(eps * v[idx:idx+numel].view(p.shape))
            idx += numel
            
        loss_minus = loss_fn(self.model(x), y)
        grad_minus = torch.autograd.grad(loss_minus, self.model.parameters())
        flat_grad_minus = torch.cat([g.contiguous().view(-1) for g in grad_minus])
        
        # Restore original
        for i, p in enumerate(self.model.parameters()):
            p.data.copy_(original_params[i])
            
        return (flat_grad_plus - flat_grad_minus) / (2 * eps) + self.damping * v

class ConjugateGradientSolver:
    def __init__(self, max_iter=10, tolerance=1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def solve(self, A_fn, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        
        for i in range(self.max_iter):
            Ap = A_fn(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
            x.add_(alpha * p)
            r.sub_(alpha * Ap)
            
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < self.tolerance:
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            
        return x

class NeumannSeriesApproximation:
    def __init__(self, truncate_iter=5, alpha=0.01):
        self.truncate_iter = truncate_iter
        self.alpha = alpha

    def solve(self, hvp_fn, b):
        # Approximates H^-1 * b using Neumann series: (I - alpha*H)^i * b
        p = b.clone()
        res = b.clone()
        for _ in range(self.truncate_iter):
            Hp = hvp_fn(p)
            p = p - self.alpha * Hp
            res.add_(p)
        return self.alpha * res

class KroneckerFactoredApproximateCurvature:
    def __init__(self, model):
        self.model = model
        self.A_factors = {}
        self.G_factors = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle_fwd = module.register_forward_pre_hook(self._save_input(name))
                handle_bwd = module.register_full_backward_hook(self._save_grad_output(name))
                self.handles.extend([handle_fwd, handle_bwd])

    def _save_input(self, name):
        def hook(module, input):
            a = input[0].detach()
            if a.dim() > 2:
                a = a.view(-1, a.size(-1))
            a = torch.cat([a, torch.ones(a.size(0), 1, device=a.device)], dim=1)
            self.A_factors[name] = torch.matmul(a.t(), a) / a.size(0)
        return hook

    def _save_grad_output(self, name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            if g.dim() > 2:
                g = g.view(-1, g.size(-1))
            self.G_factors[name] = torch.matmul(g.t(), g) / g.size(0)
        return hook

    def compute_preconditioned_gradient(self, flat_grad, damping=0.1):
        idx = 0
        preconditioned = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                numel = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                g = flat_grad[idx:idx+numel]
                idx += numel
                
                if name in self.A_factors and name in self.G_factors:
                    A = self.A_factors[name] + damping * torch.eye(self.A_factors[name].size(0), device=A.device)
                    G = self.G_factors[name] + damping * torch.eye(self.G_factors[name].size(0), device=G.device)
                    
                    A_inv = torch.linalg.inv(A)
                    G_inv = torch.linalg.inv(G)
                    
                    W_grad = g.view(module.weight.size(0), -1)
                    precond_W = torch.matmul(G_inv, torch.matmul(W_grad, A_inv))
                    preconditioned.append(precond_W.view(-1))
                else:
                    preconditioned.append(g)
        return torch.cat(preconditioned)
