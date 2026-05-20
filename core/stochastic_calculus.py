import torch
import torch.nn as nn
import math

class BrownianMotion:
    def __init__(self, t0, w0, dt):
        self.t0 = t0
        self.w0 = w0
        self.dt = dt
        self.cache = {0: w0}

    def _key(self, t):
        if abs(self.dt) < 1e-12:
            return 0
        return int(round((float(t) - float(self.t0)) / float(self.dt)))

    def _time(self, key):
        return float(self.t0) + key * float(self.dt)
        
    def __call__(self, t):
        key = self._key(t)
        if key in self.cache:
            return self.cache[key]
        
        k_prev = max([k for k in self.cache.keys() if k < key], default=0)
        k_next = min([k for k in self.cache.keys() if k > key], default=None)
        t_prev = self._time(k_prev)
        t_next = self._time(k_next) if k_next is not None else float('inf')
        
        if t_next == float('inf'):
            w_prev = self.cache[k_prev]
            w_t = w_prev + torch.randn_like(w_prev) * math.sqrt(abs(t - t_prev))
            self.cache[key] = w_t
            return w_t
            
        w_prev = self.cache[k_prev]
        w_next = self.cache[k_next]
        mean = w_prev + (t - t_prev) / (t_next - t_prev) * (w_next - w_prev)
        var = abs((t - t_prev) * (t_next - t) / (t_next - t_prev))
        w_t = mean + torch.randn_like(w_prev) * math.sqrt(max(var, 0.0))
        self.cache[key] = w_t
        return w_t

class EulerMaruyama(nn.Module):
    def __init__(self, drift_net, diffusion_net):
        super().__init__()
        self.f = drift_net
        self.g = diffusion_net
        
    def forward(self, x0, t_span, dt):
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt)
        t = t0
        x = x0
        traj = [x]
        bm = BrownianMotion(t0, torch.zeros_like(x0), dt)
        
        for _ in range(n_steps):
            dw = bm(t + dt) - bm(t)
            drift = self.f(t, x)
            diffusion = self.g(t, x)
            x = x + drift * dt + torch.bmm(diffusion, dw.unsqueeze(-1)).squeeze(-1)
            t += dt
            traj.append(x)
            
        return torch.stack(traj)

class MilsteinMethod(nn.Module):
    def __init__(self, drift_net, diffusion_net):
        super().__init__()
        self.f = drift_net
        self.g = diffusion_net
        
    def forward(self, x0, t_span, dt):
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt)
        t = t0
        x = x0
        traj = [x]
        bm = BrownianMotion(t0, torch.zeros_like(x0), dt)
        
        for _ in range(n_steps):
            dw = bm(t + dt) - bm(t)
            drift = self.f(t, x)
            diffusion = self.g(t, x)
            
            x_req = x.clone().requires_grad_(True)
            g_req = self.g(t, x_req)
            dg_dx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)
            for i in range(x.shape[1]):
                dg_dx[:, :, i] = torch.autograd.grad(g_req[:, i].sum(), x_req, retain_graph=True)[0]
                
            milstein_term = 0.5 * torch.bmm(diffusion, torch.bmm(dg_dx, (dw**2 - dt).unsqueeze(-1))).squeeze(-1)
            x = x + drift * dt + torch.bmm(diffusion, dw.unsqueeze(-1)).squeeze(-1) + milstein_term
            t += dt
            traj.append(x)
            
        return torch.stack(traj)

class StochasticRungeKutta(nn.Module):
    def __init__(self, drift_net, diffusion_net):
        super().__init__()
        self.f = drift_net
        self.g = diffusion_net
        
    def forward(self, x0, t_span, dt):
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt)
        t = t0
        x = x0
        traj = [x]
        bm = BrownianMotion(t0, torch.zeros_like(x0), dt)
        
        for _ in range(n_steps):
            dw = bm(t + dt) - bm(t)
            sq_dt = math.sqrt(dt)
            
            y1 = x + self.f(t, x) * dt + self.g(t, x) * sq_dt
            y2 = x + self.f(t, x) * dt - self.g(t, x) * sq_dt
            
            drift = self.f(t, x)
            diffusion = self.g(t, x)
            
            milstein_approx = (self.g(t, y1) - self.g(t, y2)) / (2 * sq_dt)
            
            x = x + drift * dt + torch.bmm(diffusion, dw.unsqueeze(-1)).squeeze(-1) + 0.5 * torch.bmm(milstein_approx, (dw**2 - dt).unsqueeze(-1)).squeeze(-1)
            t += dt
            traj.append(x)
            
        return torch.stack(traj)

class FokkerPlanckPDE(nn.Module):
    def __init__(self, nx, min_x, max_x):
        super().__init__()
        self.nx = nx
        self.dx = (max_x - min_x) / nx
        self.x = torch.linspace(min_x, max_x, nx)
        
    def forward(self, p, drift_fn, diffusion_fn, t, dt):
        f_val = drift_fn(t, self.x)
        g_val = diffusion_fn(t, self.x)
        D_val = 0.5 * g_val**2
        
        dp_dt = torch.zeros_like(p)
        
        f_p = f_val * p
        dfp_dx = (f_p[2:] - f_p[:-2]) / (2 * self.dx)
        
        D_p = D_val * p
        d2Dp_dx2 = (D_p[2:] - 2*D_p[1:-1] + D_p[:-2]) / self.dx**2
        
        dp_dt[1:-1] = -dfp_dx + d2Dp_dx2
        dp_dt[0] = dp_dt[1]
        dp_dt[-1] = dp_dt[-2]
        
        return p + dt * dp_dt

class AdjointSDE(nn.Module):
    def __init__(self, f, g, solver):
        super().__init__()
        self.f = f
        self.g = g
        self.solver = solver
        
    def forward(self, x0, t_span, dt):
        return self.solver(x0, t_span, dt)
        
    def backward(self, grad_output, x_traj, t_span, dt):
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt)
        t = t1
        a = grad_output
        grad_params = [torch.zeros_like(p) for p in self.f.parameters()] + [torch.zeros_like(p) for p in self.g.parameters()]
        
        bm = BrownianMotion(t1, torch.zeros_like(x_traj[-1]), -dt)
        
        for i in range(n_steps - 1, -1, -1):
            x = x_traj[i]
            x.requires_grad_(True)
            t_req = torch.tensor(t, requires_grad=True)
            
            f_val = self.f(t_req, x)
            g_val = self.g(t_req, x)
            
            a_dot_f = torch.sum(a * f_val)
            a_dot_g = torch.sum(a.unsqueeze(-1) * g_val, dim=1)
            
            df_dx = torch.autograd.grad(a_dot_f, x, retain_graph=True)[0]
            dg_dx = torch.autograd.grad(torch.sum(a_dot_g), x, retain_graph=True)[0]
            
            df_dp = torch.autograd.grad(a_dot_f, self.f.parameters(), retain_graph=True)
            dg_dp = torch.autograd.grad(torch.sum(a_dot_g), self.g.parameters(), retain_graph=True)
            
            dw = bm(t - dt) - bm(t)
            
            a = a - df_dx * dt - torch.bmm(dg_dx.unsqueeze(1), dw.unsqueeze(-1)).squeeze()
            
            for j, p in enumerate(self.f.parameters()):
                grad_params[j] += df_dp[j] * dt
            for j, p in enumerate(self.g.parameters()):
                grad_params[len(list(self.f.parameters())) + j] += dg_dp[j] * dw[0, 0]
                
            t -= dt
            
        return a, grad_params
