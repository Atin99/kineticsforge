import torch
import torch.nn as nn
import math

class VectorField(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU, layer_norm=True):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self, t, x):
        t_vec = torch.ones_like(x[..., :1]) * t
        xt = torch.cat([x, t_vec], dim=-1)
        return self.net(xt)

class HyperNetVectorField(nn.Module):
    def __init__(self, context_dim, state_dim, hidden_dim):
        super().__init__()
        self.hyper = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, state_dim * hidden_dim + hidden_dim * state_dim + hidden_dim + state_dim)
        )
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
    def forward(self, t, x, context):
        weights = self.hyper(context)
        idx = 0
        w1 = weights[..., idx:idx + self.state_dim * self.hidden_dim].view(-1, self.hidden_dim, self.state_dim)
        idx += self.state_dim * self.hidden_dim
        b1 = weights[..., idx:idx + self.hidden_dim].view(-1, self.hidden_dim)
        idx += self.hidden_dim
        w2 = weights[..., idx:idx + self.hidden_dim * self.state_dim].view(-1, self.state_dim, self.hidden_dim)
        idx += self.hidden_dim * self.state_dim
        b2 = weights[..., idx:idx + self.state_dim].view(-1, self.state_dim)
        h = torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1
        h = torch.tanh(h)
        return torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2

class TensorContractionField(nn.Module):
    def __init__(self, order, dims):
        super().__init__()
        self.order = order
        self.dims = dims
        self.tensors = nn.ParameterList([nn.Parameter(torch.randn(dims)) for _ in range(order)])
    def forward(self, t, x):
        res = x
        for i in range(self.order):
            res = torch.tensordot(res, self.tensors[i], dims=([-1], [0]))
        return res

class PIController:
    def __init__(self, atol, rtol, k_p=0.075, k_i=0.175, k_d=0.01, safe_min=0.2, safe_max=10.0, fac=0.9):
        self.atol = atol
        self.rtol = rtol
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.fac = fac
        self.err_prev = 1.0
        self.err_prev_prev = 1.0
    def step(self, err, dt):
        err = max(err, 1e-8)
        alpha = self.fac * (err ** -self.k_p) * (self.err_prev ** self.k_i) * (self.err_prev_prev ** -self.k_d)
        alpha = max(self.safe_min, min(self.safe_max, alpha))
        self.err_prev_prev = self.err_prev
        self.err_prev = err
        return dt * alpha

class BaseSolver:
    def __init__(self, controller):
        self.controller = controller
    def step(self, f, t, y, dt):
        raise NotImplementedError

class ExplicitRungeKuttaGeneral(BaseSolver):
    def __init__(self, tableau, controller):
        super().__init__(controller)
        self.c = tableau['c']
        self.a = tableau['a']
        self.b = tableau['b']
        self.b_err = tableau['b_err']
        self.s = len(self.c)
    def step(self, f, t, y, dt, args=None):
        k = [f(t, y, *args) if args else f(t, y)]
        for i in range(1, self.s):
            t_i = t + self.c[i] * dt
            y_i = y.clone()
            for j in range(i):
                if self.a[i][j] != 0:
                    y_i += dt * self.a[i][j] * k[j]
            k.append(f(t_i, y_i, *args) if args else f(t_i, y_i))
        y_next = y.clone()
        y_err = torch.zeros_like(y)
        for i in range(self.s):
            if self.b[i] != 0:
                y_next += dt * self.b[i] * k[i]
            if self.b_err[i] != 0:
                y_err += dt * (self.b[i] - self.b_err[i]) * k[i]
        err_norm = torch.norm(y_err / (self.controller.atol + self.controller.rtol * torch.max(torch.abs(y), torch.abs(y_next))))
        dt_next = self.controller.step(err_norm.item(), dt)
        return y_next, dt_next, k, err_norm.item() < 1.0

class RadauIIA(BaseSolver):
    def __init__(self, order, controller):
        super().__init__(controller)
        self.order = order
        self.max_iters = 50
        if order == 3:
            self.c = [1/3, 1.0]
            self.a = [[5/12, -1/12], [3/4, 1/4]]
            self.b = [3/4, 1/4]
        elif order == 5:
            self.c = [(4 - math.sqrt(6))/10, (4 + math.sqrt(6))/10, 1.0]
            self.a = [
                [(88 - 7*math.sqrt(6))/360, (296 - 169*math.sqrt(6))/1800, (-2 + 3*math.sqrt(6))/225],
                [(296 + 169*math.sqrt(6))/1800, (88 + 7*math.sqrt(6))/360, (-2 - 3*math.sqrt(6))/225],
                [(16 - math.sqrt(6))/36, (16 + math.sqrt(6))/36, 1/9]
            ]
            self.b = [(16 - math.sqrt(6))/36, (16 + math.sqrt(6))/36, 1/9]
        self.s = len(self.c)
    def step(self, f, t, y, dt, args=None):
        z = [torch.zeros_like(y) for _ in range(self.s)]
        for _ in range(self.max_iters):
            z_prev = [zi.clone() for zi in z]
            for i in range(self.s):
                t_i = t + self.c[i] * dt
                y_i = y.clone()
                for j in range(self.s):
                    y_i += self.a[i][j] * z[j]
                f_val = f(t_i, y_i, *args) if args else f(t_i, y_i)
                z[i] = dt * f_val
            diff = sum(torch.norm(z[i] - z_prev[i]) for i in range(self.s))
            if diff < 1e-7:
                break
        y_next = y.clone()
        for i in range(self.s):
            y_next += z[i] * (self.b[i] / sum(self.a[i]))
        return y_next, dt, z, True

class SymplecticEuler(BaseSolver):
    def __init__(self, controller=None):
        super().__init__(controller)
    def step(self, f_q, f_p, t, q, p, dt, args=None):
        p_next = p + dt * (f_p(t, q, p, *args) if args else f_p(t, q, p))
        q_next = q + dt * (f_q(t, q, p_next, *args) if args else f_q(t, q, p_next))
        return q_next, p_next, dt, [], True

class HutchinsonTraceEstimator:
    def __init__(self, num_samples=10, dist='rademacher'):
        self.num_samples = num_samples
        self.dist = dist
    def __call__(self, f, x):
        trace = 0.0
        for _ in range(self.num_samples):
            if self.dist == 'rademacher':
                v = torch.randint(0, 2, x.size(), dtype=x.dtype, device=x.device) * 2 - 1
            else:
                v = torch.randn_like(x)
            v.requires_grad_(True)
            with torch.enable_grad():
                fx = f(x)
                vjp = torch.autograd.grad(fx, x, v, create_graph=True)[0]
            trace += torch.sum(vjp * v)
        return trace / self.num_samples

class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, vector_field, trace_estimator):
        super().__init__()
        self.vector_field = vector_field
        self.trace_estimator = trace_estimator
    def forward(self, t, state):
        z, logp = state[..., :-1], state[..., -1:]
        z = z.requires_grad_(True)
        dz_dt = self.vector_field(t, z)
        dlogp_dt = -self.trace_estimator(lambda x: self.vector_field(t, x), z)
        return torch.cat([dz_dt, dlogp_dt.unsqueeze(-1)], dim=-1)

class SDE_EulerMaruyama:
    def __init__(self, dt):
        self.dt = dt
    def step(self, f_drift, f_diff, t, y, args=None):
        drift = f_drift(t, y, *args) if args else f_drift(t, y)
        diff = f_diff(t, y, *args) if args else f_diff(t, y)
        dw = torch.randn_like(y) * math.sqrt(self.dt)
        if diff.dim() > y.dim():
            dw_term = torch.matmul(diff, dw.unsqueeze(-1)).squeeze(-1)
        else:
            dw_term = diff * dw
        return y + drift * self.dt + dw_term

class FractionalDerivativeCaputo:
    def __init__(self, alpha, memory_length):
        self.alpha = alpha
        self.memory_length = memory_length
        self.memory = []
        self.weights = []
    def update_weights(self, dt):
        self.weights = [((i+1)**(1-self.alpha) - i**(1-self.alpha)) / math.gamma(2-self.alpha) for i in range(self.memory_length)]
    def step(self, f, t, y, dt):
        if len(self.memory) == 0:
            self.update_weights(dt)
        self.memory.append(y.clone())
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
        frac_sum = torch.zeros_like(y)
        for i, val in enumerate(reversed(self.memory)):
            frac_sum += self.weights[i] * (val - (self.memory[-2] if len(self.memory)>1 else val))
        return y + dt**self.alpha * frac_sum + dt * f(t, y)

class MultiVariateTaylorExpansion:
    def __init__(self, order, dim):
        self.order = order
        self.dim = dim
        self.tensors = nn.ParameterList([nn.Parameter(torch.randn([dim] * (i+1))) for i in range(order)])
    def forward(self, x):
        res = torch.zeros_like(x)
        for i in range(self.order):
            term = self.tensors[i]
            for j in range(i):
                term = torch.tensordot(term, x, dims=([-1], [0]))
            res += term / math.factorial(i+1)
        return res

class KoopmanOperatorLinearization(nn.Module):
    def __init__(self, encoder_net, decoder_net, latent_dim, dict_size):
        super().__init__()
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.K = nn.Parameter(torch.randn(latent_dim + dict_size, latent_dim + dict_size))
        self.dictionary = nn.Sequential(nn.Linear(latent_dim, dict_size), nn.Tanh())
    def forward(self, x, steps):
        z = self.encoder(x)
        obs = self.dictionary(z)
        lifted = torch.cat([z, obs], dim=-1)
        trajectories = []
        curr = lifted
        for _ in range(steps):
            curr = torch.matmul(curr, self.K)
            trajectories.append(self.decoder(curr[..., :z.shape[-1]]))
        return torch.stack(trajectories, dim=1)

class RiemannianManifoldIntegrator:
    def __init__(self, metric_tensor_fn):
        self.g = metric_tensor_fn
    def christoffel_symbols(self, x):
        x_req = x.clone().requires_grad_(True)
        g_val = self.g(x_req)
        g_inv = torch.inverse(g_val)
        gamma = torch.zeros(x.shape[0], x.shape[0], x.shape[0])
        for k in range(x.shape[0]):
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    dg_ij_xk = torch.autograd.grad(g_val[i,j], x_req, retain_graph=True)[0][k]
                    dg_ik_xj = torch.autograd.grad(g_val[i,k], x_req, retain_graph=True)[0][j]
                    dg_jk_xi = torch.autograd.grad(g_val[j,k], x_req, retain_graph=True)[0][i]
                    gamma[k,i,j] = 0.5 * sum(g_inv[k,m] * (dg_ik_xj + dg_jk_xi - dg_ij_xk) for m in range(x.shape[0]))
        return gamma
    def geodesic_step(self, x, v, dt):
        gamma = self.christoffel_symbols(x)
        dv = torch.zeros_like(v)
        for k in range(x.shape[0]):
            dv[k] = -sum(gamma[k,i,j] * v[i] * v[j] for i in range(x.shape[0]) for j in range(x.shape[0]))
        x_next = x + dt * v
        v_next = v + dt * dv
        return x_next, v_next

class HamiltonianNeuralNetwork(nn.Module):
    def __init__(self, hamiltonian_net):
        super().__init__()
        self.H = hamiltonian_net
    def forward(self, t, state):
        q, p = torch.chunk(state, 2, dim=-1)
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        h_val = self.H(torch.cat([q, p], dim=-1))
        dq_dt = torch.autograd.grad(h_val, p, create_graph=True, grad_outputs=torch.ones_like(h_val))[0]
        dp_dt = -torch.autograd.grad(h_val, q, create_graph=True, grad_outputs=torch.ones_like(h_val))[0]
        return torch.cat([dq_dt, dp_dt], dim=-1)

class LagrangianNeuralNetwork(nn.Module):
    def __init__(self, lagrangian_net):
        super().__init__()
        self.L = lagrangian_net
    def forward(self, t, state):
        q, q_dot = torch.chunk(state, 2, dim=-1)
        q = q.requires_grad_(True)
        q_dot = q_dot.requires_grad_(True)
        l_val = self.L(torch.cat([q, q_dot], dim=-1))
        dl_dqdot = torch.autograd.grad(l_val, q_dot, create_graph=True, grad_outputs=torch.ones_like(l_val))[0]
        hessian = torch.zeros(q_dot.shape[0], q_dot.shape[0])
        for i in range(q_dot.shape[0]):
            hessian[i] = torch.autograd.grad(dl_dqdot[i], q_dot, retain_graph=True)[0]
        dl_dq = torch.autograd.grad(l_val, q, retain_graph=True, grad_outputs=torch.ones_like(l_val))[0]
        q_ddot = torch.linalg.solve(hessian, dl_dq)
        return torch.cat([q_dot, q_ddot], dim=-1)

class AdjointODE:
    def __init__(self, vector_field, solver, tol=1e-5):
        self.f = vector_field
        self.solver = solver
        self.tol = tol
    def solve(self, y0, t_span, context=None):
        trajectory = [y0]
        t_seq = [t_span[0]]
        t = t_span[0]
        y = y0
        dt = 1e-3
        while t < t_span[1]:
            dt = min(dt, t_span[1] - t)
            args = (context,) if context is not None else ()
            y_next, dt_next, _, accepted = self.solver.step(self.f, t, y, dt, args)
            if accepted:
                t += dt
                y = y_next
                trajectory.append(y.clone())
                t_seq.append(t)
            dt = dt_next
        return torch.stack(trajectory), torch.tensor(t_seq)
    def adjoint_solve(self, trajectory, t_seq, grad_output, context=None):
        a = grad_output[-1]
        adj_trajectory = [a.clone()]
        args = (context,) if context is not None else ()
        for i in range(len(t_seq)-1, 0, -1):
            t_curr = t_seq[i]
            dt = t_seq[i-1] - t_curr
            y_curr = trajectory[i]
            def adj_f(t, a_state, *f_args):
                with torch.enable_grad():
                    y_var = y_curr.clone().requires_grad_(True)
                    f_val = self.f(t, y_var, *f_args)
                    vjp = torch.autograd.grad(f_val, y_var, a_state)[0]
                return -vjp
            a_next, _, _, _ = self.solver.step(adj_f, t_curr, a, dt, args)
            a = a_next
            adj_trajectory.append(a.clone())
        adj_trajectory.reverse()
        return torch.stack(adj_trajectory)

class GeneralGraphNeuralODE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, message_layers, update_layers):
        super().__init__()
        msg_net = []
        curr = node_dim * 2 + edge_dim
        for h in message_layers:
            msg_net.extend([nn.Linear(curr, h), nn.LeakyReLU()])
            curr = h
        msg_net.append(nn.Linear(curr, hidden_dim))
        self.msg = nn.Sequential(*msg_net)
        upd_net = []
        curr = node_dim + hidden_dim
        for h in update_layers:
            upd_net.extend([nn.Linear(curr, h), nn.LeakyReLU()])
            curr = h
        upd_net.append(nn.Linear(curr, node_dim))
        self.upd = nn.Sequential(*upd_net)
    def forward(self, t, x, edge_index, edge_attr):
        row, col = edge_index
        src = x[row]
        dst = x[col]
        msg_in = torch.cat([src, dst, edge_attr], dim=-1)
        messages = self.msg(msg_in)
        aggr = torch.zeros(x.shape[0], messages.shape[-1], device=x.device)
        aggr.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)
        upd_in = torch.cat([x, aggr], dim=-1)
        return self.upd(upd_in)

class StochasticGraphNeuralODE(GeneralGraphNeuralODE):
    def __init__(self, node_dim, edge_dim, hidden_dim, message_layers, update_layers, noise_dim):
        super().__init__(node_dim, edge_dim, hidden_dim, message_layers, update_layers)
        self.diff_net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_dim * noise_dim)
        )
        self.noise_dim = noise_dim
        self.node_dim = node_dim
    def diffusion(self, t, x):
        return self.diff_net(x).view(-1, self.node_dim, self.noise_dim)

class GeneralizedBoundaryValueProblem:
    def __init__(self, ode_solver, max_shooting_iters=100, tol=1e-5):
        self.solver = ode_solver
        self.max_iters = max_shooting_iters
        self.tol = tol
    def solve(self, f, bc_fn, t_span, y0_guess):
        y0 = y0_guess.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([y0], max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9)
        def closure():
            optimizer.zero_grad()
            traj, _ = self.solver.solve(y0, t_span)
            res = bc_fn(traj[0], traj[-1])
            loss = torch.sum(res**2)
            loss.backward()
            return loss
        for _ in range(self.max_iters):
            loss = optimizer.step(closure)
            if loss < self.tol:
                break
        traj, t_seq = self.solver.solve(y0, t_span)
        return traj, t_seq

class DelayDifferentialEquation:
    def __init__(self, history_fn, max_delay):
        self.history = history_fn
        self.max_delay = max_delay
        self.buffer = []
        self.t_buffer = []
    def interp(self, t):
        if len(self.t_buffer) == 0 or t <= self.t_buffer[0]:
            return self.history(t)
        idx = torch.searchsorted(torch.tensor(self.t_buffer), t)
        if idx == len(self.t_buffer):
            return self.buffer[-1]
        t0, t1 = self.t_buffer[idx-1], self.t_buffer[idx]
        y0, y1 = self.buffer[idx-1], self.buffer[idx]
        return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    def step(self, f, t, y, dt, delays):
        y_delays = [self.interp(t - d) for d in delays]
        k1 = f(t, y, *y_delays)
        k2 = f(t + dt/2, y + dt/2 * k1, *[self.interp(t + dt/2 - d) for d in delays])
        k3 = f(t + dt/2, y + dt/2 * k2, *[self.interp(t + dt/2 - d) for d in delays])
        k4 = f(t + dt, y + dt * k3, *[self.interp(t + dt - d) for d in delays])
        y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.t_buffer.append(t + dt)
        self.buffer.append(y_next.clone())
        if self.t_buffer[-1] - self.t_buffer[0] > self.max_delay:
            self.t_buffer.pop(0)
            self.buffer.pop(0)
        return y_next


from core.ude_wrapper import PhysicsResidualAudit, UniversalDifferentialEquation
