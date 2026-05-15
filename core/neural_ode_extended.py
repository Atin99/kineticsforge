import torch
import torch.nn as nn
import math

class PortHamiltonianNeuralODE(nn.Module):
    def __init__(self, state_dim, dissipation=True):
        super().__init__()
        self.state_dim = state_dim
        self.half = state_dim // 2
        self.H_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
        self.dissipation = dissipation
        if dissipation:
            self.R_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Softplus(),
                nn.Linear(64, self.half)
            )
        self.J = torch.zeros(state_dim, state_dim)
        for i in range(self.half):
            self.J[i, self.half + i] = 1.0
            self.J[self.half + i, i] = -1.0

    def forward(self, t, x):
        x_req = x.clone().requires_grad_(True)
        H = self.H_net(x_req)
        dH_dx = torch.autograd.grad(H.sum(), x_req, create_graph=True)[0]
        dx_dt = torch.matmul(dH_dx, self.J.T.to(x.device))
        if self.dissipation:
            R_diag = torch.nn.functional.softplus(self.R_net(x))
            dx_dt[..., self.half:] -= R_diag * dH_dx[..., self.half:]
        return dx_dt

class SymplecticIntegrator:
    def __init__(self, order=4):
        self.order = order
        if order == 2:
            self.coeffs_d = [0.5, 0.5]
            self.coeffs_k = [1.0]
        elif order == 4:
            c1 = 1.0 / (2 * (2 - 2**(1/3)))
            c2 = (1 - 2**(1/3)) / (2 * (2 - 2**(1/3)))
            d1 = 1.0 / (2 - 2**(1/3))
            d2 = -2**(1/3) / (2 - 2**(1/3))
            self.coeffs_d = [c1, c2, c2, c1]
            self.coeffs_k = [d1, d2, d1]

    def step(self, grad_H_q, grad_H_p, q, p, dt):
        for i in range(len(self.coeffs_d)):
            q = q + self.coeffs_d[i] * dt * grad_H_p(p)
            if i < len(self.coeffs_k):
                p = p - self.coeffs_k[i] * dt * grad_H_q(q)
        return q, p

class ProjectionMethod:
    def __init__(self, constraint_fn, max_iter=50, tol=1e-8):
        self.constraint = constraint_fn
        self.max_iter = max_iter
        self.tol = tol

    def project(self, y):
        for _ in range(self.max_iter):
            c = self.constraint(y)
            if torch.norm(c) < self.tol:
                break
            y_req = y.clone().requires_grad_(True)
            c_req = self.constraint(y_req)
            J = torch.autograd.functional.jacobian(self.constraint, y_req)
            JJt = torch.matmul(J, J.T)
            lam = torch.linalg.solve(JJt, c_req)
            y = y - torch.matmul(J.T, lam)
        return y

class LieGroupIntegrator:
    def __init__(self, algebra_dim):
        self.dim = algebra_dim

    def exp_map_so3(self, omega, dt):
        theta = torch.norm(omega) * dt
        if theta < 1e-10:
            return torch.eye(3, device=omega.device)
        K = torch.zeros(3, 3, device=omega.device)
        w = omega * dt / theta
        K[0, 1] = -w[2]; K[0, 2] = w[1]
        K[1, 0] = w[2]; K[1, 2] = -w[0]
        K[2, 0] = -w[1]; K[2, 1] = w[0]
        return torch.eye(3, device=omega.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

    def exp_map_se3(self, twist, dt):
        omega = twist[:3]
        v = twist[3:]
        R = self.exp_map_so3(omega, dt)
        theta = torch.norm(omega) * dt
        if theta < 1e-10:
            t = v * dt
        else:
            K = torch.zeros(3, 3, device=omega.device)
            w = omega / torch.norm(omega)
            K[0, 1] = -w[2]; K[0, 2] = w[1]
            K[1, 0] = w[2]; K[1, 2] = -w[0]
            K[2, 0] = -w[1]; K[2, 1] = w[0]
            V = torch.eye(3, device=omega.device) + (1 - torch.cos(theta)) / theta**2 * K + (theta - torch.sin(theta)) / theta**3 * torch.matmul(K, K)
            t = torch.matmul(V, v * dt)
        T = torch.eye(4, device=omega.device)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

class AdaptiveMultiRateIntegrator:
    def __init__(self, fast_solver, slow_solver, partition_fn):
        self.fast = fast_solver
        self.slow = slow_solver
        self.partition = partition_fn

    def step(self, f_fast, f_slow, t, y, dt_slow, n_fast_per_slow):
        y_slow_idx, y_fast_idx = self.partition(y)
        y_slow = y[y_slow_idx]
        y_fast = y[y_fast_idx]
        dy_slow = f_slow(t, y)
        y_slow_next = y_slow + dt_slow * dy_slow[y_slow_idx]
        dt_fast = dt_slow / n_fast_per_slow
        for i in range(n_fast_per_slow):
            t_fast = t + i * dt_fast
            y_combined = y.clone()
            y_combined[y_slow_idx] = y_slow + (i / n_fast_per_slow) * (y_slow_next - y_slow)
            dy_fast = f_fast(t_fast, y_combined)
            y_fast = y_fast + dt_fast * dy_fast[y_fast_idx]
        y_next = y.clone()
        y_next[y_slow_idx] = y_slow_next
        y_next[y_fast_idx] = y_fast
        return y_next

class ExponentialIntegrator:
    def __init__(self, linear_part, nonlinear_fn):
        self.A = linear_part
        self.N = nonlinear_fn

    def etd1(self, t, y, dt):
        exp_A = torch.matrix_exp(self.A * dt)
        A_inv = torch.linalg.inv(self.A)
        phi1 = torch.matmul(A_inv, exp_A - torch.eye(self.A.shape[0], device=y.device))
        return torch.matmul(exp_A, y) + dt * torch.matmul(phi1, self.N(t, y))

    def etd_rk2(self, t, y, dt):
        exp_A = torch.matrix_exp(self.A * dt)
        exp_A_half = torch.matrix_exp(self.A * dt / 2)
        A_inv = torch.linalg.inv(self.A)
        phi1 = torch.matmul(A_inv, exp_A - torch.eye(self.A.shape[0], device=y.device))
        N_n = self.N(t, y)
        a_n = torch.matmul(exp_A_half, y) + (dt / 2) * torch.matmul(A_inv, exp_A_half - torch.eye(self.A.shape[0], device=y.device)) @ N_n
        N_a = self.N(t + dt/2, a_n)
        return torch.matmul(exp_A, y) + dt * torch.matmul(phi1, 2 * N_a - N_n)

class ImplicitMidpointRule:
    def __init__(self, max_iter=50, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def step(self, f, t, y, dt):
        y_mid = y.clone()
        for _ in range(self.max_iter):
            y_mid_old = y_mid.clone()
            f_mid = f(t + dt/2, y_mid)
            y_mid = y + (dt/2) * f_mid
            if torch.norm(y_mid - y_mid_old) < self.tol:
                break
        return y + dt * f(t + dt/2, y_mid)

class GalerkinProjectionODE(nn.Module):
    def __init__(self, basis_fns, n_modes):
        super().__init__()
        self.basis_fns = basis_fns
        self.n_modes = n_modes
        self.coupling_tensor = nn.Parameter(torch.randn(n_modes, n_modes, n_modes) * 0.01)
        self.linear_operator = nn.Parameter(torch.randn(n_modes, n_modes) * 0.01)
        self.forcing = nn.Parameter(torch.randn(n_modes) * 0.01)

    def forward(self, t, a):
        da_dt = torch.matmul(self.linear_operator, a) + self.forcing
        for i in range(self.n_modes):
            da_dt[i] += torch.einsum('jk,j,k->', self.coupling_tensor[i], a, a)
        return da_dt

class StiffnessDetector:
    def __init__(self, threshold=100.0):
        self.threshold = threshold

    def estimate_stiffness_ratio(self, f, t, y, dt):
        y_req = y.clone().requires_grad_(True)
        f_val = f(t, y_req)
        n = y.shape[0]
        J = torch.zeros(n, n, device=y.device)
        for i in range(n):
            J[i] = torch.autograd.grad(f_val[i], y_req, retain_graph=True)[0]
        eigenvalues = torch.linalg.eigvals(J)
        real_parts = eigenvalues.real
        max_eig = torch.max(torch.abs(real_parts))
        min_eig = torch.min(torch.abs(real_parts[real_parts.abs() > 1e-12]))
        ratio = max_eig / (min_eig + 1e-12)
        return ratio.item()

    def is_stiff(self, f, t, y, dt):
        return self.estimate_stiffness_ratio(f, t, y, dt) > self.threshold

class AutoSwitchingSolver:
    def __init__(self, explicit_solver, implicit_solver, detector):
        self.explicit = explicit_solver
        self.implicit = implicit_solver
        self.detector = detector
        self.check_interval = 50
        self.step_count = 0
        self.using_implicit = False

    def step(self, f, t, y, dt, args=None):
        self.step_count += 1
        if self.step_count % self.check_interval == 0:
            self.using_implicit = self.detector.is_stiff(f, t, y, dt)
        if self.using_implicit:
            return self.implicit.step(f, t, y, dt)
        else:
            return self.explicit.step(f, t, y, dt, args)
