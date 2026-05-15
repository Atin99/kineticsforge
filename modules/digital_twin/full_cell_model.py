import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(2, self.width)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class P2DContinuumSolver:
    def __init__(self, nx=50, nr=20):
        self.nx = nx
        self.nr = nr
        self.dx = 1.0 / nx
        self.dr = 1.0 / nr
        self.D_s_anode = 1e-14
        self.D_s_cathode = 1e-13
        self.D_e = 1e-10
        self.t_plus = 0.36
        self.kappa = 1.0
        self.sigma_anode = 100.0
        self.sigma_cathode = 10.0
        self.F = 96485.0
        self.R = 8.314

    def solid_diffusion_spherical(self, c_s, D_s):
        batch = c_s.shape[0]
        dc_dt = torch.zeros_like(c_s)
        r = torch.linspace(self.dr, 1.0, self.nr, device=c_s.device).unsqueeze(0)
        c_i_plus_1 = c_s[:, 2:]
        c_i = c_s[:, 1:-1]
        c_i_minus_1 = c_s[:, :-2]
        r_i = r[:, 1:-1]
        dc_dt[:, 1:-1] = D_s * ( (c_i_plus_1 - 2*c_i + c_i_minus_1) / self.dr**2 + (2/r_i) * (c_i_plus_1 - c_i_minus_1) / (2*self.dr) )
        dc_dt[:, 0] = D_s * 6 * (c_s[:, 1] - c_s[:, 0]) / self.dr**2
        return dc_dt

    def electrolyte_diffusion(self, c_e, j_Li):
        dc_dt = torch.zeros_like(c_e)
        c_i_plus_1 = c_e[:, 2:]
        c_i = c_e[:, 1:-1]
        c_i_minus_1 = c_e[:, :-2]
        dc_dt[:, 1:-1] = self.D_e * (c_i_plus_1 - 2*c_i + c_i_minus_1) / self.dx**2 + (1 - self.t_plus) * j_Li[:, 1:-1] / self.F
        dc_dt[:, 0] = dc_dt[:, 1]
        dc_dt[:, -1] = dc_dt[:, -2]
        return dc_dt

    def butler_volmer(self, eta, k0, c_e, c_s_surf, c_s_max, T):
        alpha = 0.5
        exchange_current = k0 * (c_e**alpha) * ((c_s_max - c_s_surf)**alpha) * (c_s_surf**alpha)
        return exchange_current * (torch.exp(alpha * self.F * eta / (self.R * T)) - torch.exp(-(1 - alpha) * self.F * eta / (self.R * T)))

    def step(self, state_dict, I_app, T, dt):
        c_s_anode = state_dict['c_s_anode']
        c_s_cathode = state_dict['c_s_cathode']
        c_e = state_dict['c_e']
        phi_s = state_dict['phi_s']
        phi_e = state_dict['phi_e']
        
        eta_anode = phi_s[:, :self.nx//2] - phi_e[:, :self.nx//2]
        eta_cathode = phi_s[:, self.nx//2:] - phi_e[:, self.nx//2:]
        
        j_anode = self.butler_volmer(eta_anode, 1e-5, c_e[:, :self.nx//2], c_s_anode[:, -1], 30000.0, T)
        j_cathode = self.butler_volmer(eta_cathode, 1e-4, c_e[:, self.nx//2:], c_s_cathode[:, -1], 50000.0, T)
        j_Li = torch.cat([j_anode, j_cathode], dim=1)
        
        dc_s_anode_dt = self.solid_diffusion_spherical(c_s_anode, self.D_s_anode)
        dc_s_cathode_dt = self.solid_diffusion_spherical(c_s_cathode, self.D_s_cathode)
        
        dc_s_anode_dt[:, -1] -= j_anode / self.F
        dc_s_cathode_dt[:, -1] -= j_cathode / self.F
        
        dc_e_dt = self.electrolyte_diffusion(c_e, j_Li)
        
        c_s_anode_next = c_s_anode + dt * dc_s_anode_dt
        c_s_cathode_next = c_s_cathode + dt * dc_s_cathode_dt
        c_e_next = c_e + dt * dc_e_dt
        
        return {
            'c_s_anode': c_s_anode_next,
            'c_s_cathode': c_s_cathode_next,
            'c_e': c_e_next,
            'phi_s': phi_s,
            'phi_e': phi_e
        }

class PhysicsInformedNeuralNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f"tanh_{i}", nn.Tanh())
        self.net.add_module(f"linear_out", nn.Linear(layers[-2], layers[-1]))
        
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)
        
    def compute_physics_loss(self, x, t, D):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.forward(x, t)
        
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        
        pde_residual = du_dt - D * d2u_dx2
        return torch.mean(pde_residual**2)

class HybridDigitalTwin:
    def __init__(self, nx=50, nr=20):
        self.p2d = P2DContinuumSolver(nx, nr)
        self.fno_surrogate = FNO1d(modes=16, width=64)
        self.pinn_electrolyte = PhysicsInformedNeuralNetwork([2, 50, 50, 50, 1])
        self.state = self.initialize_state(1, nx, nr)
        
    def initialize_state(self, batch, nx, nr):
        return {
            'c_s_anode': torch.ones(batch, nr) * 25000.0,
            'c_s_cathode': torch.ones(batch, nr) * 10000.0,
            'c_e': torch.ones(batch, nx) * 1000.0,
            'phi_s': torch.ones(batch, nx) * 3.8,
            'phi_e': torch.zeros(batch, nx)
        }
        
    def simulate_hybrid(self, I_profile, T_profile, dt, steps):
        traj = []
        curr_state = self.state
        for i in range(steps):
            I = I_profile[i]
            T = T_profile[i]
            # Use exact P2D for solid phase
            next_state = self.p2d.step(curr_state, I, T, dt)
            # Use FNO surrogate for fast electrolyte potential updates
            phi_e_pred = self.fno_surrogate(next_state['c_e'].unsqueeze(-1))
            next_state['phi_e'] = phi_e_pred.squeeze(-1)
            curr_state = next_state
            traj.append(curr_state)
        return traj

class ThreeDimensionalThermalCoupling(nn.Module):
    def __init__(self, nx, ny, nz):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = 0.01
        self.k_th = 0.5
        self.rho_Cp = 2000000.0
        
    def forward(self, T, q_gen, dt):
        T_pad = F.pad(T.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='replicate').squeeze(0).squeeze(0)
        
        dT_dx2 = (T_pad[2:, 1:-1, 1:-1] - 2*T + T_pad[:-2, 1:-1, 1:-1]) / self.dx**2
        dT_dy2 = (T_pad[1:-1, 2:, 1:-1] - 2*T + T_pad[1:-1, :-2, 1:-1]) / self.dx**2
        dT_dz2 = (T_pad[1:-1, 1:-1, 2:] - 2*T + T_pad[1:-1, 1:-1, :-2]) / self.dx**2
        
        nabla2_T = dT_dx2 + dT_dy2 + dT_dz2
        dT_dt = (self.k_th * nabla2_T + q_gen) / self.rho_Cp
        return T + dt * dT_dt

class FullCellDigitalTwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.electrochem = HybridDigitalTwin(nx=100, nr=30)
        self.thermal = ThreeDimensionalThermalCoupling(10, 10, 20)
        self.aging_sde = torch.nn.GRU(input_size=5, hidden_size=32, num_layers=3)
        self.degradation_mapper = nn.Linear(32, 2)
        
    def step(self, I_app, T_amb, dt):
        q_gen = torch.abs(I_app) * 0.05 + 0.1 # Joule + Entropic heating
        
        T_3d = torch.ones(10, 10, 20) * T_amb
        T_next = self.thermal(T_3d, q_gen, dt)
        T_avg = torch.mean(T_next)
        
        self.electrochem.state = self.electrochem.p2d.step(self.electrochem.state, I_app, T_avg, dt)
        
        sde_in = torch.tensor([I_app, T_avg, self.electrochem.state['c_e'].mean(), self.electrochem.state['c_s_anode'].mean(), self.electrochem.state['c_s_cathode'].mean()]).unsqueeze(0).unsqueeze(0)
        out, _ = self.aging_sde(sde_in)
        degrad = self.degradation_mapper(out.squeeze())
        
        self.electrochem.p2d.D_s_anode *= (1.0 - degrad[0] * dt)
        self.electrochem.p2d.sigma_cathode *= (1.0 - degrad[1] * dt)
        
        return self.electrochem.state, T_next
