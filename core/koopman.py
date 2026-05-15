import torch
import torch.nn as nn
import torch.linalg

class ObservableDictionary(nn.Module):
    def __init__(self, state_dim, num_observables, poly_degree=2):
        super().__init__()
        self.state_dim = state_dim
        self.num_observables = num_observables
        self.poly_degree = poly_degree
        self.linear_dict = nn.Linear(state_dim, num_observables)
        self.rbf_centers = nn.Parameter(torch.randn(num_observables, state_dim))
        self.rbf_widths = nn.Parameter(torch.ones(num_observables))
    def radial_basis(self, x):
        dist = torch.cdist(x, self.rbf_centers)
        return torch.exp(-dist**2 / (self.rbf_widths**2 + 1e-8))
    def polynomial_basis(self, x):
        features = [x]
        for d in range(2, self.poly_degree + 1):
            features.append(x**d)
        return torch.cat(features, dim=-1)
    def forward(self, x):
        phi_lin = self.linear_dict(x)
        phi_rbf = self.radial_basis(x)
        phi_poly = self.polynomial_basis(x)
        return torch.cat([x, phi_lin, phi_rbf, phi_poly], dim=-1)

class ExactDMD:
    def __init__(self, svd_rank=None):
        self.svd_rank = svd_rank
        self.K = None
        self.eigenvalues = None
        self.eigenvectors = None
    def fit(self, X, Y):
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if self.svd_rank is not None:
            U = U[:, :self.svd_rank]
            S = S[:self.svd_rank]
            Vh = Vh[:self.svd_rank, :]
        S_inv = torch.diag(1.0 / (S + 1e-12))
        self.K = torch.linalg.multi_dot([U.T, Y, Vh.T, S_inv])
        self.eigenvalues, self.eigenvectors = torch.linalg.eig(self.K)
        return self
    def predict(self, X, steps=1):
        curr = X @ self.K
        traj = [curr]
        for _ in range(steps - 1):
            curr = curr @ self.K
            traj.append(curr)
        return torch.stack(traj, dim=1)

class ExtendedDMD:
    def __init__(self, dictionary, svd_rank=None):
        self.dict = dictionary
        self.svd_rank = svd_rank
        self.dmd = ExactDMD(svd_rank=svd_rank)
    def fit(self, X, Y):
        Phi_X = self.dict(X)
        Phi_Y = self.dict(Y)
        self.dmd.fit(Phi_X, Phi_Y)
        return self
    def predict(self, X, steps=1):
        Phi_X = self.dict(X)
        pred_Phi = self.dmd.predict(Phi_X, steps)
        return pred_Phi[..., :X.shape[-1]]

class DeepKoopmanOperator(nn.Module):
    def __init__(self, state_dim, latent_dim, num_eigs, encoder_layers, decoder_layers):
        super().__init__()
        enc = []
        curr = state_dim
        for h in encoder_layers:
            enc.extend([nn.Linear(curr, h), nn.ReLU()])
            curr = h
        enc.append(nn.Linear(curr, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        curr = latent_dim
        for h in decoder_layers:
            dec.extend([nn.Linear(curr, h), nn.ReLU()])
            curr = h
        dec.append(nn.Linear(curr, state_dim))
        self.decoder = nn.Sequential(*dec)
        self.K_matrix = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.K_complex = nn.Parameter(torch.randn(num_eigs, 2))
    def block_diagonal_K(self):
        blocks = []
        for i in range(self.K_complex.shape[0]):
            alpha, beta = self.K_complex[i, 0], self.K_complex[i, 1]
            blocks.append(torch.tensor([[alpha, -beta], [beta, alpha]], device=self.K_complex.device))
        return torch.block_diag(*blocks)
    def forward(self, x, steps=1):
        z = self.encoder(x)
        K = self.block_diagonal_K()
        pad_dim = K.shape[0] - z.shape[-1]
        if pad_dim > 0:
            z_pad = torch.nn.functional.pad(z, (0, pad_dim))
        else:
            z_pad = z[..., :K.shape[0]]
        traj_z = []
        curr = z_pad
        for _ in range(steps):
            curr = curr @ K.T
            traj_z.append(curr)
        traj_z = torch.stack(traj_z, dim=1)
        if pad_dim > 0:
            traj_z_reduced = traj_z[..., :-pad_dim]
        else:
            traj_z_reduced = traj_z
        return self.decoder(traj_z_reduced)

class ContinuousKoopmanGenerator(nn.Module):
    def __init__(self, dictionary, latent_dim):
        super().__init__()
        self.dict = dictionary
        self.L = nn.Parameter(torch.randn(latent_dim, latent_dim))
    def forward(self, x, t):
        z = self.dict(x)
        exp_L = torch.matrix_exp(self.L * t)
        z_t = z @ exp_L.T
        return z_t[..., :x.shape[-1]]
    def generator_loss(self, x, dx_dt):
        z = self.dict(x)
        x_req = x.clone().requires_grad_(True)
        z_req = self.dict(x_req)
        dz_dx = torch.zeros(x.shape[0], z.shape[-1], x.shape[-1], device=x.device)
        for i in range(z.shape[-1]):
            dz_dx[:, i, :] = torch.autograd.grad(z_req[:, i].sum(), x_req, retain_graph=True)[0]
        dz_dt = torch.bmm(dz_dx, dx_dt.unsqueeze(-1)).squeeze(-1)
        pred_dz_dt = z @ self.L.T
        return torch.mean((dz_dt - pred_dz_dt)**2)

class KoopmanMPC:
    def __init__(self, koopman_model, horizon, Q, R):
        self.model = koopman_model
        self.horizon = horizon
        self.Q = Q
        self.R = R
    def optimize(self, x0, u_bounds, max_iter=100):
        z0 = self.model.dict(x0)
        u_seq = torch.zeros(self.horizon, u_bounds.shape[0], requires_grad=True, device=x0.device)
        optimizer = torch.optim.LBFGS([u_seq], max_iter=20)
        B_z = self.model.B_matrix
        def closure():
            optimizer.zero_grad()
            cost = 0
            z_curr = z0
            for t in range(self.horizon):
                z_curr = z_curr @ self.model.K.T + u_seq[t] @ B_z.T
                cost += z_curr @ self.Q @ z_curr.T + u_seq[t] @ self.R @ u_seq[t].T
            cost.backward()
            return cost
        for _ in range(max_iter):
            optimizer.step(closure)
            with torch.no_grad():
                u_seq.clamp_(u_bounds[:, 0], u_bounds[:, 1])
        return u_seq

class ParametricKoopman(nn.Module):
    def __init__(self, state_dim, param_dim, latent_dim, dict_layers):
        super().__init__()
        net = []
        curr = state_dim + param_dim
        for h in dict_layers:
            net.extend([nn.Linear(curr, h), nn.ELU()])
            curr = h
        net.append(nn.Linear(curr, latent_dim))
        self.dict = nn.Sequential(*net)
        self.K_net = nn.Sequential(
            nn.Linear(param_dim, latent_dim * latent_dim)
        )
        self.latent_dim = latent_dim
    def forward(self, x, p, steps=1):
        xp = torch.cat([x, p], dim=-1)
        z = self.dict(xp)
        K = self.K_net(p).view(-1, self.latent_dim, self.latent_dim)
        traj = []
        curr = z.unsqueeze(-1)
        for _ in range(steps):
            curr = torch.bmm(K, curr)
            traj.append(curr.squeeze(-1))
        return torch.stack(traj, dim=1)

class TimeDelayEmbedding(nn.Module):
    def __init__(self, delay_steps, step_size=1):
        super().__init__()
        self.delay_steps = delay_steps
        self.step_size = step_size
    def forward(self, x_seq):
        batch, seq_len, dim = x_seq.shape
        embedded = []
        for i in range(self.delay_steps * self.step_size, seq_len):
            emb = x_seq[:, i - self.delay_steps * self.step_size : i + 1 : self.step_size, :].reshape(batch, -1)
            embedded.append(emb)
        return torch.stack(embedded, dim=1)

class HankelKoopmanDMD:
    def __init__(self, delay_steps, svd_rank=None):
        self.delay = TimeDelayEmbedding(delay_steps)
        self.dmd = ExactDMD(svd_rank)
    def fit(self, X_seq):
        H = self.delay(X_seq)
        H_x = H[:, :-1, :].reshape(-1, H.shape[-1])
        H_y = H[:, 1:, :].reshape(-1, H.shape[-1])
        self.dmd.fit(H_x, H_y)
        return self

class TensorKoopman:
    def __init__(self, dims, rank):
        self.dims = dims
        self.rank = rank
        self.core = nn.Parameter(torch.randn([rank] * len(dims)))
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(d, rank)) for d in dims])
    def reconstruct_K(self):
        K = self.core
        for i, f in enumerate(self.factors):
            K = torch.tensordot(K, f, dims=([0], [1]))
        return K
    def step(self, z):
        K = self.reconstruct_K()
        return torch.tensordot(K, z, dims=(list(range(len(self.dims))), list(range(len(self.dims)))))
