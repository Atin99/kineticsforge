import torch
import torch.nn as nn
import itertools
import math

class SparseRegressionOptimizer:
    def __init__(self, threshold, alpha=0.1, max_iter=1000, tol=1e-5):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    def _soft_threshold(self, x, l):
        return torch.sign(x) * torch.relu(torch.abs(x) - l)
    def _hard_threshold(self, x, l):
        mask = torch.abs(x) > l
        return x * mask

class STLSQ(SparseRegressionOptimizer):
    def optimize(self, Theta, X_dot):
        coef = torch.linalg.lstsq(Theta, X_dot).solution
        for _ in range(self.max_iter):
            small_inds = torch.abs(coef) < self.threshold
            coef[small_inds] = 0.0
            for i in range(X_dot.shape[1]):
                big_inds = ~small_inds[:, i]
                if torch.sum(big_inds) > 0:
                    coef[big_inds, i] = torch.linalg.lstsq(Theta[:, big_inds], X_dot[:, i]).solution
            if torch.all(torch.abs(coef[small_inds]) < 1e-10):
                break
        return coef

class SR3(SparseRegressionOptimizer):
    def optimize(self, Theta, X_dot):
        nu = 1.0
        n_features = Theta.shape[1]
        n_targets = X_dot.shape[1]
        W = torch.linalg.lstsq(Theta, X_dot).solution
        C = W.clone()
        H = torch.matmul(Theta.T, Theta) + nu * torch.eye(n_features, device=Theta.device)
        invH = torch.inverse(H)
        Theta_X = torch.matmul(Theta.T, X_dot)
        for _ in range(self.max_iter):
            W_prev = W.clone()
            W = torch.matmul(invH, Theta_X + nu * C)
            C = self._hard_threshold(W, self.threshold)
            if torch.norm(W - W_prev) < self.tol:
                break
        return C

class LASSO(SparseRegressionOptimizer):
    def optimize(self, Theta, X_dot):
        W = torch.zeros(Theta.shape[1], X_dot.shape[1], device=Theta.device)
        L = torch.max(torch.linalg.eigvalsh(torch.matmul(Theta.T, Theta)))
        for _ in range(self.max_iter):
            W_prev = W.clone()
            grad = torch.matmul(Theta.T, torch.matmul(Theta, W) - X_dot)
            W = self._soft_threshold(W - grad / L, self.alpha / L)
            if torch.norm(W - W_prev) < self.tol:
                break
        return W

class FeatureLibrary:
    def __init__(self, include_bias=True):
        self.include_bias = include_bias
        self.functions = []
        self.n_features = 0
    def fit(self, X):
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError

class PolynomialLibrary(FeatureLibrary):
    def __init__(self, degree, include_bias=True):
        super().__init__(include_bias)
        self.degree = degree
    def fit(self, X):
        self.n_input_features = X.shape[1]
        self.combinations = []
        for d in range(1 if not self.include_bias else 0, self.degree + 1):
            self.combinations.extend(list(itertools.combinations_with_replacement(range(self.n_input_features), d)))
        self.n_features = len(self.combinations)
        return self
    def transform(self, X):
        features = torch.ones(X.shape[0], self.n_features, device=X.device)
        for i, comb in enumerate(self.combinations):
            if len(comb) > 0:
                prod = X[:, comb[0]]
                for idx in comb[1:]:
                    prod = prod * X[:, idx]
                features[:, i] = prod
        return features

class FourierLibrary(FeatureLibrary):
    def __init__(self, n_frequencies, include_bias=True):
        super().__init__(include_bias)
        self.n_frequencies = n_frequencies
    def fit(self, X):
        self.n_input_features = X.shape[1]
        self.n_features = self.n_input_features * self.n_frequencies * 2
        if self.include_bias:
            self.n_features += 1
        return self
    def transform(self, X):
        features = torch.ones(X.shape[0], self.n_features, device=X.device)
        idx = 1 if self.include_bias else 0
        for f in range(1, self.n_frequencies + 1):
            for i in range(self.n_input_features):
                features[:, idx] = torch.sin(f * X[:, i])
                features[:, idx+1] = torch.cos(f * X[:, i])
                idx += 2
        return features

class CustomLibrary(FeatureLibrary):
    def __init__(self, functions, include_bias=True):
        super().__init__(include_bias)
        self.funcs = functions
    def fit(self, X):
        self.n_input_features = X.shape[1]
        self.n_features = len(self.funcs)
        if self.include_bias:
            self.n_features += 1
        return self
    def transform(self, X):
        features = torch.ones(X.shape[0], self.n_features, device=X.device)
        idx = 1 if self.include_bias else 0
        for f in self.funcs:
            features[:, idx] = f(X)
            idx += 1
        return features

class GeneralizedSINDy:
    def __init__(self, optimizer, library):
        self.optimizer = optimizer
        self.library = library
        self.coefficients = None
    def fit(self, X, t=None, X_dot=None):
        if X_dot is None:
            if t is None:
                raise ValueError
            X_dot = self._differentiate(X, t)
        self.library.fit(X)
        Theta = self.library.transform(X)
        self.coefficients = self.optimizer.optimize(Theta, X_dot)
        return self
    def predict(self, X):
        Theta = self.library.transform(X)
        return torch.matmul(Theta, self.coefficients)
    def _differentiate(self, X, t):
        X_dot = torch.zeros_like(X)
        dt = t[1:] - t[:-1]
        X_dot[1:-1] = (X[2:] - X[:-2]) / (dt[:-1] + dt[1:]).unsqueeze(-1)
        X_dot[0] = (X[1] - X[0]) / dt[0].unsqueeze(-1)
        X_dot[-1] = (X[-1] - X[-2]) / dt[-1].unsqueeze(-1)
        return X_dot

class EnsembleSINDy:
    def __init__(self, base_optimizer, base_library, n_models=10, subset_frac=0.8):
        self.n_models = n_models
        self.subset_frac = subset_frac
        self.models = [GeneralizedSINDy(base_optimizer, base_library) for _ in range(n_models)]
    def fit(self, X, t=None, X_dot=None):
        n_samples = X.shape[0]
        subset_size = int(self.subset_frac * n_samples)
        for i in range(self.n_models):
            indices = torch.randperm(n_samples)[:subset_size]
            indices, _ = torch.sort(indices)
            X_sub = X[indices]
            t_sub = t[indices] if t is not None else None
            X_dot_sub = X_dot[indices] if X_dot is not None else None
            self.models[i].fit(X_sub, t_sub, X_dot_sub)
        return self
    def aggregate_coefficients(self):
        coefs = torch.stack([m.coefficients for m in self.models])
        return torch.mean(coefs, dim=0), torch.std(coefs, dim=0)

class ConstrainedSINDy(GeneralizedSINDy):
    def __init__(self, optimizer, library, equality_constraints=None):
        super().__init__(optimizer, library)
        self.eq_constraints = equality_constraints
    def fit(self, X, t=None, X_dot=None):
        if X_dot is None:
            X_dot = self._differentiate(X, t)
        self.library.fit(X)
        Theta = self.library.transform(X)
        if self.eq_constraints is not None:
            C, d = self.eq_constraints
            Theta_C = torch.matmul(Theta, C.T)
            Theta_proj = Theta - torch.matmul(Theta_C, torch.linalg.pinv(C.T))
            self.coefficients = self.optimizer.optimize(Theta_proj, X_dot)
        else:
            self.coefficients = self.optimizer.optimize(Theta, X_dot)
        return self

class ParetoSINDy:
    def __init__(self, library, thresholds):
        self.library = library
        self.thresholds = thresholds
        self.models = []
    def fit(self, X, t=None, X_dot=None):
        for th in self.thresholds:
            opt = STLSQ(threshold=th)
            model = GeneralizedSINDy(opt, self.library)
            model.fit(X, t, X_dot)
            self.models.append(model)
        return self
    def evaluate_pareto(self, X, X_dot):
        complexities = []
        errors = []
        for m in self.models:
            pred = m.predict(X)
            errors.append(torch.norm(pred - X_dot).item())
            complexities.append(torch.sum(m.coefficients != 0).item())
        return complexities, errors

class PhysicsInformedSINDy(GeneralizedSINDy):
    def __init__(self, optimizer, library, physics_loss_fn, physics_weight=0.1):
        super().__init__(optimizer, library)
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = physics_weight
    def fit(self, X, t=None, X_dot=None):
        if X_dot is None:
            X_dot = self._differentiate(X, t)
        self.library.fit(X)
        Theta = self.library.transform(X)
        coef = torch.zeros(Theta.shape[1], X_dot.shape[1], requires_grad=True, device=Theta.device)
        opt = torch.optim.Adam([coef], lr=0.01)
        for _ in range(self.optimizer.max_iter):
            opt.zero_grad()
            pred = torch.matmul(Theta, coef)
            data_loss = torch.mean((pred - X_dot)**2)
            phys_loss = self.physics_loss_fn(pred, X)
            reg_loss = self.optimizer.alpha * torch.sum(torch.abs(coef))
            loss = data_loss + self.physics_weight * phys_loss + reg_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                coef.data = self.optimizer._hard_threshold(coef.data, self.optimizer.threshold)
        self.coefficients = coef.detach()
        return self

def initialize_cathode_sindy(threshold=1e-3):
    funcs = [
        lambda x: x[:, 0],
        lambda x: x[:, 1],
        lambda x: x[:, 0]**2,
        lambda x: x[:, 1]**2,
        lambda x: x[:, 0]*x[:, 1],
        lambda x: x[:, 0]**3,
        lambda x: torch.sin(x[:, 1]),
        lambda x: torch.exp(-x[:, 0]),
        lambda x: x[:, 2],
        lambda x: x[:, 0]*x[:, 2],
        lambda x: x[:, 1]*x[:, 2],
    ]
    lib = CustomLibrary(funcs, include_bias=True)
    opt = STLSQ(threshold=threshold, max_iter=5000)
    return GeneralizedSINDy(opt, lib)
