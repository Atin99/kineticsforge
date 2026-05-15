import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutWrapper(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base = base_model
        self.p = dropout_rate

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def predict_with_uncertainty(self, *args, n_samples=50, **kwargs):
        self.train()
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(*args, **kwargs)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)
        self.eval()
        return mean, epistemic

class DeepEnsemble:
    def __init__(self, model_class, model_kwargs, n_models=5):
        self.models = [model_class(**model_kwargs) for _ in range(n_models)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in self.models]

    def train_step(self, x, y, loss_fn):
        losses = []
        for model, opt in zip(self.models, self.optimizers):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def predict(self, x):
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x))
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.var(dim=0)

class HeteroscedasticHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.log_var_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.mean_head(x)
        log_var = self.log_var_head(x)
        return mu, log_var

    def loss(self, x, y):
        mu, log_var = self.forward(x)
        precision = torch.exp(-log_var)
        return torch.mean(precision * (y - mu)**2 + log_var)

class CalibratedUncertainty:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def expected_calibration_error(self, means, stds, targets):
        z_scores = torch.abs(targets - means) / (stds + 1e-8)
        confidences = torch.linspace(0.1, 0.99, self.n_bins)
        from scipy.stats import norm
        ece = 0.0
        for conf in confidences:
            z_thresh = norm.ppf((1 + conf.item()) / 2)
            empirical = (z_scores < z_thresh).float().mean()
            ece += abs(empirical.item() - conf.item())
        return ece / self.n_bins

    def temperature_scale(self, log_var, temperature):
        return log_var + torch.log(torch.tensor(temperature))
