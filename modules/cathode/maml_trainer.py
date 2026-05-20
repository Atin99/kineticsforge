import torch
import torch.nn as nn
from copy import deepcopy

class MAMLTrainer:
    """First-order MAML trainer.

    This implementation adapts a cloned model per task and transfers query
    gradients back to the meta model. That is FOMAML by design; it does not
    claim second-order MAML gradients through the inner optimizer.
    """

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5, first_order=True):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_steps = inner_steps
        self.first_order = first_order
        if not first_order:
            raise NotImplementedError("Second-order MAML needs a differentiable inner update; use FOMAML here or core_trainer.py.")
    def clone_model(self):
        return deepcopy(self.model)
    def inner_loop(self, task_support, cloned_model):
        optimizer = torch.optim.SGD(cloned_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            x, y = task_support
            pred = cloned_model(*x)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            # FOMAML intentionally avoids create_graph=True in the inner loop.
            loss.backward()
            optimizer.step()
        return cloned_model
    def meta_train_step(self, tasks):
        meta_loss = 0.0
        self.meta_optimizer.zero_grad()
        for support, query in tasks:
            cloned = self.clone_model()
            adapted = self.inner_loop(support, cloned)
            qx, qy = query
            pred = adapted(*qx)
            loss = nn.functional.mse_loss(pred, qy)
            loss.backward()
            for p_meta, p_adapted in zip(self.model.parameters(), adapted.parameters()):
                if p_meta.grad is None:
                    p_meta.grad = torch.zeros_like(p_meta)
                if p_adapted.grad is not None:
                    p_meta.grad += p_adapted.grad
            meta_loss += loss.item()
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= len(tasks)
        self.meta_optimizer.step()
        return meta_loss / len(tasks)

class ANILTrainer:
    def __init__(self, feature_extractor, head, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
        self.feat = feature_extractor
        self.head = head
        self.inner_lr = inner_lr
        self.meta_opt = torch.optim.Adam(list(self.feat.parameters()) + list(self.head.parameters()), lr=meta_lr)
        self.inner_steps = inner_steps
    def inner_loop(self, support_features, support_y, cloned_head):
        opt = torch.optim.SGD(cloned_head.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            pred = cloned_head(support_features)
            loss = nn.functional.mse_loss(pred, support_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return cloned_head
    def meta_train_step(self, tasks):
        meta_loss = 0.0
        self.meta_opt.zero_grad()
        for support, query in tasks:
            sx, sy = support
            qx, qy = query
            sf = self.feat(*sx).detach()
            qf = self.feat(*qx)
            c_head = deepcopy(self.head)
            a_head = self.inner_loop(sf, sy, c_head)
            pred = a_head(qf)
            loss = nn.functional.mse_loss(pred, qy)
            loss.backward()
            for pm, pa in zip(self.head.parameters(), a_head.parameters()):
                if pm.grad is None: pm.grad = torch.zeros_like(pm)
                if pa.grad is not None: pm.grad += pa.grad
            meta_loss += loss.item()
        for p in self.head.parameters():
            if p.grad is not None: p.grad /= len(tasks)
        self.meta_opt.step()
        return meta_loss / len(tasks)

class ReptileTrainer:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
    def meta_train_step(self, task):
        c_model = deepcopy(self.model)
        opt = torch.optim.SGD(c_model.parameters(), lr=self.inner_lr)
        x, y = task
        for _ in range(self.inner_steps):
            pred = c_model(*x)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for pm, pc in zip(self.model.parameters(), c_model.parameters()):
                pm.data += self.meta_lr * (pc.data - pm.data)

class iMAMLTrainer:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, cg_steps=5, inner_steps=5, reg=0.1):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_opt = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.cg_steps = cg_steps
        self.inner_steps = inner_steps
        self.reg = reg
    def inner_loop(self, x, y, c_model):
        opt = torch.optim.SGD(c_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            pred = c_model(*x)
            loss = nn.functional.mse_loss(pred, y)
            for p, cp in zip(self.model.parameters(), c_model.parameters()):
                loss += self.reg * torch.sum((p - cp)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return c_model
    def cg_solve(self, A_fn, b, steps):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        for _ in range(steps):
            Ap = A_fn(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-6:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x
    def meta_train_step(self, tasks):
        meta_loss = 0.0
        self.meta_opt.zero_grad()
        for support, query in tasks:
            sx, sy = support
            qx, qy = query
            c_model = deepcopy(self.model)
            a_model = self.inner_loop(sx, sy, c_model)
            pred = a_model(*qx)
            loss = nn.functional.mse_loss(pred, qy)
            loss_grads = torch.autograd.grad(loss, a_model.parameters())
            flat_loss_grads = torch.cat([g.contiguous().view(-1) for g in loss_grads])
            def hvp(v):
                v_list = []
                idx = 0
                for p in a_model.parameters():
                    numel = p.numel()
                    v_list.append(v[idx:idx+numel].view(p.shape))
                    idx += numel
                inner_pred = a_model(*sx)
                inner_loss = nn.functional.mse_loss(inner_pred, sy)
                inner_grads = torch.autograd.grad(inner_loss, a_model.parameters(), create_graph=True)
                gvp = sum(torch.sum(ig * v) for ig, v in zip(inner_grads, v_list))
                hvp_grads = torch.autograd.grad(gvp, a_model.parameters(), retain_graph=True)
                return torch.cat([g.contiguous().view(-1) for g in hvp_grads]) + self.reg * v
            inv_hvp = self.cg_solve(hvp, flat_loss_grads, self.cg_steps)
            idx = 0
            for p in self.model.parameters():
                numel = p.numel()
                grad_upd = inv_hvp[idx:idx+numel].view(p.shape)
                if p.grad is None:
                    p.grad = grad_upd
                else:
                    p.grad += grad_upd
                idx += numel
            meta_loss += loss.item()
        for p in self.model.parameters():
            p.grad /= len(tasks)
        self.meta_opt.step()
        return meta_loss / len(tasks)
