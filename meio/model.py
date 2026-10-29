"""ResMLP multi-exit reconstruction; residual wrappers adapted from official LLIO.
MEIO's missing backbone is not represented as released author code.
See upstream/llio/model_MLP.py and PROVENANCE.md.
"""
import math
import torch
from torch import nn

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, x):
        return x * self.g + self.b

class Residual(nn.Module):
    # Official LLIO PreAffinePostLayerScale, with a descriptive local name.
    def __init__(self, dim, fn):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, 1, dim), 0.1))
        self.affine = Affine(dim); self.affine_out = Affine(dim); self.fn = fn
    def forward(self, x):
        return self.affine_out(x + self.fn(self.affine(x)) * self.scale)

class Block(nn.Module):
    def __init__(self, tokens, dim, expansion, dropout):
        super().__init__()
        self.tokens = Residual(dim, nn.Conv1d(tokens, tokens, 1, bias=False))
        hidden = int(dim * expansion)
        self.channels = Residual(dim, nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                                    nn.Dropout(dropout), nn.Linear(hidden, dim)))
    def forward(self, x):
        return self.channels(self.tokens(x))

class Head(nn.Module):
    def __init__(self, dim, low, high):
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.log_sigma = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.low, self.high = math.log(low), math.log(high)
        nn.init.normal_(self.mean[-1].weight, std=.01); nn.init.zeros_(self.mean[-1].bias)
        nn.init.zeros_(self.log_sigma[-1].weight); nn.init.constant_(self.log_sigma[-1].bias, math.log(.2))
    def forward(self, x):
        x = x.mean(1)
        return self.mean(x), self.log_sigma(x).clamp(self.low, self.high)

class MEIO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg); self.depth = cfg["depth"]
        tokens = cfg["window"] // cfg["patch"]
        self.embedding = nn.Conv1d(6, cfg["dim"], cfg["patch"], stride=cfg["patch"])
        self.blocks = nn.ModuleList([Block(tokens, cfg["dim"], cfg["expansion"], cfg["dropout"]) for _ in range(self.depth)])
        self.heads = nn.ModuleList([Head(cfg["dim"], cfg["min_sigma"], cfg["max_sigma"]) for _ in range(self.depth)])
    def embed(self, x):
        if x.ndim != 3 or x.shape[1:] != (6, self.cfg["window"]):
            raise ValueError("expected [batch, 6, window]")
        return self.embedding(x).transpose(1, 2)
    def forward(self, x):
        h = self.embed(x); means, logs = [], []
        for block, head in zip(self.blocks, self.heads):
            h = block(h); mean, log = head(h); means.append(mean); logs.append(log)
        return torch.stack(means, 1), torch.stack(logs, 1)
    def final(self, x):
        h = self.embed(x)
        for block in self.blocks: h = block(h)
        return self.heads[-1](h)
    @torch.no_grad()
    def dynamic(self, x, threshold, scales=None):
        if self.training: raise RuntimeError("dynamic routing requires model.eval()")
        if not math.isfinite(threshold) or threshold < 0: raise ValueError("threshold must be finite and nonnegative")
        scales = torch.ones(self.depth, 3, device=x.device) if scales is None else torch.as_tensor(scales, device=x.device, dtype=x.dtype)
        if scales.shape != (self.depth, 3) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("invalid calibration scales")
        ids = torch.arange(len(x), device=x.device); h = self.embed(x)
        means = x.new_empty(len(x), 3); sigmas = x.new_empty(len(x), 3)
        exits = torch.empty(len(x), dtype=torch.long, device=x.device)
        for i, (block, head) in enumerate(zip(self.blocks, self.heads)):
            h = block(h); mean, log = head(h); sigma = log.exp() * scales[i]
            stop = torch.linalg.vector_norm(sigma, dim=1) <= threshold
            if i == self.depth - 1: stop = torch.ones_like(stop)
            chosen = ids[stop]; means[chosen] = mean[stop]; sigmas[chosen] = sigma[stop]; exits[chosen] = i
            ids = ids[~stop]; h = h[~stop]
            if len(ids) == 0: break
        return means, sigmas, exits
    def operation_costs(self):
        # Linear/Conv multiply-add = 2 FLOPs. Excludes activations/routing/memory.
        d = self.cfg["dim"]; t = self.cfg["window"] // self.cfg["patch"]; h = int(d*self.cfg["expansion"])
        embedding = 2*t*d*6*self.cfg["patch"]
        block = 2*t*t*d + 4*t*d*h
        head = 4*(d*d + d*3)
        return dict(final=embedding+self.depth*block+head,
                    dynamic=[embedding+(i+1)*(block+head) for i in range(self.depth)])

def exit_weights(depth, gamma, device):
    # Adapted from official EarlyExistPDRModule.gen_scale_array.
    w = (torch.arange(1, depth+1, device=device, dtype=torch.float32)/depth).pow(gamma)
    return w / w.sum()

def losses(means, log_sigma, target, weights, nll_fraction):
    if not 0 <= nll_fraction <= 1: raise ValueError("invalid NLL mixing fraction")
    means, log_sigma, target = means.float(), log_sigma.float(), target.float()
    error = means - target[:, None, :]
    mse = error.square().mean((0, 2))
    # Official log-standard-deviation convention, corrected whole-NLL weighting.
    nll = (.5 * error.square() * torch.exp(-2*log_sigma) + log_sigma).mean((0, 2))
    return (1-nll_fraction)*(weights*mse).sum() + nll_fraction*(weights*nll).sum()
