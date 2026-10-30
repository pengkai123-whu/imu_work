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
    def __init__(self, dim, fn, init_scale=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, 1, dim), init_scale))
        self.affine = Affine(dim); self.affine_out = Affine(dim); self.fn = fn
    def forward(self, x):
        return self.affine_out(x + self.fn(self.affine(x)) * self.scale)

class TokenSSM(nn.Module):
    """Diagonal state-space map used as a causal replacement for token MLP mixing."""
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.logit_decay = nn.Parameter(torch.full((dim,), 1.5))

    def forward(self, tokens):
        value, gate = self.in_proj(tokens).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(tokens.dtype)
        state = torch.zeros_like(value[:, 0]); outputs = []
        for step in range(tokens.shape[1]):
            state = decay * state + (1.0 - decay) * value[:, step]
            outputs.append(state * torch.sigmoid(gate[:, step]))
        return self.out_proj(torch.stack(outputs, dim=1))


class Block(nn.Module):
    def __init__(self, tokens, dim, expansion, dropout, mixer="mlp"):
        super().__init__()
        self.tokens = Residual(dim, nn.Conv1d(tokens, tokens, 1, bias=False))
        hidden = int(dim * expansion)
        self.channels = Residual(dim, nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                                    nn.Dropout(dropout), nn.Linear(hidden, dim)))
        self.block_mixer = mixer
        self.ssm_channels = Residual(dim, TokenSSM(dim), init_scale=0.0) if mixer in {"ssm", "hybrid_ssm"} else None
        # Hybrid mode preserves the trained MLP path and lets the causal SSM
        # branch enter through a zero-initialized residual gate.
        self.ssm_gate = nn.Parameter(torch.zeros(1, 1, dim)) if mixer == "hybrid_ssm" else None
    def forward(self, x):
        x = self.tokens(x)
        if self.block_mixer == "ssm":
            return self.ssm_channels(x)
        if self.block_mixer == "hybrid_ssm":
            return self.channels(x) + self.ssm_gate * self.ssm_channels(x)
        return self.channels(x)

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
        selected_ssm = {int(i) for i in cfg.get("ssm_blocks", [])}
        blocks = []
        for index in range(self.depth):
            mixer = cfg.get("block_mixer", "mlp")
            if index in selected_ssm:
                mixer = "hybrid_ssm"
            blocks.append(Block(tokens, cfg["dim"], cfg["expansion"], cfg["dropout"], mixer))
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList([Head(cfg["dim"], cfg["min_sigma"], cfg["max_sigma"]) for _ in range(self.depth)])
    def embed(self, x):
        if x.ndim != 3 or x.shape[1:] != (6, self.cfg["window"]):
            raise ValueError("expected [batch, 6, window]")
        return self.embedding(x).transpose(1, 2)
    def _predict(self, head, h):
        mean, log = head(h)
        correction = getattr(self, "output_residual", None)
        if correction is not None:
            mean = mean + self.output_gate * correction(h)
        return mean, log
    def forward(self, x):
        h = self.embed(x); means, logs = [], []
        for block, head in zip(self.blocks, self.heads):
            h = block(h); mean, log = self._predict(head, h); means.append(mean); logs.append(log)
        return torch.stack(means, 1), torch.stack(logs, 1)
    def final(self, x):
        h = self.embed(x)
        for block in self.blocks: h = block(h)
        return self._predict(self.heads[-1], h)
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
            h = block(h); mean, log = self._predict(head, h); sigma = log.exp() * scales[i]
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


class SpectralResidual(nn.Module):
    """Signed Fourier/graph/Haar residual aligned to MEIO patch tokens."""
    def __init__(self, dim, tokens, domains=None):
        super().__init__()
        self.tokens = tokens
        domains = [str(v).lower() for v in (domains or ["fourier", "graph", "wavelet"])]
        allowed = {"fourier", "graph", "wavelet"}
        if not set(domains).issubset(allowed) or not domains:
            raise ValueError("spectral_domains must be a non-empty subset of fourier, graph, wavelet")
        self.domains = tuple(domains)
        self.freq = nn.Sequential(nn.Conv1d(12, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))
        self.graph = nn.Sequential(nn.Conv1d(6, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))
        self.wave = nn.Sequential(nn.Conv1d(6, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))
        self.fuse = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim))

    @staticmethod
    def _haar(x):
        even, odd = x[..., 0::2], x[..., 1::2]
        return (even + odd) * 0.70710678, (even - odd) * 0.70710678

    def _pool(self, x):
        return nn.functional.adaptive_avg_pool1d(x, self.tokens).transpose(1, 2)

    def forward(self, x):
        spec = torch.fft.rfft(x, dim=-1)
        freq = self._pool(self.freq(torch.cat((spec.real, spec.imag), dim=1)))

        acc, gyr = x[:, :3], x[:, 3:]
        graph_input = torch.cat(((acc + gyr) * 0.5, acc - gyr), dim=1)
        graph = self._pool(self.graph(graph_input))

        low, high = self._haar(x)
        wave = self._pool(self.wave(torch.cat((low, high), dim=-1)))
        # Keep one fixed-size fusion layer for fair ablations; disabled domains
        # contribute an exact zero tensor while parameter count stays constant.
        parts = [freq if "fourier" in self.domains else torch.zeros_like(freq),
                 graph if "graph" in self.domains else torch.zeros_like(graph),
                 wave if "wavelet" in self.domains else torch.zeros_like(wave)]
        return self.fuse(torch.cat(parts, dim=-1))


class CausalPatchResidual(nn.Module):
    """Causal local patch mixer; the left padding prevents future-token access."""
    def __init__(self, dim):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, bias=False)
        self.pointwise = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        # tokens: [batch, patch_tokens, dim], and padding is only on the left.
        z = tokens.transpose(1, 2)
        z = nn.functional.pad(z, (2, 0))
        z = self.pointwise(self.depthwise(z)).transpose(1, 2)
        return self.norm(z)


class DiagonalSSMResidual(nn.Module):
    """Small causal diagonal state-space mixer over MEIO patch tokens."""
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.logit_decay = nn.Parameter(torch.full((dim,), 1.5))

    def forward(self, tokens):
        value, gate = self.in_proj(tokens).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(tokens.dtype)
        state = torch.zeros_like(value[:, 0])
        outputs = []
        for step in range(tokens.shape[1]):
            state = decay * state + (1.0 - decay) * value[:, step]
            outputs.append(state * torch.sigmoid(gate[:, step]))
        return self.norm(self.out_proj(torch.stack(outputs, dim=1)))


class SampleSSMResidual(nn.Module):
    """Causal SSM over the 200 raw IMU samples before patch pooling."""
    def __init__(self, dim, tokens):
        super().__init__()
        self.tokens = tokens
        self.in_proj = nn.Linear(6, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.logit_decay = nn.Parameter(torch.full((dim,), 1.5))

    def forward(self, x):
        # x: [batch, 6, samples]. The recurrence is one-way in sample time.
        value, gate = self.in_proj(x.transpose(1, 2)).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(x.dtype)
        state = torch.zeros_like(value[:, 0])
        outputs = []
        for step in range(value.shape[1]):
            state = decay * state + (1.0 - decay) * value[:, step]
            outputs.append(state * torch.sigmoid(gate[:, step]))
        z = self.norm(self.out_proj(torch.stack(outputs, dim=1))).transpose(1, 2)
        return nn.functional.adaptive_avg_pool1d(z, self.tokens).transpose(1, 2)


class OutputResidual(nn.Module):
    """Small displacement correction from the current causal token state."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens):
        return self.net(tokens.mean(1))


class MEIOSpectral(MEIO):
    """MEIO with a zero-initialized causal three-domain token residual."""
    def __init__(self, cfg):
        super().__init__(cfg)
        tokens = cfg["window"] // cfg["patch"]
        self.spectral_residual = SpectralResidual(cfg["dim"], tokens, cfg.get("spectral_domains"))
        # At initialization this is exactly the frozen MEIO representation.
        self.spectral_gate = nn.Parameter(torch.zeros(1, 1, cfg["dim"]))
        if cfg.get("ssm_mixer", False):
            self.temporal_mixer = DiagonalSSMResidual(cfg["dim"])
        elif cfg.get("temporal_mixer", False):
            self.temporal_mixer = CausalPatchResidual(cfg["dim"])
        else:
            self.temporal_mixer = None
        self.temporal_gate = nn.Parameter(torch.zeros(1, 1, cfg["dim"])) if self.temporal_mixer is not None else None
        self.output_residual = OutputResidual(cfg["dim"]) if cfg.get("output_correction", False) else None
        self.output_gate = nn.Parameter(torch.zeros(1, 3)) if self.output_residual is not None else None
        self.sample_ssm = SampleSSMResidual(cfg["dim"], tokens) if cfg.get("sample_ssm", False) else None
        self.sample_ssm_gate = nn.Parameter(torch.zeros(1, 1, cfg["dim"])) if self.sample_ssm is not None else None

    def embed(self, x):
        base = super().embed(x)
        out = base + self.spectral_gate * self.spectral_residual(x)
        if self.sample_ssm is not None:
            out = out + self.sample_ssm_gate * self.sample_ssm(x)
        if self.temporal_mixer is not None:
            out = out + self.temporal_gate * self.temporal_mixer(base)
        return out

    def operation_costs(self):
        costs = super().operation_costs()
        # Conservative accounting for the three 1x1 projections and FFT.
        d = self.cfg["dim"]; n = self.cfg["window"]
        extra = 2 * (12 + 6 + 6) * d * n + 5 * n * int(torch.log2(torch.tensor(float(n))))
        if self.temporal_mixer is not None:
            extra += 2 * (3 * d + d * d + d * d) * (self.cfg["window"] // self.cfg["patch"])
        costs["final"] += extra
        costs["dynamic"] = [value + extra for value in costs["dynamic"]]
        return costs


def make_model(cfg):
    return MEIOSpectral(cfg) if cfg.get("variant") == "spectral" else MEIO(cfg)

def exit_weights(depth, gamma, device):
    # Adapted from official EarlyExistPDRModule.gen_scale_array.
    w = (torch.arange(1, depth+1, device=device, dtype=torch.float32)/depth).pow(gamma)
    return w / w.sum()

def losses(means, log_sigma, target, weights, nll_fraction, trend_weight=0.0):
    if not 0 <= nll_fraction <= 1: raise ValueError("invalid NLL mixing fraction")
    means, log_sigma, target = means.float(), log_sigma.float(), target.float()
    error = means - target[:, None, :]
    mse = error.square().mean((0, 2))
    # Official log-standard-deviation convention, corrected whole-NLL weighting.
    nll = (.5 * error.square() * torch.exp(-2*log_sigma) + log_sigma).mean((0, 2))
    total = (1-nll_fraction)*(weights*mse).sum() + nll_fraction*(weights*nll).sum()
    if trend_weight:
        if trend_weight < 0 or means.shape[0] < 2:
            raise ValueError('invalid trend loss configuration')
        # BlockBatches keeps each batch locally ordered. Matching adjacent
        # displacement changes reduces systematic drift without using future
        # samples at inference time; the first/last record boundary is only a
        # small transition and is harmless for the screening objective.
        pred_delta = means[1:, -1] - means[:-1, -1]
        target_delta = target[1:] - target[:-1]
        total = total + float(trend_weight) * (pred_delta - target_delta).square().mean()
    return total
