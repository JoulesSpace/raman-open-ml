"""Generative augmentation for 1-D Raman spectra: a conditional diffusion model.

The classic augmentations in :mod:`raman_ml.augment` (noise, shift, warp, mixup)
only perturb existing spectra. A generative model instead learns the *data
distribution* and samples genuinely new spectra, which is the modern approach to
augmenting tiny spectral datasets.

We ship **two** purpose-built 1-D conditional generators and let the benchmark
(`scripts/run_generative_augmentation.py`) decide which is better rather than
asserting it up front:

  * :class:`SpectralDiffusion` - a class-conditional DDPM (Ho et al. 2020) with a
    fixed-resolution 1-D conv residual denoiser, sinusoidal timestep embedding and
    an additive class embedding (FiLM-style conditioning);
  * :class:`SpectralGAN` - a class-conditional 1-D WGAN-GP (Gulrajani et al. 2017).

Plus lightweight statistical baselines (random resampling, a PCA-Gaussian copula).
On the few-shot bacteria-ID benchmark the WGAN-GP actually edged out the DDPM
(the DDPM is data-hungry and undertrains on a few hundred spectra), and neither
beat cheap classical augmentation - reported honestly in the README.

    diff = SpectralDiffusion(n_classes=30, epochs=300).fit(X, y)
    X_syn, y_syn = diff.generate_balanced(per_class=40)

Both models standardise per-wavenumber to ~N(0, 1) internally and invert that on
sampling, so callers pass and receive spectra in their own scale.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class _ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv residual block with an injected (time + class) bias."""

    def __init__(self, c: int, emb_dim: int, k: int = 7):
        super().__init__()
        g = min(8, c)
        self.norm1 = nn.GroupNorm(g, c)
        self.conv1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.emb = nn.Linear(emb_dim, c)
        self.norm2 = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class _DiffNet(nn.Module):
    """epsilon-prediction network: predict the noise added at timestep t."""

    def __init__(self, base: int = 64, n_blocks: int = 4, n_classes: int | None = None,
                 emb_dim: int = 128):
        super().__init__()
        self.time = nn.Sequential(
            _SinusoidalPosEmb(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))
        self.label = nn.Embedding(n_classes, emb_dim) if n_classes else None
        self.in_conv = nn.Conv1d(1, base, 7, padding=3)
        self.blocks = nn.ModuleList([_ResBlock(base, emb_dim) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(min(8, base), base)
        self.out_conv = nn.Conv1d(base, 1, 7, padding=3)
        self.act = nn.SiLU()

    def forward(self, x, t, y=None):
        emb = self.time(t)
        if self.label is not None and y is not None:
            emb = emb + self.label(y)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out_conv(self.act(self.out_norm(h)))


class SpectralDiffusion:
    """Class-conditional DDPM over 1-D spectra (sklearn-ish fit / sample API)."""

    def __init__(self, n_classes: int | None = None, timesteps: int = 200,
                 base: int = 64, n_blocks: int = 4, epochs: int = 300,
                 batch_size: int = 128, lr: float = 2e-4, seed: int = 0,
                 verbose: bool = False):
        self.n_classes = n_classes
        self.timesteps = timesteps
        self.base = base
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.mu_ = self.sd_ = None
        self.length = None

    def _schedule(self):
        betas = torch.linspace(1e-4, 0.02, self.timesteps, device=self.device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        return betas, alphas, abar

    def fit(self, X, y=None):
        _seed_everything(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.length = X.shape[1]
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-6
        Xn = (X - self.mu_) / self.sd_

        Xt = torch.tensor(Xn, device=self.device).unsqueeze(1)
        yt = (torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)
              if (self.n_classes and y is not None) else None)
        self.betas, self.alphas, self.abar = self._schedule()

        self.net = _DiffNet(self.base, self.n_blocks, self.n_classes).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        n = Xt.shape[0]
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=gen, device=self.device)
            running = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x0 = Xt[idx]
                yb = yt[idx] if yt is not None else None
                t = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device,
                                  generator=gen)
                eps = torch.randn(x0.shape, device=self.device, generator=gen)
                ab = self.abar[t][:, None, None]
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                pred = self.net(xt, t, yb)
                loss = F.mse_loss(pred, eps)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * x0.shape[0]
            self.final_loss_ = running / n
            if self.verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                print(f"  diffusion epoch {ep:4d}  loss={self.final_loss_:.4f}")
        return self

    @torch.no_grad()
    def sample(self, n: int, y=None, batch: int = 256):
        """Generate ``n`` spectra (optionally conditioned on class labels ``y``)."""
        self.net.eval()
        out = []
        labels = None if y is None else torch.tensor(np.asarray(y), dtype=torch.long,
                                                      device=self.device)
        done = 0
        while done < n:
            b = min(batch, n - done)
            x = torch.randn(b, 1, self.length, device=self.device)
            yb = labels[done:done + b] if labels is not None else None
            for t in reversed(range(self.timesteps)):
                tb = torch.full((b,), t, device=self.device, dtype=torch.long)
                eps = self.net(x, tb, yb)
                alpha_t, abar_t, beta_t = self.alphas[t], self.abar[t], self.betas[t]
                mean = (x - beta_t / (1 - abar_t).sqrt() * eps) / alpha_t.sqrt()
                x = mean + (beta_t.sqrt() * torch.randn_like(x) if t > 0 else 0.0)
            out.append(x.squeeze(1).cpu().numpy())
            done += b
        Xn = np.concatenate(out, axis=0)
        return Xn * self.sd_ + self.mu_

    def generate_balanced(self, per_class: int):
        """Sample ``per_class`` spectra for every class. Returns (X_syn, y_syn)."""
        if not self.n_classes:
            raise ValueError("generate_balanced needs a class-conditional model")
        y = np.repeat(np.arange(self.n_classes), per_class)
        X = self.sample(len(y), y=y)
        return X, y


# --------------------------------------------------------------------------- #
# Conditional 1-D WGAN-GP - the adversarial alternative, for a fair comparison #
# --------------------------------------------------------------------------- #
class _Generator(nn.Module):
    def __init__(self, length, n_classes, nz=64, base=64):
        super().__init__()
        self.length = length
        self.L0 = (length + 7) // 8
        self.base = base
        self.fc = nn.Linear(nz + n_classes, base * self.L0)
        self.net = nn.Sequential(
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, base, 4, stride=2, padding=1),
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, base // 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(base // 2), nn.ReLU(),
            nn.ConvTranspose1d(base // 2, 1, 4, stride=2, padding=1))

    def forward(self, z, onehot):
        h = self.fc(torch.cat([z, onehot], dim=1)).view(-1, self.base, self.L0)
        return self.net(h)[..., :self.length]


class _Critic(nn.Module):
    def __init__(self, length, n_classes, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1 + n_classes, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(base, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(base, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.fc = nn.Linear(base * ((length + 7) // 8), 1)

    def forward(self, x, onehot):
        lab = onehot[:, :, None].expand(-1, -1, x.shape[-1])
        h = self.net(torch.cat([x, lab], dim=1)).flatten(1)
        return self.fc(h)


class SpectralGAN:
    """Class-conditional 1-D WGAN-GP generator of spectra (same API as the DDPM).

    The adversarial counterpart to :class:`SpectralDiffusion`, for a head-to-head
    comparison. WGAN-GP (Gulrajani et al. 2017) is the more stable GAN variant;
    empirically on the few-shot benchmark here it outperformed the DDPM, so the
    common "diffusion always beats GAN" intuition does not hold on this small,
    smooth-signal problem.
    """

    def __init__(self, n_classes: int, nz: int = 64, base: int = 64,
                 epochs: int = 300, batch_size: int = 128, lr: float = 1e-4,
                 n_critic: int = 5, gp_lambda: float = 10.0, seed: int = 0,
                 verbose: bool = False):
        self.n_classes = n_classes
        self.nz = nz
        self.base = base
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = self.D = None
        self.mu_ = self.sd_ = None
        self.length = None

    def _onehot(self, y):
        oh = torch.zeros(len(y), self.n_classes, device=self.device)
        oh[torch.arange(len(y)), y] = 1.0
        return oh

    def _grad_penalty(self, real, fake, oh):
        eps = torch.rand(real.shape[0], 1, 1, device=self.device)
        xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d = self.D(xhat, oh)
        g = torch.autograd.grad(d.sum(), xhat, create_graph=True)[0]
        return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X, y):
        _seed_everything(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.length = X.shape[1]
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu_) / self.sd_, device=self.device).unsqueeze(1)
        yt = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)

        self.G = _Generator(self.length, self.n_classes, self.nz, self.base).to(self.device)
        self.D = _Critic(self.length, self.n_classes, self.base).to(self.device)
        og = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        od = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.0, 0.9))
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        n = Xt.shape[0]
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=gen, device=self.device)
            d_running = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                real, oh = Xt[idx], self._onehot(yt[idx])
                # critic steps
                for _ in range(self.n_critic):
                    z = torch.randn(real.shape[0], self.nz, device=self.device, generator=gen)
                    fake = self.G(z, oh).detach()
                    d_loss = (self.D(fake, oh).mean() - self.D(real, oh).mean()
                              + self.gp_lambda * self._grad_penalty(real, fake, oh))
                    od.zero_grad()
                    d_loss.backward()
                    od.step()
                    d_running += d_loss.item()
                # generator step
                z = torch.randn(real.shape[0], self.nz, device=self.device, generator=gen)
                g_loss = -self.D(self.G(z, oh), oh).mean()
                og.zero_grad()
                g_loss.backward()
                og.step()
            self.final_critic_loss_ = d_running / max(1, n)
            if self.verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                print(f"  gan epoch {ep:4d}  critic_loss={self.final_critic_loss_:.3f}")
        return self

    @torch.no_grad()
    def sample(self, n: int, y, batch: int = 512):
        self.G.eval()
        labels = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)
        out, done = [], 0
        while done < n:
            b = min(batch, n - done)
            z = torch.randn(b, self.nz, device=self.device)
            oh = self._onehot(labels[done:done + b])
            out.append(self.G(z, oh).squeeze(1).cpu().numpy())
            done += b
        return np.concatenate(out, axis=0) * self.sd_ + self.mu_

    def generate_balanced(self, per_class: int):
        y = np.repeat(np.arange(self.n_classes), per_class)
        return self.sample(len(y), y=y), y


# --------------------------------------------------------------------------- #
# Lightweight statistical baselines (the families tabular generators cover)    #
# --------------------------------------------------------------------------- #
def random_resample_balanced(X, y, per_class: int, jitter: float = 0.0, seed: int = 0):
    """Naive baseline: draw existing rows with replacement (optionally jittered).

    This is the honest control - it adds no new information, only copies - so any
    method that fails to beat it is not learning the data distribution.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    Xs, ys = [], []
    for c in np.unique(y):
        rows = X[y == c]
        pick = rows[rng.integers(0, len(rows), size=per_class)]
        if jitter:
            pick = pick + rng.standard_normal(pick.shape).astype(np.float32) * jitter
        Xs.append(pick)
        ys.append(np.full(per_class, c))
    return np.concatenate(Xs), np.concatenate(ys)


class PCAGaussian:
    """Per-class Gaussian fitted in PCA space (a stable copula-style baseline).

    A full-resolution Gaussian/copula over ~1000 correlated channels is rank
    deficient from a handful of spectra, so we model a low-dimensional PCA
    representation with a Ledoit-Wolf-shrunk covariance per class and invert the
    PCA on sampling. This is the tractable form of the Bayesian / Gaussian-copula
    family for spectra.
    """

    def __init__(self, n_classes: int, n_components: int = 30, seed: int = 0):
        self.n_classes = n_classes
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y):
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        nc = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=nc, random_state=self.seed).fit(X)
        Z = self.pca.transform(X)
        self.stats = {}
        for c in np.unique(y):
            zc = Z[y == c]
            cov = LedoitWolf().fit(zc).covariance_ if len(zc) > 1 else np.eye(nc) * 1e-3
            self.stats[int(c)] = (zc.mean(0), cov)
        return self

    def generate_balanced(self, per_class: int):
        rng = np.random.default_rng(self.seed)
        Xs, ys = [], []
        for c, (mean, cov) in self.stats.items():
            z = rng.multivariate_normal(mean, cov, size=per_class)
            Xs.append(self.pca.inverse_transform(z).astype(np.float32))
            ys.append(np.full(per_class, c))
        return np.concatenate(Xs), np.concatenate(ys)
