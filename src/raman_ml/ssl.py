"""Self-supervised masked-spectrum pretraining (SMAE-style) for Raman.

Masked-autoencoder pretraining (He et al. 2022; for Raman: SMAE, arXiv:2504.16130)
learns a representation from UNLABELLED spectra by masking random wavenumber
patches and reconstructing them, then the encoder is fine-tuned for the labelled
task. It is the research-flagged lever for closing the last accuracy gap in
low-/shifted-label regimes, and it is a capability no maintained OSS Raman library
ships.

Honest scope: we pretrain on *input spectra only* (no labels), and to avoid any
debate about transductive leakage the runner pretrains on reference+finetune
inputs, never on test labels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SpectralMAE(nn.Module):
    """Conv masked-autoencoder: encoder downsamples, decoder reconstructs.

    ``embed(x)`` returns a global-pooled latent for downstream heads.
    """

    def __init__(self, n_in: int, base: int = 64):
        super().__init__()
        self.n_in = n_in
        self.enc = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=2, padding=7), nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.Conv1d(base, base * 2, 7, stride=2, padding=3),
            nn.BatchNorm1d(base * 2), nn.ReLU(inplace=True),
            nn.Conv1d(base * 2, base * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(base * 4), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(base * 4, base * 2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(base * 2, base, 5, padding=2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(base, 1, 5, padding=2),
        )
        self.embed_dim = base * 4

    def forward(self, x):
        r = self.dec(self.enc(x))
        return F.interpolate(r, size=self.n_in, mode="linear", align_corners=False)

    def feature_map(self, x):
        return self.enc(x)              # (B, C, L') - keeps peak positions

    def embed(self, x):
        return self.enc(x).mean(dim=2)  # global pool (for clustering / quick probes)


def _mask(x, mask_ratio, patch, rng):
    """Zero ``mask_ratio`` of contiguous patches; return (masked_x, mask)."""
    b, _, L = x.shape
    n_patch = L // patch
    n_mask = int(mask_ratio * n_patch)
    m = torch.zeros_like(x)
    for i in range(b):
        idx = rng.choice(n_patch, size=n_mask, replace=False)
        for p in idx:
            m[i, 0, p * patch:(p + 1) * patch] = 1.0
    return x * (1 - m), m


class MAEClassifier:
    """SSL-pretrained classifier: masked-AE pretrain, then fine-tune enc + head."""

    def __init__(self, n_out, base=64, pretrain_epochs=30, finetune_epochs=30,
                 batch_size=256, lr=1e-3, mask_ratio=0.5, patch=20, pool_out=8,
                 dropout=0.3, seed=0):
        self.n_out = n_out
        self.base = base
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.patch = patch
        self.pool_out = pool_out          # keep spatial detail in the clf head
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mae = None
        self.head = None

    def _build_head(self):
        # adaptive-pooled spatial head (NOT global mean) so peak positions survive
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(self.pool_out), nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mae.embed_dim * self.pool_out, self.n_out),
        ).to(self.device)

    def save_encoder(self, path):
        torch.save({"base": self.base, "n_in": self.n_in_,
                    "state_dict": self.mae.state_dict()}, path)

    def load_encoder(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.base = ck["base"]
        self.n_in_ = ck["n_in"]
        self.mae = SpectralMAE(self.n_in_, self.base).to(self.device)
        self.mae.load_state_dict(ck["state_dict"])
        return self

    def pretrain(self, X_unlabeled):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_in_ = X_unlabeled.shape[1]
        self.mae = SpectralMAE(self.n_in_, self.base).to(self.device)
        Xt = torch.tensor(np.asarray(X_unlabeled, np.float32)).unsqueeze(1)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        opt = torch.optim.Adam(self.mae.parameters(), lr=self.lr)
        self.mae.train()
        for _ in range(self.pretrain_epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                xm, m = _mask(xb, self.mask_ratio, self.patch, rng)
                rec = self.mae(xm)
                loss = ((rec - xb) ** 2 * m).sum() / (m.sum() + 1e-8)  # masked MSE
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def fit(self, X, y, epochs=None):
        assert self.mae is not None, "call pretrain() or load_encoder() before fit()"
        epochs = epochs or self.finetune_epochs
        self.head = self._build_head()
        Xt = torch.tensor(np.asarray(X, np.float32)).unsqueeze(1)
        yt = torch.tensor(np.asarray(y, np.int64))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        params = list(self.mae.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr * 0.3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.mae.train()
        self.head.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.head(self.mae.feature_map(xb))
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.mae.eval()
        self.head.eval()
        Xt = torch.tensor(np.asarray(X, np.float32)).unsqueeze(1)
        out = []
        for i in range(0, len(Xt), 1024):
            b = Xt[i:i + 1024].to(self.device)
            logits = self.head(self.mae.feature_map(b))
            out.append(torch.softmax(logits, 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
