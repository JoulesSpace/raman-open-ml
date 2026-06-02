"""Compact 1-D CNNs for Raman spectra, plus thin sklearn-style wrappers.

The CNNs are intentionally small so they train on CPU in a few minutes. The
wrappers expose ``fit`` / ``predict`` (and ``predict`` returns class indices for
the classifier) so the CNN can be compared on equal footing with the sklearn
models in the runner scripts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_global_determinism(seed: int = 0):
    """Seed Python/NumPy/torch and request deterministic cuDNN.

    Note: exact bit-reproducibility on CUDA is not guaranteed for all ops, but
    this removes the obvious sources of run-to-run drift.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _conv_block(c_in, c_out, k=7, pool=2):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool),
    )


class SpectralCNN(nn.Module):
    """1-D CNN backbone shared by the classifier and regressor.

    ``pool_out`` controls how much spatial detail survives into the head: a
    larger value (e.g. 4) keeps peak-position information useful for fine-grained
    classification, while ``1`` (global average pooling) is a strong regulariser
    for the tiny regression dataset.
    """

    def __init__(self, n_in: int, n_out: int, channels=(32, 64, 128),
                 pool_out: int = 4, dropout: float = 0.3):
        super().__init__()
        blocks = []
        c_prev = 1
        for c in channels:
            blocks.append(_conv_block(c_prev, c))
            c_prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_prev * pool_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def forward(self, x):  # x: (B, 1, L)
        return self.head(self.pool(self.features(x)))

    def embed(self, x):  # penultimate features (before final Linear)
        return self.head[:-1](self.pool(self.features(x)))


class _SqueezeExcite1D(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    The SE-ResNet ensemble is the current open-world SOTA on bacteria-ID
    (Lebron et al. 2024, 87.8%); SE is applied on the residual branch.
    """

    def __init__(self, c, r=8):
        super().__init__()
        h = max(1, c // r)
        self.fc1 = nn.Linear(c, h)
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):  # x: (B, C, L)
        s = x.mean(dim=2)                       # squeeze over length
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None]                # excite (rescale channels)


class _ResBlock1D(nn.Module):
    """Basic residual block (two convs + skip), optional SE attention."""

    def __init__(self, c_in, c_out, stride=1, k=9, dropout=0.0, se=False):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.se = _SqueezeExcite1D(c_out) if se else None
        self.short = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        if self.se is not None:
            out = self.se(out)
        return torch.relu(out + self.short(x))


class ResNet1D(nn.Module):
    """ResNet-18-style 1-D CNN for spectra.

    This is the architecture family that set the bar on bacteria-ID (Ho et al.
    2019 used a ~25-layer 1-D ResNet). Defaults give a ResNet-18 over 1000-point
    spectra; depth/width are configurable.
    """

    def __init__(self, n_in: int, n_out: int, base: int = 64,
                 layers=(2, 2, 2, 2), k: int = 9, dropout: float = 0.1,
                 se: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        chans = [base, base * 2, base * 4, base * 8]
        blocks, c_prev = [], base
        for i, nb in enumerate(layers):
            c = chans[i]
            for j in range(nb):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(_ResBlock1D(c_prev, c, stride=stride, k=k,
                                          dropout=dropout, se=se))
                c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_prev, n_out)

    def forward(self, x):  # x: (B, 1, L)
        return self.fc(self.embed(x))

    def embed(self, x):  # penultimate features (before final Linear)
        x = self.blocks(self.stem(x))
        return self.pool(x).flatten(1)


class _MultiScaleBlock(nn.Module):
    """Parallel multi-kernel conv branches (SANet-style), concatenated.

    The 2026 cross-dataset benchmark (arXiv:2601.16107) found multi-scale CNNs
    (SANet) the strongest on bacteria-ID, beating plain ResNet and transformers,
    because parallel receptive fields capture both narrow Raman peaks and broad
    envelopes. Each block runs kernels of several widths in parallel.
    """

    def __init__(self, c_in, c_out, stride=1, kernels=(3, 5, 7, 11),
                 se=False, dropout=0.0):
        super().__init__()
        assert c_out % len(kernels) == 0, "c_out must divide by #kernels"
        cb = c_out // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(c_in, cb, k, stride=stride, padding=k // 2, bias=False)
            for k in kernels])
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.se = _SqueezeExcite1D(c_out) if se else None
        self.short = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.drop(torch.relu(self.bn(out)))
        if self.se is not None:
            out = self.se(out)
        return torch.relu(out + self.short(x))


class MSResNet1D(nn.Module):
    """Multi-scale residual 1-D CNN (SANet-inspired)."""

    def __init__(self, n_in: int, n_out: int, base: int = 64,
                 layers=(2, 2, 2), se: bool = True, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        chans = [base, base * 2, base * 4, base * 8]
        blocks, c_prev = [], base
        for i, nb in enumerate(layers):
            c = chans[i]
            for j in range(nb):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(_MultiScaleBlock(c_prev, c, stride=stride, se=se,
                                               dropout=dropout))
                c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_prev, n_out)

    def forward(self, x):
        return self.fc(self.embed(x))

    def embed(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)


class _BaseCNN:
    """Common training loop for the sklearn-style CNN wrappers."""

    def __init__(self, n_out, epochs=20, batch_size=128, lr=1e-3,
                 weight_decay=1e-4, seed=0, verbose=False,
                 channels=(32, 64, 128), pool_out=4, dropout=0.3,
                 arch="cnn", resnet_base=64, resnet_layers=(2, 2, 2, 2),
                 se=False, augment=None):
        self.n_out = n_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose
        self.channels = channels
        self.pool_out = pool_out
        self.dropout = dropout
        self.arch = arch
        self.resnet_base = resnet_base
        self.resnet_layers = resnet_layers
        self.se = se
        self.augment = augment  # callable on a numpy batch (B, L), or None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _batch_transform(self, xb, yb):
        """Optional per-batch transform on GPU tensors (override in subclasses)."""
        return xb, yb

    def _build_model(self, n_in):
        if self.arch == "msresnet":
            return MSResNet1D(n_in, self.n_out, base=self.resnet_base,
                              layers=self.resnet_layers, se=self.se,
                              dropout=self.dropout)
        if self.arch == "resnet":
            return ResNet1D(n_in, self.n_out, base=self.resnet_base,
                            layers=self.resnet_layers, dropout=self.dropout,
                            se=self.se)
        return SpectralCNN(n_in, self.n_out, channels=self.channels,
                           pool_out=self.pool_out, dropout=self.dropout)

    def _make_loader(self, X, y, shuffle):
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        yt = self._target_tensor(y)
        gen = torch.Generator().manual_seed(self.seed)   # reproducible shuffling
        return DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size,
                          shuffle=shuffle, generator=gen)

    def _run_epochs(self, loader, opt, epochs, sched=None):
        loss_fn = self._loss_fn()
        self.model.train()
        for ep in range(epochs):
            running = 0.0
            for xb, yb in loader:
                if self.augment is not None:
                    arr = self.augment(xb.squeeze(1).numpy())
                    xb = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(1)
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb, yb = self._batch_transform(xb, yb)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
            if sched is not None:
                sched.step()
            if self.verbose:
                print(f"    epoch {ep + 1:>3}/{epochs}  "
                      f"loss={running / len(loader.dataset):.4f}")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.n_in_ = X.shape[1]
        self.model = self._build_model(self.n_in_).to(self.device)
        loader = self._make_loader(X, y, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, self.epochs // 3), gamma=0.5)
        self._run_epochs(loader, opt, self.epochs, sched)
        return self

    def save(self, path):
        """Persist the trained model (weights + the config needed to rebuild)."""
        torch.save({
            "cls": type(self).__name__, "n_in": self.n_in_, "n_out": self.n_out,
            "arch": self.arch, "channels": self.channels,
            "pool_out": self.pool_out, "dropout": self.dropout,
            "resnet_base": self.resnet_base, "resnet_layers": self.resnet_layers,
            "se": self.se, "y_mean": getattr(self, "_y_mean", 0.0),
            "y_std": getattr(self, "_y_std", 1.0),
            "state_dict": self.model.state_dict(),
        }, path)
        return path

    def finetune(self, X, y, epochs=10, lr=None):
        """Continue training a fitted model on new (target-domain) data.

        Transfer learning: pretrain with ``fit`` on a large source set, then
        ``finetune`` on a smaller target-domain set at a lower learning rate.
        This is the bacteria-ID protocol (Ho et al. 2019) that closes the
        reference->test campaign-shift gap.
        """
        assert self.model is not None, "call fit() before finetune()"
        lr = lr if lr is not None else self.lr * 0.1
        loader = self._make_loader(X, y, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr,
                               weight_decay=self.weight_decay)
        self._run_epochs(loader, opt, epochs)
        return self

    @torch.no_grad()
    def _raw_predict(self, X):
        self.model.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        outs = []
        for i in range(0, len(Xt), 1024):
            batch = Xt[i:i + 1024].to(self.device)
            outs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def embed(self, X):
        """Penultimate-layer features (for Mahalanobis OOD, clustering, etc.)."""
        self.model.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        outs = []
        for i in range(0, len(Xt), 1024):
            batch = Xt[i:i + 1024].to(self.device)
            outs.append(self.model.embed(batch).cpu().numpy())
        return np.concatenate(outs)

    def logits(self, X):
        return self._raw_predict(X)


class CNNClassifier(_BaseCNN):
    def __init__(self, *args, label_smoothing=0.0, **kwargs):
        self.label_smoothing = label_smoothing
        super().__init__(*args, **kwargs)

    def _target_tensor(self, y):
        return torch.tensor(np.asarray(y, dtype=np.int64))

    def _loss_fn(self):
        return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    @torch.no_grad()
    def mc_dropout_proba(self, X, n_samples=30):
        """Monte-Carlo dropout: keep dropout on at inference for an uncertainty
        estimate. Returns mean probs and per-sample predictive std (n, C)."""
        self.model.train()  # enable dropout
        probs = []
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        for _ in range(n_samples):
            outs = []
            for i in range(0, len(Xt), 1024):
                logits = self.model(Xt[i:i + 1024].to(self.device))
                outs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs.append(np.concatenate(outs))
        probs = np.stack(probs)
        self.model.eval()
        return probs.mean(0), probs.std(0)

    def predict(self, X):
        return self._raw_predict(X).argmax(axis=1)

    def predict_proba(self, X):
        logits = torch.tensor(self._raw_predict(X))
        return torch.softmax(logits, dim=1).numpy()


class CNNRegressor(_BaseCNN):
    """Regressor with internal target standardisation.

    Raman concentration targets (e.g. log10 particles/mL ~ 9..15) sit far from
    a freshly-initialised network's near-zero outputs, which makes MSE training
    diverge. We z-score the target during fit and invert it at predict time.
    """

    def __init__(self, *args, cmixup=False, cmix_sigma=0.5, cmix_alpha=2.0,
                 **kwargs):
        kwargs.setdefault("n_out", 1)
        # Compact + heavily pooled: the quantification set is tiny (48 spectra).
        kwargs.setdefault("channels", (16, 32, 64))
        kwargs.setdefault("pool_out", 1)
        kwargs.setdefault("dropout", 0.4)
        super().__init__(*args, **kwargs)
        self.cmixup = cmixup
        self.cmix_sigma = cmix_sigma
        self.cmix_alpha = cmix_alpha
        self._y_mean = 0.0
        self._y_std = 1.0

    def _batch_transform(self, xb, yb):
        """C-Mixup (Yao et al. 2022): mix pairs weighted by label proximity.

        The partner is sampled with prob proportional to a Gaussian kernel over
        target distance, so only spectra of similar concentration are blended -
        the right mixup for a dilution-series regression.
        """
        if not self.cmixup:
            return xb, yb
        with torch.no_grad():
            y = yb.squeeze(1)
            d2 = (y[:, None] - y[None, :]) ** 2
            w = torch.softmax(-d2 / (2 * self.cmix_sigma ** 2), dim=1)
            idx = torch.multinomial(w, 1).squeeze(1)
        lam = float(np.random.beta(self.cmix_alpha, self.cmix_alpha))
        return lam * xb + (1 - lam) * xb[idx], lam * yb + (1 - lam) * yb[idx]

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        return super().fit(X, y)

    def _target_tensor(self, y):
        y = (np.asarray(y, dtype=np.float32) - self._y_mean) / self._y_std
        return torch.tensor(y).unsqueeze(1)

    def _loss_fn(self):
        return nn.MSELoss()

    def predict(self, X):
        return self._raw_predict(X).ravel() * self._y_std + self._y_mean


class DeepEnsemble:
    """Ensemble of M independently-seeded models (Lakshminarayanan et al. 2017).

    The most reliable practical uncertainty estimate for neural nets: the mean is
    the prediction, the spread across members is epistemic uncertainty, and the
    averaging also lifts point accuracy. Wrap any of the CNN wrappers above by
    passing a zero-arg ``factory`` that returns a fresh estimator.
    """

    def __init__(self, factory, n_members=5, seed=0):
        self.factory = factory
        self.n_members = n_members
        self.seed = seed
        self.members = []

    def fit(self, X, y):
        self.members = []
        for i in range(self.n_members):
            m = self.factory()
            if hasattr(m, "seed"):
                m.seed = self.seed + i
            m.fit(X, y)
            self.members.append(m)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X):
        if hasattr(self.members[0], "predict_proba"):
            return self.predict_proba(X).argmax(1)
        preds = np.stack([m.predict(X) for m in self.members])
        return preds.mean(0)

    def predict_with_uncertainty(self, X):
        """Regression: returns (mean, std) across ensemble members."""
        preds = np.stack([m.predict(X) for m in self.members])
        return preds.mean(0), preds.std(0)


def load_cnn(path, map_location="cpu"):
    """Load a model saved by ``_BaseCNN.save`` and return a ready wrapper."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cls = CNNRegressor if ckpt["cls"] == "CNNRegressor" else CNNClassifier
    w = cls(n_out=ckpt["n_out"], arch=ckpt["arch"], channels=ckpt["channels"],
            pool_out=ckpt["pool_out"], dropout=ckpt["dropout"],
            resnet_base=ckpt["resnet_base"], resnet_layers=ckpt["resnet_layers"],
            se=ckpt["se"])
    w.n_in_ = ckpt["n_in"]
    w.device = torch.device(map_location)
    w.model = w._build_model(ckpt["n_in"]).to(w.device)
    w.model.load_state_dict(ckpt["state_dict"])
    w.model.eval()
    if ckpt["cls"] == "CNNRegressor":
        w._y_mean, w._y_std = ckpt["y_mean"], ckpt["y_std"]
    return w
