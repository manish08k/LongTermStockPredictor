"""
src/deep_learning/transformer.py – Transformer-based time-series model.

Architecture:
  - Positional encoding (sinusoidal)
  - Multi-head self-attention encoder stack
  - Global average pooling → binary classification head
  - PyTorch implementation with GPU support
  - Outperforms LSTM on longer sequences (≥60 days)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils import get_logger
from src.deep_learning.lstm import build_sequences

log = get_logger(__name__)


# ── Positional Encoding ────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module if HAS_TORCH else object):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):  # type: ignore[override]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Transformer Network ────────────────────────────────────────────────────

class _TransformerNet(nn.Module if HAS_TORCH else object):
    """
    Encoder-only Transformer for time-series classification.

    Input : (B, T, input_size)
    Output: (B,) probability of positive class
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        # Project raw features → d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm     = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)           # (B, T, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)             # global average pool over time
        return self.head(x).squeeze(-1)


# ── Public Model class ─────────────────────────────────────────────────────

class TransformerModel:
    """
    Transformer-based binary classifier for stock direction prediction.

    Compatible with the BaseModel interface (fit / predict_proba).
    """

    name = "transformer"

    def __init__(self, cfg: dict) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch required. pip install torch")

        dl_cfg = cfg.get("deep_learning", {})
        self.seq_len:        int   = dl_cfg.get("sequence_length", 60)
        self.d_model:        int   = dl_cfg.get("d_model", 64)
        self.nhead:          int   = dl_cfg.get("nhead", 4)
        self.n_enc_layers:   int   = dl_cfg.get("num_encoder_layers", 3)
        self.dim_ff:         int   = dl_cfg.get("dim_feedforward", 256)
        self.dropout:        float = dl_cfg.get("dropout", 0.1)
        self.batch_size:     int   = dl_cfg.get("batch_size", 64)
        self.epochs:         int   = dl_cfg.get("epochs", 50)
        self.lr:             float = dl_cfg.get("learning_rate", 0.0005)
        self.patience:       int   = dl_cfg.get("early_stopping_patience", 10)
        self.warmup_steps:   int   = dl_cfg.get("warmup_steps", 100)
        self.ckpt_dir:       Path  = Path(dl_cfg.get("checkpoint_dir", "models/dl_checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # d_model must be divisible by nhead
        if self.d_model % self.nhead != 0:
            self.d_model = (self.d_model // self.nhead) * self.nhead
            log.warning(f"[Transformer] Adjusted d_model to {self.d_model} (divisible by nhead={self.nhead})")

        use_gpu = dl_cfg.get("use_gpu", True)
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        log.info(
            f"[Transformer] device={self.device} | d_model={self.d_model} | "
            f"nhead={self.nhead} | layers={self.n_enc_layers}"
        )

        self._net: Optional[_TransformerNet] = None
        self._feature_names: List[str] = []
        self._input_size: int = 0

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Transformer on chronological sequential windows."""
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        X_seq, y_seq = build_sequences(X_arr, y_arr, self.seq_len)
        n = len(X_seq)
        self._input_size = X_seq.shape[2]

        split = int(0.8 * n)
        tr_loader  = self._make_loader(X_seq[:split], y_seq[:split], shuffle=True)
        val_loader = self._make_loader(X_seq[split:], y_seq[split:], shuffle=False)

        self._net = _TransformerNet(
            input_size=self._input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.n_enc_layers,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout,
            max_seq_len=self.seq_len + 10,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=1e-3
        )
        # Cosine schedule with warm restarts
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.epochs * len(tr_loader),
            pct_start=0.1,
        )
        criterion  = nn.BCELoss()
        ckpt_path  = self.ckpt_dir / "transformer_best.pt"
        best_val   = math.inf
        patience_c = 0

        log.info(
            f"[Transformer] Training: {n} sequences | input={self._input_size} | "
            f"params={sum(p.numel() for p in self._net.parameters()):,}"
        )

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            tr_loss = self._run_epoch(tr_loader, criterion, optimizer, scheduler)

            self._net.eval()
            val_loss = self._run_epoch(val_loader, criterion)

            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    f"[Transformer] Epoch {epoch:3d}/{self.epochs} | "
                    f"train={tr_loss:.4f} | val={val_loss:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                patience_c = 0
                torch.save(self._net.state_dict(), ckpt_path)
            else:
                patience_c += 1
                if patience_c >= self.patience:
                    log.info(f"[Transformer] Early stopping at epoch {epoch}")
                    break

        if ckpt_path.exists():
            self._net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        log.info(f"[Transformer] Training complete. Best val_loss={best_val:.4f}")

    # ── predict_proba ──────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each sample. First seq_len rows are padded with 0.5."""
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_arr = X[self._feature_names].values.astype(np.float32)
        if len(X_arr) <= self.seq_len:
            return np.full(len(X_arr), 0.5, dtype=np.float32)

        X_seq, _ = build_sequences(X_arr, np.zeros(len(X_arr)), self.seq_len)
        loader = self._make_loader(X_seq, np.zeros(len(X_seq)), shuffle=False)

        self._net.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self._net(xb.to(self.device)).cpu().numpy())

        seq_probs = np.concatenate(preds)
        padded = np.full(len(X_arr), 0.5, dtype=np.float32)
        padded[self.seq_len:] = seq_probs
        return padded

    def feature_importances(self) -> pd.Series:
        """Transformers don't expose direct feature importances. Returns zeros."""
        return pd.Series(np.zeros(len(self._feature_names)), index=self._feature_names)

    # ── save / load ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import joblib
        state = {
            "state_dict":    self._net.state_dict() if self._net else None,
            "feature_names": self._feature_names,
            "input_size":    self._input_size,
            "d_model":       self.d_model,
            "nhead":         self.nhead,
            "n_enc_layers":  self.n_enc_layers,
            "dim_ff":        self.dim_ff,
            "dropout":       self.dropout,
            "seq_len":       self.seq_len,
        }
        joblib.dump(state, path)
        log.info(f"[Transformer] Saved → {path}")

    def load(self, path: Path) -> None:
        import joblib
        state = joblib.load(path)
        self._feature_names = state["feature_names"]
        self._input_size    = state["input_size"]
        self._net = _TransformerNet(
            input_size=state["input_size"],
            d_model=state["d_model"],
            nhead=state["nhead"],
            num_encoder_layers=state["n_enc_layers"],
            dim_feedforward=state["dim_ff"],
            dropout=state["dropout"],
            max_seq_len=state["seq_len"] + 10,
        ).to(self.device)
        if state["state_dict"]:
            self._net.load_state_dict(state["state_dict"])
        log.info(f"[Transformer] Loaded from {path}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = False
    ) -> "DataLoader":
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
        optimizer: Optional["torch.optim.Optimizer"] = None,
        scheduler=None,
    ) -> float:
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self._net(xb)
            loss  = criterion(preds, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)
