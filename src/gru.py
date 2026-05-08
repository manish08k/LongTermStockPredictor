"""
src/deep_learning/gru.py – GRU model for time-series stock prediction.

Architecture:
  - Multi-layer GRU (lighter than LSTM, fewer parameters)
  - Binary classification head
  - Bidirectional option for richer temporal context
  - PyTorch implementation with GPU support
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

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
from src.deep_learning.lstm import build_sequences  # reuse sequence builder

log = get_logger(__name__)


# ── PyTorch Module ─────────────────────────────────────────────────────────

class _GRUNet(nn.Module if HAS_TORCH else object):
    """
    Stacked GRU (optionally bidirectional) with a binary classification head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        out, _ = self.gru(x)      # (B, T, H * dirs)
        last = out[:, -1, :]      # last time step
        last = self.norm(last)
        return self.head(last).squeeze(-1)


# ── Public Model class ─────────────────────────────────────────────────────

class GRUModel:
    """
    GRU-based binary classifier for stock direction prediction.

    Compatible with the BaseModel interface (fit / predict_proba).
    Typically faster and more memory-efficient than LSTMModel.
    """

    name = "gru"

    def __init__(self, cfg: dict) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GRUModel. pip install torch")

        dl_cfg = cfg.get("deep_learning", {})
        self.seq_len:       int   = dl_cfg.get("sequence_length", 60)
        self.hidden:        int   = dl_cfg.get("hidden_size", 128)
        self.n_layers:      int   = dl_cfg.get("num_layers", 2)
        self.dropout:       float = dl_cfg.get("dropout", 0.2)
        self.bidirectional: bool  = dl_cfg.get("bidirectional", False)
        self.batch_size:    int   = dl_cfg.get("batch_size", 64)
        self.epochs:        int   = dl_cfg.get("epochs", 50)
        self.lr:            float = dl_cfg.get("learning_rate", 0.001)
        self.patience:      int   = dl_cfg.get("early_stopping_patience", 10)
        self.ckpt_dir:      Path  = Path(dl_cfg.get("checkpoint_dir", "models/dl_checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        use_gpu = dl_cfg.get("use_gpu", True)
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        log.info(
            f"[GRU] device={self.device} | "
            f"bidirectional={self.bidirectional}"
        )

        self._net: Optional[_GRUNet] = None
        self._feature_names: List[str] = []
        self._input_size: int = 0

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train GRU on chronological sequential windows."""
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        X_seq, y_seq = build_sequences(X_arr, y_arr, self.seq_len)
        n = len(X_seq)
        self._input_size = X_seq.shape[2]

        split = int(0.8 * n)
        X_tr,  X_val  = X_seq[:split], X_seq[split:]
        y_tr,  y_val  = y_seq[:split], y_seq[split:]

        tr_loader  = self._make_loader(X_tr,  y_tr,  shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        self._net = _GRUNet(
            input_size=self._input_size,
            hidden_size=self.hidden,
            num_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

        optimizer  = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion  = nn.BCELoss()
        ckpt_path  = self.ckpt_dir / "gru_best.pt"

        best_val   = math.inf
        patience_c = 0

        log.info(
            f"[GRU] Training: {n} sequences | input={self._input_size} | "
            f"hidden={self.hidden} | layers={self.n_layers}"
        )

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            tr_loss = self._run_epoch(tr_loader, criterion, optimizer)

            self._net.eval()
            val_loss = self._run_epoch(val_loader, criterion, optimizer=None)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    f"[GRU] Epoch {epoch:3d}/{self.epochs} | "
                    f"train={tr_loss:.4f} | val={val_loss:.4f}"
                )

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                patience_c = 0
                torch.save(self._net.state_dict(), ckpt_path)
            else:
                patience_c += 1
                if patience_c >= self.patience:
                    log.info(f"[GRU] Early stopping at epoch {epoch}")
                    break

        if ckpt_path.exists():
            self._net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        log.info(f"[GRU] Training complete. Best val_loss={best_val:.4f}")

    # ── predict_proba ──────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each sample. Pads first seq_len rows with 0.5."""
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
                xb = xb.to(self.device)
                preds.append(self._net(xb).cpu().numpy())

        seq_probs = np.concatenate(preds)
        padded = np.full(len(X_arr), 0.5, dtype=np.float32)
        padded[self.seq_len:] = seq_probs
        return padded

    def feature_importances(self) -> pd.Series:
        return pd.Series(np.zeros(len(self._feature_names)), index=self._feature_names)

    # ── save / load ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import joblib
        state = {
            "state_dict":   self._net.state_dict() if self._net else None,
            "feature_names": self._feature_names,
            "input_size":   self._input_size,
            "hidden":       self.hidden,
            "n_layers":     self.n_layers,
            "dropout":      self.dropout,
            "bidirectional": self.bidirectional,
            "seq_len":      self.seq_len,
        }
        joblib.dump(state, path)
        log.info(f"[GRU] Saved → {path}")

    def load(self, path: Path) -> None:
        import joblib
        state = joblib.load(path)
        self._feature_names = state["feature_names"]
        self._input_size    = state["input_size"]
        self._net = _GRUNet(
            input_size=state["input_size"],
            hidden_size=state["hidden"],
            num_layers=state["n_layers"],
            dropout=state["dropout"],
            bidirectional=state.get("bidirectional", False),
        ).to(self.device)
        if state["state_dict"]:
            self._net.load_state_dict(state["state_dict"])
        log.info(f"[GRU] Loaded from {path}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
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
        optimizer: Optional["torch.optim.Optimizer"],
    ) -> float:
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self._net(xb)
            loss  = criterion(preds, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)
