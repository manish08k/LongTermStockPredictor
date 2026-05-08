"""
src/deep_learning/lstm.py – LSTM model for time-series stock prediction.

Architecture:
  - Multi-layer LSTM with dropout
  - Binary classification head (up/down)
  - PyTorch implementation with GPU support
  - Early stopping + checkpoint saving
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

log = get_logger(__name__)


# ── Dataset builder ────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature array into overlapping sequences for LSTM input.

    Parameters
    ----------
    X       : (n_samples, n_features) feature array
    y       : (n_samples,) target array
    seq_len : Number of time steps per sequence

    Returns
    -------
    X_seq : (n_sequences, seq_len, n_features)
    y_seq : (n_sequences,)
    """
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"Not enough samples ({n}) for seq_len={seq_len}")

    X_seq = np.stack([X[i: i + seq_len] for i in range(n - seq_len)])
    y_seq = y[seq_len:]
    return X_seq, y_seq


# ── PyTorch Module ─────────────────────────────────────────────────────────

class _LSTMNet(nn.Module if HAS_TORCH else object):
    """Stacked LSTM with a binary classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        out, _ = self.lstm(x)         # (B, T, H)
        last = out[:, -1, :]          # take last time step
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


# ── Public Model class ─────────────────────────────────────────────────────

class LSTMModel:
    """
    LSTM-based binary classifier for stock direction prediction.

    Compatible with the BaseModel interface (fit / predict_proba).
    """

    name = "lstm"

    def __init__(self, cfg: dict) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMModel. pip install torch")

        dl_cfg = cfg.get("deep_learning", {})
        self.seq_len:    int   = dl_cfg.get("sequence_length", 60)
        self.hidden:     int   = dl_cfg.get("hidden_size", 128)
        self.n_layers:   int   = dl_cfg.get("num_layers", 2)
        self.dropout:    float = dl_cfg.get("dropout", 0.2)
        self.batch_size: int   = dl_cfg.get("batch_size", 64)
        self.epochs:     int   = dl_cfg.get("epochs", 50)
        self.lr:         float = dl_cfg.get("learning_rate", 0.001)
        self.patience:   int   = dl_cfg.get("early_stopping_patience", 10)
        self.ckpt_dir:   Path  = Path(dl_cfg.get("checkpoint_dir", "models/dl_checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        use_gpu = dl_cfg.get("use_gpu", True)
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        log.info(f"[LSTM] device={self.device}")

        self._net: Optional[_LSTMNet] = None
        self._feature_names: List[str] = []
        self._input_size: int = 0

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the LSTM on sequential windows of features.

        Parameters
        ----------
        X : Feature DataFrame (sorted by time)
        y : Binary target Series
        """
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        # Build sequences
        X_seq, y_seq = build_sequences(X_arr, y_arr, self.seq_len)
        n = len(X_seq)
        self._input_size = X_seq.shape[2]

        # Train/val split (80/20 chronological)
        split = int(0.8 * n)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        tr_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        # Build network
        self._net = _LSTMNet(
            input_size=self._input_size,
            hidden_size=self.hidden,
            num_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, verbose=False
        )
        criterion = nn.BCELoss()

        best_val_loss = math.inf
        patience_ctr = 0
        ckpt_path = self.ckpt_dir / "lstm_best.pt"

        log.info(
            f"[LSTM] Training: {n} sequences | input={self._input_size} | "
            f"hidden={self.hidden} | layers={self.n_layers} | device={self.device}"
        )

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self._net.train()
            tr_loss = self._run_epoch(tr_loader, criterion, optimizer)

            # ── Validate ──
            self._net.eval()
            val_loss = self._run_epoch(val_loader, criterion, optimizer=None)
            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    f"[LSTM] Epoch {epoch:3d}/{self.epochs} | "
                    f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}"
                )

            # ── Early stopping ──
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_ctr = 0
                torch.save(self._net.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    log.info(f"[LSTM] Early stopping at epoch {epoch}")
                    break

        # Load best checkpoint
        if ckpt_path.exists():
            self._net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        log.info(f"[LSTM] Training complete. Best val_loss={best_val_loss:.4f}")

    # ── predict_proba ──────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns array of shape (n_samples,) with P(up) probabilities.
        Pads first seq_len-1 rows with 0.5 to match original index.
        """
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_arr = X[self._feature_names].values.astype(np.float32)

        if len(X_arr) <= self.seq_len:
            log.warning("[LSTM] Not enough samples for prediction; returning 0.5")
            return np.full(len(X_arr), 0.5, dtype=np.float32)

        X_seq, _ = build_sequences(X_arr, np.zeros(len(X_arr)), self.seq_len)
        loader = self._make_loader(X_seq, np.zeros(len(X_seq)), shuffle=False)

        self._net.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                out = self._net(xb).cpu().numpy()
                preds.append(out)

        seq_probs = np.concatenate(preds)
        # Pad beginning (no sequence available for first seq_len rows)
        padded = np.full(len(X_arr), 0.5, dtype=np.float32)
        padded[self.seq_len:] = seq_probs
        return padded

    # ── feature_importances (not supported for LSTM) ───────────────────────

    def feature_importances(self) -> pd.Series:
        """LSTM does not provide direct feature importances. Returns zeros."""
        return pd.Series(
            np.zeros(len(self._feature_names)), index=self._feature_names
        )

    # ── save / load ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        import joblib
        state = {
            "state_dict": self._net.state_dict() if self._net else None,
            "feature_names": self._feature_names,
            "input_size": self._input_size,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
        }
        joblib.dump(state, path)
        log.info(f"[LSTM] Saved → {path}")

    def load(self, path: Path) -> None:
        import joblib
        state = joblib.load(path)
        self._feature_names = state["feature_names"]
        self._input_size = state["input_size"]
        self._net = _LSTMNet(
            input_size=state["input_size"],
            hidden_size=state["hidden"],
            num_layers=state["n_layers"],
            dropout=state["dropout"],
        ).to(self.device)
        if state["state_dict"]:
            self._net.load_state_dict(state["state_dict"])
        log.info(f"[LSTM] Loaded from {path}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False,
    ) -> "DataLoader":
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
        optimizer: Optional["torch.optim.Optimizer"],
    ) -> float:
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self._net(xb)
            loss = criterion(preds, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)
