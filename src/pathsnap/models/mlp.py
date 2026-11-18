"""Multi-Layer Perceptron (MLP) Candidate Model.  OWNER: M4.  Contract: README 3.7.

Algorithm family: Neural Network.
Suitable for PathSnap: Captures continuous non-linear representations across geometric and temporal features.
Includes REAL epoch-wise train & validation loss recording (REQUIRED for Deliverables §3.3 loss curves).
"""
from __future__ import annotations
from typing import Optional
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathsnap.models.base import CandidateModel


class _PyTorchMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPCandidateModel(CandidateModel):
    name: str = "mlp"
    family: str = "Neural"
    accepts_eval_set = True          # training.cross_validate passes CV fold 0 for a real per-epoch history
    HISTORY_TRAIN_ROWS = 200_000     # fixed train subsample on which the per-epoch train log loss is measured
    PREDICT_CHUNK = 262_144

    def __init__(
        self,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 15,
        pos_weight: float = 4.0,  # handles candidate class imbalance (~1:5)
        random_state: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        self.params = {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "pos_weight": pos_weight,
            "random_state": random_state,
            "device": device,
            **kwargs,
        }
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.device = device
        self.random_state = random_state
        self.net: Optional[_PyTorchMLP] = None
        self.in_features: Optional[int] = None

        # REAL epoch-wise history (only when fit() receives eval_set); otherwise None, as the contract requires
        self.history: Optional[dict] = None

    def fit(self, X, y, groups=None, eval_set=None, **kw) -> "MLPCandidateModel":
        """Train on all rows given. If the training script passes eval_set=(X_val, y_val) (CV fold 0, see
        training.cross_validate) the unweighted log loss on a fixed train subsample and on that held-out fold is
        recorded after every epoch. The validation fold is only monitored, never used to stop or select."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.in_features = X_arr.shape[1]
        self.net = _PyTorchMLP(self.in_features, self.hidden_dims, self.dropout).to(self.device)

        gen = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr)),
                            batch_size=self.batch_size, shuffle=True, generator=gen,
                            drop_last=len(X_arr) > self.batch_size)   # BatchNorm cannot train on a 1-row batch
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.history = None
        if eval_set is not None:
            rng = np.random.default_rng(self.random_state)
            mon = rng.choice(len(X_arr), min(len(X_arr), self.HISTORY_TRAIN_ROWS), replace=False)
            X_mon, y_mon = X_arr[mon], y_arr[mon]
            X_val = np.asarray(eval_set[0], dtype=np.float32)
            y_val = np.asarray(eval_set[1], dtype=np.float32)
            hist = {"iteration": [], "train_logloss": [], "val_logloss": []}

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(bx), by)
                loss.backward()
                optimizer.step()
            if eval_set is not None:
                hist["iteration"].append(epoch)
                hist["train_logloss"].append(self._log_loss(X_mon, y_mon))
                hist["val_logloss"].append(self._log_loss(X_val, y_val))
                print(f"      epoch {epoch:2d}: train logloss {hist['train_logloss'][-1]:.4f}  "
                      f"val logloss {hist['val_logloss'][-1]:.4f}", flush=True)
        if eval_set is not None:
            self.history = hist
        return self

    def _log_loss(self, X, y) -> float:
        p = np.clip(self.predict_proba(X).astype(np.float64), 1e-7, 1 - 1e-7)
        y = np.asarray(y, dtype=np.float64)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict_proba(self, X) -> np.ndarray:
        if self.net is None:
            raise ValueError("MLPCandidateModel is not fitted.")
        self.net.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X_arr), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X_arr), self.PREDICT_CHUNK):
                xb = torch.from_numpy(X_arr[i:i + self.PREDICT_CHUNK]).to(self.device)
                out[i:i + self.PREDICT_CHUNK] = torch.sigmoid(self.net(xb)).cpu().numpy()
        return out

    def save(self, path: str) -> None:
        state = {
            "params": self.params,
            "in_features": self.in_features,
            "state_dict": self.net.state_dict() if self.net else None,
            "history": self.history,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "MLPCandidateModel":
        data = joblib.load(path)
        instance = cls(**data["params"])
        instance.in_features = data["in_features"]
        instance.history = data.get("history")
        if data["state_dict"] is not None and instance.in_features is not None:
            instance.net = _PyTorchMLP(instance.in_features, instance.hidden_dims, instance.dropout)
            instance.net.load_state_dict(data["state_dict"])
            instance.net.eval()
        return instance
