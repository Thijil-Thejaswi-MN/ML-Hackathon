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

        # REAL epoch-wise history dictionary for reports & plots
        self.history: Optional[dict] = {"epoch": [], "train_loss": [], "val_loss": []}

    def fit(self, X, y, groups=None, **kw) -> "MLPCandidateModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.in_features = X_arr.shape[1]
        self.net = _PyTorchMLP(self.in_features, self.hidden_dims, self.dropout).to(self.device)

        # Train / val split for epoch curves (15% validation split)
        n = len(X_arr)
        indices = np.arange(n)
        np.random.shuffle(indices)
        val_size = int(0.15 * n)
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        X_train, y_train = torch.tensor(X_arr[train_idx]), torch.tensor(y_arr[train_idx])
        X_val, y_val = torch.tensor(X_arr[val_idx]), torch.tensor(y_arr[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True
        )

        pos_w = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.history = {"epoch": [], "train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            train_losses = []
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.net(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Evaluate val loss
            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_val.to(self.device))
                val_loss = criterion(val_logits, y_val.to(self.device)).item()

            avg_train_loss = float(np.mean(train_losses))
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(avg_train_loss, 5))
            self.history["val_loss"].append(round(val_loss, 5))

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.net is None:
            raise ValueError("MLPCandidateModel is not fitted.")
        self.net.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X_arr, device=self.device)
        with torch.no_grad():
            logits = self.net(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

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
        instance.history = data.get("history", {})
        if data["state_dict"] is not None and instance.in_features is not None:
            instance.net = _PyTorchMLP(instance.in_features, instance.hidden_dims, instance.dropout)
            instance.net.load_state_dict(data["state_dict"])
            instance.net.eval()
        return instance
