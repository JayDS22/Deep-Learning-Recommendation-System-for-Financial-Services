"""
Training Engine
Handles model training loop, evaluation, checkpointing, and metrics logging.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.config.settings import get_config


class Trainer:
    """Training engine for the recommendation model."""

    def __init__(self, model: nn.Module, config=None, device: str = None):
        self.config = config or get_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.criterion = nn.BCELoss()
        self.history: Dict[str, List[float]] = defaultdict(list)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            user_ids = batch["user_id"].to(self.device)
            product_ids = batch["product_id"].to(self.device)
            ratings = batch["rating"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(user_ids, product_ids)
            loss = self.criterion(predictions, ratings)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test data."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        n_batches = 0

        for batch in dataloader:
            user_ids = batch["user_id"].to(self.device)
            product_ids = batch["product_id"].to(self.device)
            ratings = batch["rating"].to(self.device)

            predictions = self.model(user_ids, product_ids)
            loss = self.criterion(predictions, ratings)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
            n_batches += 1

        preds = np.array(all_predictions)
        labels = np.array(all_labels)
        binary_preds = (preds > 0.5).astype(int)
        binary_labels = (labels > 0.5).astype(int)

        accuracy = np.mean(binary_preds == binary_labels)
        precision = self._precision(binary_preds, binary_labels)
        recall = self._recall(binary_preds, binary_labels)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        auc = self._auc_approx(preds, binary_labels)

        return {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        num_epochs = num_epochs or self.config.model.num_epochs
        patience = self.config.model.early_stopping_patience
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nTraining on {self.device} for up to {num_epochs} epochs...")
        print("-" * 70)

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            self.scheduler.step(val_loss)
            elapsed = time.time() - start_time

            # Log history
            self.history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                f"Val F1: {val_metrics['f1_score']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                    break

        self._load_best_model()
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        return dict(self.history)

    def _save_best_model(self):
        """Save the best model checkpoint."""
        save_dir = self.config.data.model_save_dir
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "best_model.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": dict(self.history),
        }, path)

    def _load_best_model(self):
        """Load the best model checkpoint."""
        path = os.path.join(self.config.data.model_save_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_metrics(self, test_metrics: Dict[str, float], filepath: str = None):
        """Save training history and test metrics."""
        filepath = filepath or os.path.join(self.config.data.model_save_dir, "metrics.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output = {
            "training_history": {k: [round(v, 5) for v in vals] for k, vals in self.history.items()},
            "test_metrics": {k: round(v, 5) for k, v in test_metrics.items()},
            "model_config": {
                "embedding_dim": self.config.model.embedding_dim,
                "mlp_layers": self.config.model.mlp_layers,
                "dropout_rate": self.config.model.dropout_rate,
                "learning_rate": self.config.model.learning_rate,
            },
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

    # ------- Metric Helpers -------
    @staticmethod
    def _precision(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        return tp / (tp + fp + 1e-8)

    @staticmethod
    def _recall(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        return tp / (tp + fn + 1e-8)

    @staticmethod
    def _auc_approx(scores, labels):
        """Approximate AUC-ROC without sklearn."""
        if len(np.unique(labels)) < 2:
            return 0.5
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n_correct = sum(1 for p in pos for n in neg if p > n)
        n_total = len(pos) * len(neg)
        return n_correct / max(n_total, 1)
