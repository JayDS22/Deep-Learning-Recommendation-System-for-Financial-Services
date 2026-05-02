"""
Neural Collaborative Filtering (NeuMF) Model
Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
for financial product recommendation.

Reference: He et al., "Neural Collaborative Filtering", WWW 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GMFLayer(nn.Module):
    """Generalized Matrix Factorization component."""

    def __init__(self, num_users: int, num_products: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)

    def forward(self, user_ids: torch.Tensor, product_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        product_emb = self.product_embedding(product_ids)
        return user_emb * product_emb  # Element-wise product


class MLPLayer(nn.Module):
    """Multi-Layer Perceptron component for learning non-linear interactions."""

    def __init__(
        self,
        num_users: int,
        num_products: int,
        embedding_dim: int,
        hidden_layers: list,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)

        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, product_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        product_emb = self.product_embedding(product_ids)
        concat = torch.cat([user_emb, product_emb], dim=-1)
        return self.mlp(concat)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) - the full recommendation model.

    Combines GMF (linear interaction patterns) with MLP (non-linear interaction
    patterns) through a fusion layer that predicts user-product affinity.
    """

    def __init__(
        self,
        num_users: int,
        num_products: int,
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 64,
        mlp_hidden_layers: list = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [256, 128, 64, 32]

        self.gmf = GMFLayer(num_users, num_products, gmf_embedding_dim)
        self.mlp = MLPLayer(
            num_users, num_products, mlp_embedding_dim,
            mlp_hidden_layers, dropout_rate
        )

        # Fusion layer
        fusion_input_dim = gmf_embedding_dim + mlp_hidden_layers[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:
        gmf_out = self.gmf(user_ids, product_ids)
        mlp_out = self.mlp(user_ids, product_ids)
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        prediction = self.fusion(concat).squeeze(-1)
        return prediction

    def get_user_embedding(self, user_id: torch.Tensor) -> torch.Tensor:
        """Get combined user embedding for similarity search."""
        gmf_emb = self.gmf.user_embedding(user_id)
        mlp_emb = self.mlp.user_embedding(user_id)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)

    def get_product_embedding(self, product_id: torch.Tensor) -> torch.Tensor:
        """Get combined product embedding for similarity search."""
        gmf_emb = self.gmf.product_embedding(product_id)
        mlp_emb = self.mlp.product_embedding(product_id)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)


class DeepFMLayer(nn.Module):
    """
    Deep Factorization Machine component for feature-rich recommendation.
    Captures both low-order (FM) and high-order (DNN) feature interactions.
    """

    def __init__(self, feature_dim: int, embedding_dim: int = 16, hidden_layers: list = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        # FM component - first-order
        self.linear = nn.Linear(feature_dim, 1)

        # FM component - second-order (factorized)
        self.fm_embedding = nn.Linear(feature_dim, embedding_dim)

        # DNN component
        layers = []
        input_dim = feature_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # First-order
        first_order = self.linear(features)

        # Second-order (simplified FM interaction)
        emb = self.fm_embedding(features)
        square_of_sum = emb.sum(dim=-1, keepdim=True) ** 2
        sum_of_square = (emb ** 2).sum(dim=-1, keepdim=True)
        second_order = 0.5 * (square_of_sum - sum_of_square)

        # DNN
        dnn_out = self.dnn(features)

        return torch.sigmoid(first_order + second_order + dnn_out).squeeze(-1)


class HybridRecommender(nn.Module):
    """
    Hybrid model combining NeuMF with content-based DeepFM features.
    Uses both collaborative filtering signals AND product/user attributes.
    """

    def __init__(
        self,
        num_users: int,
        num_products: int,
        user_feature_dim: int,
        product_feature_dim: int,
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 64,
        mlp_hidden_layers: list = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        if mlp_hidden_layers is None:
            mlp_hidden_layers = [256, 128, 64, 32]

        self.neumf = NeuMF(
            num_users, num_products,
            gmf_embedding_dim, mlp_embedding_dim,
            mlp_hidden_layers, dropout_rate,
        )

        # Content-based branch
        content_input_dim = user_feature_dim + product_feature_dim
        self.content_branch = nn.Sequential(
            nn.Linear(content_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Final fusion
        self.alpha = nn.Parameter(torch.tensor(0.7))  # Learnable weight

    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        product_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cf_score = self.neumf(user_ids, product_ids)

        if user_features is not None and product_features is not None:
            content_input = torch.cat([user_features, product_features], dim=-1)
            content_score = self.content_branch(content_input).squeeze(-1)
            alpha = torch.sigmoid(self.alpha)
            return alpha * cf_score + (1 - alpha) * content_score

        return cf_score
