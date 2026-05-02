"""
Data Preprocessing & PyTorch Dataset
Handles train/val/test splits, negative sampling, and DataLoader creation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from src.config.settings import get_config


class InteractionDataset(Dataset):
    """PyTorch Dataset for user-product interactions with negative sampling."""

    def __init__(
        self,
        user_ids: np.ndarray,
        product_ids: np.ndarray,
        ratings: np.ndarray,
        user_features: Optional[np.ndarray] = None,
        product_features: Optional[np.ndarray] = None,
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.product_ids = torch.LongTensor(product_ids)
        self.ratings = torch.FloatTensor(ratings)
        self.user_features = (
            torch.FloatTensor(user_features) if user_features is not None else None
        )
        self.product_features = (
            torch.FloatTensor(product_features) if product_features is not None else None
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        sample = {
            "user_id": self.user_ids[idx],
            "product_id": self.product_ids[idx],
            "rating": self.ratings[idx],
        }
        if self.user_features is not None:
            sample["user_features"] = self.user_features[idx]
        if self.product_features is not None:
            sample["product_features"] = self.product_features[idx]
        return sample


class DataPreprocessor:
    """Preprocesses raw financial data for the recommendation model."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.user_feature_encoders: Dict[str, LabelEncoder] = {}
        self.product_feature_encoders: Dict[str, LabelEncoder] = {}

    def preprocess(
        self,
        users: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> Dict:
        """Full preprocessing pipeline."""
        # Encode IDs
        interactions = interactions.copy()
        interactions["user_idx"] = self.user_encoder.fit_transform(interactions["user_id"])
        interactions["product_idx"] = self.product_encoder.fit_transform(interactions["product_id"])

        num_users = interactions["user_idx"].nunique()
        num_products = interactions["product_idx"].nunique()

        # Normalize ratings to [0, 1]
        max_rating = interactions["rating"].max()
        interactions["rating_norm"] = interactions["rating"] / max_rating

        # Encode user features
        user_feature_cols = ["age_group", "income_bracket", "employment_type",
                            "education_level", "region", "risk_tolerance"]
        user_features_encoded = self._encode_categorical_features(
            users, user_feature_cols, self.user_feature_encoders
        )

        # Numeric user features (normalize)
        numeric_user_cols = ["credit_score", "years_as_customer",
                            "num_existing_products", "digital_engagement_score", "total_assets"]
        user_numeric = users[numeric_user_cols].values.astype(np.float32)
        user_numeric = (user_numeric - user_numeric.mean(axis=0)) / (user_numeric.std(axis=0) + 1e-8)

        user_features_all = np.hstack([user_features_encoded, user_numeric])

        # Encode product features
        product_feature_cols = ["category", "risk_level"]
        product_features_encoded = self._encode_categorical_features(
            products, product_feature_cols, self.product_feature_encoders
        )

        numeric_product_cols = ["min_investment", "annual_return_pct", "fee_pct",
                                "term_months", "is_tax_advantaged", "digital_only", "popularity_score"]
        product_numeric = products[numeric_product_cols].values.astype(np.float32)
        product_numeric = (product_numeric - product_numeric.mean(axis=0)) / (product_numeric.std(axis=0) + 1e-8)

        product_features_all = np.hstack([product_features_encoded, product_numeric])

        # Train/val/test split (by timestamp)
        interactions = interactions.sort_values("timestamp")
        n = len(interactions)
        train_end = int(n * self.config.data.train_ratio)
        val_end = int(n * (self.config.data.train_ratio + self.config.data.val_ratio))

        train_df = interactions.iloc[:train_end]
        val_df = interactions.iloc[train_end:val_end]
        test_df = interactions.iloc[val_end:]

        # Add negative samples to training data
        train_df = self._add_negative_samples(train_df, num_products)

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "num_users": num_users,
            "num_products": num_products,
            "user_features": user_features_all,
            "product_features": product_features_all,
            "user_feature_dim": user_features_all.shape[1],
            "product_feature_dim": product_features_all.shape[1],
        }

    def create_dataloaders(self, preprocessed: Dict) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders from preprocessed data."""
        batch_size = self.config.model.batch_size

        loaders = {}
        for split in ["train", "val", "test"]:
            df = preprocessed[split]
            dataset = InteractionDataset(
                user_ids=df["user_idx"].values,
                product_ids=df["product_idx"].values,
                ratings=df["rating_norm"].values,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0,
                pin_memory=True,
            )

        return loaders

    def _encode_categorical_features(
        self, df: pd.DataFrame, cols: list, encoders: dict
    ) -> np.ndarray:
        """One-hot encode categorical features."""
        encoded_parts = []
        for col in cols:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                encoders[col].fit(df[col])
            labels = encoders[col].transform(df[col])
            n_classes = len(encoders[col].classes_)
            one_hot = np.zeros((len(df), n_classes), dtype=np.float32)
            one_hot[np.arange(len(df)), labels] = 1.0
            encoded_parts.append(one_hot)
        return np.hstack(encoded_parts)

    def _add_negative_samples(
        self, df: pd.DataFrame, num_products: int
    ) -> pd.DataFrame:
        """Add negative samples for implicit feedback training."""
        n_neg = self.config.model.num_negative_samples
        positive_set = set(zip(df["user_idx"], df["product_idx"]))

        neg_rows = []
        for _, row in df.iterrows():
            user = row["user_idx"]
            count = 0
            while count < n_neg:
                neg_product = np.random.randint(0, num_products)
                if (user, neg_product) not in positive_set:
                    neg_rows.append({
                        "user_id": row["user_id"],
                        "product_id": -1,
                        "user_idx": user,
                        "product_idx": neg_product,
                        "rating": 0,
                        "rating_norm": 0.0,
                        "interaction_type": "negative",
                        "timestamp": row["timestamp"],
                        "session_duration_sec": 0,
                        "device": "none",
                    })
                    count += 1

        neg_df = pd.DataFrame(neg_rows)
        combined = pd.concat([df, neg_df], ignore_index=True)
        return combined.sample(frac=1, random_state=self.config.data.random_seed).reset_index(drop=True)
