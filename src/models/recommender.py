"""
Recommendation Engine
Provides top-K product recommendations for users using the trained NeuMF model.
Implements multiple recommendation strategies.
"""

import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.models.neumf import NeuMF
from src.config.settings import get_config


class RecommendationEngine:
    """Production-ready recommendation engine for financial products."""

    def __init__(
        self,
        model: NeuMF,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        device: str = None,
        config=None,
    ):
        self.config = config or get_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.users_df = users_df
        self.products_df = products_df
        self.interactions_df = interactions_df

        # Build user interaction history
        self.user_history: Dict[int, set] = {}
        for _, row in interactions_df.iterrows():
            uid = int(row["user_id"])
            pid = int(row["product_id"])
            self.user_history.setdefault(uid, set()).add(pid)

        # Precompute product popularity
        product_counts = interactions_df.groupby("product_id").size()
        self.product_popularity = product_counts / product_counts.sum()

    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_owned: bool = True,
        category_filter: Optional[str] = None,
        risk_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate top-K product recommendations for a user.

        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            exclude_owned: Whether to exclude products the user already interacted with
            category_filter: Filter by product category
            risk_filter: Filter by risk level

        Returns:
            List of recommendation dicts with product details and scores
        """
        candidate_products = self.products_df.copy()

        # Apply filters
        if exclude_owned and user_id in self.user_history:
            owned = self.user_history[user_id]
            candidate_products = candidate_products[
                ~candidate_products["product_id"].isin(owned)
            ]

        if category_filter:
            candidate_products = candidate_products[
                candidate_products["category"] == category_filter
            ]

        if risk_filter:
            candidate_products = candidate_products[
                candidate_products["risk_level"] == risk_filter
            ]

        if candidate_products.empty:
            return []

        # Score all candidates - clamp IDs to valid embedding range
        product_ids = candidate_products["product_id"].values
        num_products = self.model.gmf.product_embedding.num_embeddings
        num_users = self.model.gmf.user_embedding.num_embeddings
        safe_user_id = min(user_id, num_users - 1)
        safe_product_ids = np.clip(product_ids, 0, num_products - 1)
        user_tensor = torch.LongTensor([safe_user_id] * len(safe_product_ids)).to(self.device)
        product_tensor = torch.LongTensor(safe_product_ids).to(self.device)

        scores = self.model(user_tensor, product_tensor).cpu().numpy()

        # Combine with popularity for cold-start handling
        pop_scores = np.array([
            self.product_popularity.get(pid, 0.0) for pid in product_ids
        ])
        blended_scores = 0.85 * scores + 0.15 * pop_scores

        # Rank and select top-K
        top_indices = np.argsort(blended_scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            pid = int(product_ids[idx])
            product_row = candidate_products[candidate_products["product_id"] == pid].iloc[0]
            recommendations.append({
                "product_id": pid,
                "product_name": product_row["product_name"],
                "category": product_row["category"],
                "risk_level": product_row["risk_level"],
                "min_investment": float(product_row["min_investment"]),
                "annual_return_pct": float(product_row["annual_return_pct"]),
                "fee_pct": float(product_row["fee_pct"]),
                "term_months": int(product_row["term_months"]),
                "is_tax_advantaged": bool(product_row["is_tax_advantaged"]),
                "recommendation_score": round(float(blended_scores[idx]), 4),
                "confidence": self._score_to_confidence(float(blended_scores[idx])),
            })

        return recommendations

    @torch.no_grad()
    def get_similar_products(self, product_id: int, top_k: int = 5) -> List[Dict]:
        """Find similar products using embedding similarity."""
        num_products = self.model.gmf.product_embedding.num_embeddings
        safe_pid = min(product_id, num_products - 1)
        target_emb = self.model.get_product_embedding(
            torch.LongTensor([safe_pid]).to(self.device)
        )

        all_product_ids = self.products_df["product_id"].values
        safe_all_ids = np.clip(all_product_ids, 0, num_products - 1)
        all_embs = self.model.get_product_embedding(
            torch.LongTensor(safe_all_ids).to(self.device)
        )

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            target_emb.expand_as(all_embs), all_embs, dim=-1
        ).cpu().numpy()

        # Exclude self
        mask = (all_product_ids == product_id)
        if mask.any():
            similarity[mask] = -1

        top_indices = np.argsort(similarity)[::-1][:top_k]

        results = []
        for idx in top_indices:
            pid = int(all_product_ids[idx])
            row = self.products_df[self.products_df["product_id"] == pid].iloc[0]
            results.append({
                "product_id": pid,
                "product_name": row["product_name"],
                "category": row["category"],
                "risk_level": row["risk_level"],
                "similarity_score": round(float(similarity[idx]), 4),
            })

        return results

    @torch.no_grad()
    def get_similar_users(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """Find users with similar financial profiles and behavior."""
        num_users = self.model.gmf.user_embedding.num_embeddings
        safe_uid = min(user_id, num_users - 1)
        target_emb = self.model.get_user_embedding(
            torch.LongTensor([safe_uid]).to(self.device)
        )

        all_user_ids = self.users_df["user_id"].values
        safe_all_uids = np.clip(all_user_ids, 0, num_users - 1)
        all_embs = self.model.get_user_embedding(
            torch.LongTensor(safe_all_uids).to(self.device)
        )

        similarity = torch.nn.functional.cosine_similarity(
            target_emb.expand_as(all_embs), all_embs, dim=-1
        ).cpu().numpy()

        mask = (all_user_ids == user_id)
        if mask.any():
            similarity[mask] = -1
        top_indices = np.argsort(similarity)[::-1][:top_k]

        results = []
        for idx in top_indices:
            uid = int(all_user_ids[idx])
            row = self.users_df[self.users_df["user_id"] == uid].iloc[0]
            results.append({
                "user_id": uid,
                "age_group": row["age_group"],
                "income_bracket": row["income_bracket"],
                "risk_tolerance": row["risk_tolerance"],
                "similarity_score": round(float(similarity[idx]), 4),
            })

        return results

    def get_user_profile(self, user_id: int) -> Dict:
        """Get comprehensive user profile with interaction history."""
        user_row = self.users_df[self.users_df["user_id"] == user_id]
        if user_row.empty:
            return {"error": f"User {user_id} not found"}

        user_row = user_row.iloc[0]
        history = self.interactions_df[self.interactions_df["user_id"] == user_id]

        # Category affinity
        category_counts = history.merge(
            self.products_df[["product_id", "category"]], on="product_id"
        ).groupby("category").size().sort_values(ascending=False)

        return {
            "user_id": int(user_id),
            "demographics": {
                "age_group": user_row["age_group"],
                "income_bracket": user_row["income_bracket"],
                "employment_type": user_row["employment_type"],
                "education_level": user_row["education_level"],
                "region": user_row["region"],
            },
            "financial_profile": {
                "credit_score": int(user_row["credit_score"]),
                "risk_tolerance": user_row["risk_tolerance"],
                "total_assets": float(user_row["total_assets"]),
                "years_as_customer": int(user_row["years_as_customer"]),
                "num_existing_products": int(user_row["num_existing_products"]),
                "digital_engagement_score": float(user_row["digital_engagement_score"]),
            },
            "interaction_summary": {
                "total_interactions": len(history),
                "unique_products_viewed": history["product_id"].nunique(),
                "top_categories": category_counts.head(5).to_dict(),
            },
        }

    def get_trending_products(self, top_k: int = 10) -> List[Dict]:
        """Get currently trending/popular products."""
        popular = (
            self.interactions_df.groupby("product_id")
            .agg(
                interaction_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        popular["trend_score"] = (
            popular["interaction_count"] / popular["interaction_count"].max() * 0.6
            + popular["avg_rating"] / popular["avg_rating"].max() * 0.4
        )
        popular = popular.sort_values("trend_score", ascending=False).head(top_k)

        results = []
        for _, row in popular.iterrows():
            pid = int(row["product_id"])
            prod = self.products_df[self.products_df["product_id"] == pid].iloc[0]
            results.append({
                "product_id": pid,
                "product_name": prod["product_name"],
                "category": prod["category"],
                "risk_level": prod["risk_level"],
                "trend_score": round(float(row["trend_score"]), 4),
                "interaction_count": int(row["interaction_count"]),
            })

        return results

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        return "Very Low"
