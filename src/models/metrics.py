"""
Evaluation Metrics for Recommendation Systems
Implements standard information retrieval and recommendation metrics.
"""

import numpy as np
from typing import List, Dict


class RecommendationMetrics:
    """Computes standard recommendation system evaluation metrics."""

    @staticmethod
    def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """HR@K: 1 if any relevant item appears in top-K, else 0."""
        top_k = recommended[:k]
        return float(any(item in relevant for item in top_k))

    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """Precision@K: fraction of recommended items that are relevant."""
        top_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """Recall@K: fraction of relevant items that are recommended."""
        top_k = set(recommended[:k])
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: List[int], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain @ K."""
        top_k = recommended[:k]
        relevant_set = set(relevant)

        dcg = sum(
            1.0 / np.log2(i + 2) for i, item in enumerate(top_k)
            if item in relevant_set
        )

        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(recommended: List[int], relevant: List[int]) -> float:
        """MRR: reciprocal of the rank of the first relevant item."""
        relevant_set = set(relevant)
        for i, item in enumerate(recommended):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def coverage(all_recommendations: List[List[int]], total_items: int) -> float:
        """Catalog coverage: fraction of items ever recommended."""
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        return len(unique_recommended) / total_items

    @staticmethod
    def diversity(recommendations: List[Dict], key: str = "category") -> float:
        """Intra-list diversity: fraction of unique categories in recommendations."""
        if not recommendations:
            return 0.0
        categories = [r[key] for r in recommendations]
        return len(set(categories)) / len(categories)

    @classmethod
    def evaluate_all(
        cls,
        recommended_ids: List[int],
        relevant_ids: List[int],
        recommendations: List[Dict] = None,
        k: int = 10,
    ) -> Dict[str, float]:
        """Compute all metrics at once."""
        metrics = {
            f"HR@{k}": cls.hit_rate_at_k(recommended_ids, relevant_ids, k),
            f"Precision@{k}": cls.precision_at_k(recommended_ids, relevant_ids, k),
            f"Recall@{k}": cls.recall_at_k(recommended_ids, relevant_ids, k),
            f"nDCG@{k}": cls.ndcg_at_k(recommended_ids, relevant_ids, k),
            "MRR": cls.mean_reciprocal_rank(recommended_ids, relevant_ids),
        }
        if recommendations:
            metrics["Diversity"] = cls.diversity(recommendations)

        return {k: round(v, 4) for k, v in metrics.items()}
