"""
Synthetic Financial Data Generator
Generates realistic user profiles, financial products, and interaction data
for training the recommendation system.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from src.config.settings import get_config


class FinancialDataGenerator:
    """Generates synthetic financial services data for recommendation model training."""

    def __init__(self, config=None):
        self.config = config or get_config()
        np.random.seed(self.config.data.random_seed)

        self.age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        self.income_brackets = [
            "Under $30K", "$30K-$50K", "$50K-$75K",
            "$75K-$100K", "$100K-$150K", "$150K-$250K", "Over $250K"
        ]
        self.employment_types = [
            "Full-time", "Part-time", "Self-employed",
            "Retired", "Student", "Unemployed"
        ]
        self.education_levels = [
            "High School", "Associate", "Bachelor's",
            "Master's", "PhD", "Professional"
        ]
        self.regions = [
            "Northeast", "Southeast", "Midwest",
            "Southwest", "West Coast", "Pacific Northwest"
        ]

    def generate_users(self, n_users: int = None) -> pd.DataFrame:
        """Generate synthetic user profiles with financial attributes."""
        n = n_users or self.config.data.num_users

        users = pd.DataFrame({
            "user_id": range(n),
            "age_group": np.random.choice(self.age_groups, n, p=[0.15, 0.25, 0.25, 0.18, 0.12, 0.05]),
            "income_bracket": np.random.choice(self.income_brackets, n, p=[0.10, 0.15, 0.20, 0.20, 0.18, 0.12, 0.05]),
            "employment_type": np.random.choice(self.employment_types, n, p=[0.55, 0.10, 0.12, 0.08, 0.10, 0.05]),
            "education_level": np.random.choice(self.education_levels, n, p=[0.15, 0.12, 0.35, 0.25, 0.05, 0.08]),
            "region": np.random.choice(self.regions, n),
            "credit_score": np.clip(np.random.normal(700, 80, n).astype(int), 300, 850),
            "years_as_customer": np.random.exponential(5, n).astype(int).clip(0, 30),
            "num_existing_products": np.random.poisson(3, n).clip(0, 12),
            "digital_engagement_score": np.clip(np.random.beta(2, 5, n) * 100, 0, 100).round(1),
            "risk_tolerance": np.random.choice(
                self.config.product.risk_levels, n, p=[0.35, 0.45, 0.20]
            ),
            "total_assets": np.clip(
                np.random.lognormal(10.5, 1.5, n).astype(int), 1000, 5000000
            ),
        })
        return users

    def generate_products(self, n_products: int = None) -> pd.DataFrame:
        """Generate financial product catalog with attributes."""
        n = n_products or self.config.data.num_products
        categories = self.config.product.categories

        products = []
        pid = 0
        for _ in range(n):
            category = np.random.choice(categories)
            risk = self._get_product_risk(category)
            min_invest = self._get_min_investment(category)
            apr = self._get_apr(category)

            products.append({
                "product_id": pid,
                "product_name": f"{category} - Plan {pid}",
                "category": category,
                "risk_level": risk,
                "min_investment": min_invest,
                "annual_return_pct": round(apr, 2),
                "fee_pct": round(np.random.uniform(0.0, 2.5), 2),
                "term_months": np.random.choice([0, 6, 12, 24, 36, 60, 120, 240, 360]),
                "is_tax_advantaged": int(category in [
                    "Retirement Plan (401k/IRA)", "Treasury Bonds",
                    "Life Insurance", "Health Insurance"
                ]),
                "digital_only": int(np.random.random() < 0.4),
                "popularity_score": round(np.random.beta(2, 5) * 100, 1),
            })
            pid += 1

        return pd.DataFrame(products)

    def generate_interactions(
        self, users: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate user-product interaction data with implicit feedback signals."""
        n_users = len(users)
        n_products = len(products)
        density = self.config.data.interaction_density

        n_interactions = int(n_users * n_products * density)

        user_ids = np.random.choice(n_users, n_interactions)
        product_ids = np.random.choice(n_products, n_interactions)

        # Generate interaction types with realistic distribution
        interaction_types = np.random.choice(
            ["view", "click", "inquiry", "application", "purchase"],
            n_interactions,
            p=[0.45, 0.25, 0.15, 0.10, 0.05]
        )

        # Map interaction types to implicit ratings (1-5)
        rating_map = {"view": 1, "click": 2, "inquiry": 3, "application": 4, "purchase": 5}
        ratings = np.array([rating_map[it] for it in interaction_types])

        # Add temporal component
        base_date = datetime(2023, 1, 1)
        days_offset = np.random.exponential(180, n_interactions).astype(int)
        timestamps = [base_date + timedelta(days=int(d)) for d in days_offset]

        interactions = pd.DataFrame({
            "user_id": user_ids,
            "product_id": product_ids,
            "interaction_type": interaction_types,
            "rating": ratings,
            "timestamp": timestamps,
            "session_duration_sec": np.random.exponential(120, n_interactions).astype(int),
            "device": np.random.choice(["mobile", "desktop", "tablet"], n_interactions, p=[0.55, 0.35, 0.10]),
        })

        # Remove duplicate (user, product, interaction_type) - keep latest
        interactions = interactions.sort_values("timestamp").drop_duplicates(
            subset=["user_id", "product_id", "interaction_type"], keep="last"
        ).reset_index(drop=True)

        return interactions

    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the complete dataset: users, products, interactions."""
        users = self.generate_users()
        products = self.generate_products()
        interactions = self.generate_interactions(users, products)

        return users, products, interactions

    def save_dataset(
        self,
        users: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
        output_dir: str = None
    ):
        """Save generated data to CSV files."""
        output_dir = output_dir or self.config.data.processed_data_dir
        os.makedirs(output_dir, exist_ok=True)

        users.to_csv(os.path.join(output_dir, "users.csv"), index=False)
        products.to_csv(os.path.join(output_dir, "products.csv"), index=False)
        interactions.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)

        # Save metadata
        metadata = {
            "num_users": len(users),
            "num_products": len(products),
            "num_interactions": len(interactions),
            "interaction_density": len(interactions) / (len(users) * len(products)),
            "generated_at": datetime.now().isoformat(),
            "categories": self.config.product.categories,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_dir}/")
        print(f"  Users: {len(users):,}")
        print(f"  Products: {len(products):,}")
        print(f"  Interactions: {len(interactions):,}")

        return metadata

    # ---------- helpers ----------
    def _get_product_risk(self, category: str) -> str:
        risk_map = {
            "Savings Account": "Conservative",
            "Checking Account": "Conservative",
            "Certificate of Deposit (CD)": "Conservative",
            "Money Market Account": "Conservative",
            "Treasury Bonds": "Conservative",
            "Life Insurance": "Conservative",
            "Health Insurance": "Conservative",
            "Credit Card": "Moderate",
            "Personal Loan": "Moderate",
            "Mortgage": "Moderate",
            "Auto Loan": "Moderate",
            "Student Loan Refinance": "Moderate",
            "Home Equity Line (HELOC)": "Moderate",
            "Investment Fund": "Aggressive",
            "Retirement Plan (401k/IRA)": "Moderate",
            "Brokerage Account": "Aggressive",
            "ETF Portfolio": "Moderate",
            "Corporate Bonds": "Moderate",
            "REITs": "Aggressive",
            "Annuity": "Moderate",
        }
        return risk_map.get(category, np.random.choice(self.config.product.risk_levels))

    def _get_min_investment(self, category: str) -> float:
        ranges = {
            "Savings Account": (0, 100),
            "Checking Account": (0, 25),
            "Credit Card": (0, 0),
            "Personal Loan": (1000, 5000),
            "Mortgage": (10000, 50000),
            "Auto Loan": (2000, 10000),
            "Investment Fund": (1000, 25000),
            "Retirement Plan (401k/IRA)": (0, 500),
            "Certificate of Deposit (CD)": (500, 10000),
            "Money Market Account": (1000, 25000),
            "Life Insurance": (0, 200),
            "Health Insurance": (0, 100),
            "Brokerage Account": (500, 10000),
            "ETF Portfolio": (100, 5000),
            "Treasury Bonds": (100, 1000),
            "Corporate Bonds": (1000, 10000),
            "REITs": (500, 5000),
            "Annuity": (5000, 50000),
            "Student Loan Refinance": (5000, 20000),
            "Home Equity Line (HELOC)": (10000, 50000),
        }
        lo, hi = ranges.get(category, (0, 5000))
        return int(np.random.uniform(lo, hi))

    def _get_apr(self, category: str) -> float:
        apr_ranges = {
            "Savings Account": (0.5, 5.0),
            "Checking Account": (0.0, 0.5),
            "Credit Card": (15.0, 28.0),
            "Personal Loan": (6.0, 18.0),
            "Mortgage": (3.0, 8.0),
            "Auto Loan": (3.5, 12.0),
            "Investment Fund": (4.0, 15.0),
            "Retirement Plan (401k/IRA)": (5.0, 12.0),
            "Certificate of Deposit (CD)": (2.0, 5.5),
            "Money Market Account": (1.5, 5.0),
            "Life Insurance": (2.0, 6.0),
            "Health Insurance": (0.0, 0.0),
            "Brokerage Account": (5.0, 20.0),
            "ETF Portfolio": (6.0, 15.0),
            "Treasury Bonds": (2.0, 5.5),
            "Corporate Bonds": (3.5, 8.0),
            "REITs": (5.0, 14.0),
            "Annuity": (3.0, 7.0),
            "Student Loan Refinance": (3.5, 10.0),
            "Home Equity Line (HELOC)": (4.0, 12.0),
        }
        lo, hi = apr_ranges.get(category, (2.0, 10.0))
        return np.random.uniform(lo, hi)


if __name__ == "__main__":
    generator = FinancialDataGenerator()
    users, products, interactions = generator.generate_dataset()
    generator.save_dataset(users, products, interactions)
