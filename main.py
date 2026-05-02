"""
Main Pipeline - End-to-End Training & Evaluation
Generates data, trains the NeuMF model, evaluates, and saves artifacts.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import get_config
from src.data.generator import FinancialDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.neumf import NeuMF
from src.models.trainer import Trainer
from src.models.recommender import RecommendationEngine
from src.models.metrics import RecommendationMetrics


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Deep Learning Recommendation System for Financial Services"
    )
    parser.add_argument("--num-users", type=int, default=5000)
    parser.add_argument("--num-products", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args(args)

    # Setup
    config = get_config()
    config.data.num_users = args.num_users
    config.data.num_products = args.num_products
    config.model.num_epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.embedding_dim = args.embedding_dim
    config.model.learning_rate = args.lr
    config.data.random_seed = args.seed
    config.data.model_save_dir = os.path.join(args.output_dir, "models")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  {config.project_name}")
    print("=" * 70)

    # ── Step 1: Generate Data ──────────────────────────────────────
    print("\n[1/5] Generating synthetic financial data...")
    generator = FinancialDataGenerator(config)
    users, products, interactions = generator.generate_dataset()
    data_dir = os.path.join(args.output_dir, "data")
    generator.save_dataset(users, products, interactions, data_dir)

    # ── Step 2: Preprocess ─────────────────────────────────────────
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor(config)
    processed = preprocessor.preprocess(users, products, interactions)
    loaders = preprocessor.create_dataloaders(processed)

    print(f"  Train samples: {len(processed['train']):,}")
    print(f"  Val samples:   {len(processed['val']):,}")
    print(f"  Test samples:  {len(processed['test']):,}")
    print(f"  Num users:     {processed['num_users']:,}")
    print(f"  Num products:  {processed['num_products']:,}")

    # ── Step 3: Build Model ────────────────────────────────────────
    print("\n[3/5] Building NeuMF model...")
    model = NeuMF(
        num_users=processed["num_users"],
        num_products=processed["num_products"],
        gmf_embedding_dim=config.model.embedding_dim,
        mlp_embedding_dim=config.model.embedding_dim,
        mlp_hidden_layers=config.model.mlp_layers,
        dropout_rate=config.model.dropout_rate,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Step 4: Train ──────────────────────────────────────────────
    print("\n[4/5] Training model...")
    trainer = Trainer(model, config)
    history = trainer.fit(loaders["train"], loaders["val"])

    # ── Step 5: Evaluate ───────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    test_metrics = trainer.evaluate(loaders["test"])
    trainer.save_metrics(test_metrics)

    print("\n" + "=" * 50)
    print("  Test Set Results")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # ── Recommendation Demo ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Recommendation Demo")
    print("=" * 50)

    engine = RecommendationEngine(model, users, products, interactions)

    demo_users = np.random.choice(users["user_id"], 3, replace=False)
    for uid in demo_users:
        print(f"\n--- User {uid} ---")
        profile = engine.get_user_profile(uid)
        demo = profile["demographics"]
        fin = profile["financial_profile"]
        print(f"  Age: {demo['age_group']}, Income: {demo['income_bracket']}, "
              f"Risk: {fin['risk_tolerance']}, Credit: {fin['credit_score']}")

        recs = engine.recommend(uid, top_k=5)
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['category']} (Score: {rec['recommendation_score']:.3f}, "
                  f"Risk: {rec['risk_level']}, Return: {rec['annual_return_pct']:.1f}%)")

    # ── Trending Products ──────────────────────────────────────────
    print("\n--- Trending Products ---")
    trending = engine.get_trending_products(5)
    for t in trending:
        print(f"  {t['category']} - Trend Score: {t['trend_score']:.3f} "
              f"({t['interaction_count']} interactions)")

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("Done!")

    return model, engine, test_metrics


if __name__ == "__main__":
    main()
