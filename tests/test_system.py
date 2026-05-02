"""
Test Suite for the Deep Learning Recommendation System
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_config, AppConfig
from src.data.generator import FinancialDataGenerator
from src.data.preprocessor import DataPreprocessor, InteractionDataset
from src.models.neumf import NeuMF, GMFLayer, MLPLayer, HybridRecommender
from src.models.trainer import Trainer
from src.models.recommender import RecommendationEngine
from src.models.metrics import RecommendationMetrics


@pytest.fixture
def config():
    cfg = get_config()
    cfg.data.num_users = 200
    cfg.data.num_products = 30
    cfg.model.num_epochs = 2
    cfg.model.batch_size = 32
    cfg.model.embedding_dim = 16
    cfg.model.mlp_layers = [32, 16]
    return cfg


@pytest.fixture
def sample_data(config):
    gen = FinancialDataGenerator(config)
    users, products, interactions = gen.generate_dataset()
    return users, products, interactions


class TestConfig:
    def test_default_config(self):
        cfg = get_config()
        assert cfg.model.embedding_dim == 64
        assert cfg.model.learning_rate == 0.001
        assert len(cfg.product.categories) == 20

    def test_config_types(self):
        cfg = get_config()
        assert isinstance(cfg, AppConfig)
        assert isinstance(cfg.model.mlp_layers, list)


class TestDataGenerator:
    def test_generate_users(self, config):
        gen = FinancialDataGenerator(config)
        users = gen.generate_users()
        assert len(users) == config.data.num_users
        assert "user_id" in users.columns
        assert "credit_score" in users.columns
        assert users["credit_score"].between(300, 850).all()

    def test_generate_products(self, config):
        gen = FinancialDataGenerator(config)
        products = gen.generate_products()
        assert len(products) == config.data.num_products
        assert "product_id" in products.columns
        assert "category" in products.columns

    def test_generate_interactions(self, sample_data):
        users, products, interactions = sample_data
        assert len(interactions) > 0
        assert "user_id" in interactions.columns
        assert "product_id" in interactions.columns
        assert interactions["rating"].between(1, 5).all()

    def test_dataset_save_load(self, config, sample_data, tmp_path):
        users, products, interactions = sample_data
        gen = FinancialDataGenerator(config)
        gen.save_dataset(users, products, interactions, str(tmp_path))
        assert (tmp_path / "users.csv").exists()
        assert (tmp_path / "products.csv").exists()
        assert (tmp_path / "interactions.csv").exists()
        assert (tmp_path / "metadata.json").exists()


class TestPreprocessor:
    def test_preprocess(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)
        assert "train" in processed
        assert "val" in processed
        assert "test" in processed
        assert processed["num_users"] > 0
        assert processed["num_products"] > 0

    def test_dataloaders(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)
        loaders = preprocessor.create_dataloaders(processed)
        assert "train" in loaders
        batch = next(iter(loaders["train"]))
        assert "user_id" in batch
        assert "product_id" in batch
        assert "rating" in batch


class TestModels:
    def test_neumf_forward(self):
        model = NeuMF(100, 50, gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16])
        users = torch.LongTensor([0, 1, 2])
        products = torch.LongTensor([0, 1, 2])
        output = model(users, products)
        assert output.shape == (3,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_neumf_embeddings(self):
        model = NeuMF(100, 50, gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16])
        user_emb = model.get_user_embedding(torch.LongTensor([0]))
        assert user_emb.shape == (1, 32)  # gmf + mlp

    def test_gmf_layer(self):
        gmf = GMFLayer(100, 50, 16)
        out = gmf(torch.LongTensor([0, 1]), torch.LongTensor([0, 1]))
        assert out.shape == (2, 16)

    def test_mlp_layer(self):
        mlp = MLPLayer(100, 50, 16, [32, 16])
        out = mlp(torch.LongTensor([0, 1]), torch.LongTensor([0, 1]))
        assert out.shape == (2, 16)


class TestTrainer:
    def test_training(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)
        loaders = preprocessor.create_dataloaders(processed)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        trainer = Trainer(model, config)
        history = trainer.fit(loaders["train"], loaders["val"], num_epochs=2)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_evaluation(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)
        loaders = preprocessor.create_dataloaders(processed)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        trainer = Trainer(model, config)
        metrics = trainer.evaluate(loaders["test"])
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc_roc" in metrics


class TestRecommender:
    def test_recommend(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        engine = RecommendationEngine(model, users, products, interactions)
        recs = engine.recommend(0, top_k=5)
        assert len(recs) <= 5
        assert all("product_id" in r for r in recs)
        assert all("recommendation_score" in r for r in recs)

    def test_similar_products(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        engine = RecommendationEngine(model, users, products, interactions)
        similar = engine.get_similar_products(0, top_k=3)
        assert len(similar) <= 3

    def test_user_profile(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        engine = RecommendationEngine(model, users, products, interactions)
        profile = engine.get_user_profile(0)
        assert "demographics" in profile
        assert "financial_profile" in profile

    def test_trending(self, config, sample_data):
        users, products, interactions = sample_data
        preprocessor = DataPreprocessor(config)
        processed = preprocessor.preprocess(users, products, interactions)

        model = NeuMF(
            processed["num_users"], processed["num_products"],
            gmf_embedding_dim=16, mlp_embedding_dim=16, mlp_hidden_layers=[32, 16],
        )
        engine = RecommendationEngine(model, users, products, interactions)
        trending = engine.get_trending_products(5)
        assert len(trending) <= 5


class TestMetrics:
    def test_hit_rate(self):
        assert RecommendationMetrics.hit_rate_at_k([1, 2, 3], [3, 5], k=3) == 1.0
        assert RecommendationMetrics.hit_rate_at_k([1, 2, 3], [4, 5], k=3) == 0.0

    def test_ndcg(self):
        ndcg = RecommendationMetrics.ndcg_at_k([1, 2, 3], [1], k=3)
        assert ndcg == 1.0  # first item is relevant

    def test_diversity(self):
        recs = [{"category": "A"}, {"category": "B"}, {"category": "C"}]
        assert RecommendationMetrics.diversity(recs) == 1.0
        recs_dup = [{"category": "A"}, {"category": "A"}, {"category": "A"}]
        assert abs(RecommendationMetrics.diversity(recs_dup) - 1 / 3) < 0.01

    def test_evaluate_all(self):
        results = RecommendationMetrics.evaluate_all([1, 2, 3, 4, 5], [2, 4, 6])
        assert "HR@10" in results
        assert "nDCG@10" in results
        assert "MRR" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
