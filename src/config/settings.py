"""
Configuration settings for the Deep Learning Recommendation System.
Centralized config for model hyperparameters, data paths, and API settings.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Neural Collaborative Filtering model configuration."""
    embedding_dim: int = 64
    mlp_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    gmf_output_dim: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    early_stopping_patience: int = 5
    num_negative_samples: int = 4
    top_k: int = 10


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_data_dir: str = os.path.join("data", "raw")
    processed_data_dir: str = os.path.join("data", "processed")
    model_save_dir: str = os.path.join("models", "saved")
    num_users: int = 10000
    num_products: int = 500
    interaction_density: float = 0.02
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class ProductConfig:
    """Financial product categories and metadata."""
    categories: List[str] = field(default_factory=lambda: [
        "Savings Account", "Checking Account", "Credit Card",
        "Personal Loan", "Mortgage", "Auto Loan",
        "Investment Fund", "Retirement Plan (401k/IRA)",
        "Certificate of Deposit (CD)", "Money Market Account",
        "Life Insurance", "Health Insurance",
        "Brokerage Account", "ETF Portfolio",
        "Treasury Bonds", "Corporate Bonds",
        "REITs", "Annuity",
        "Student Loan Refinance", "Home Equity Line (HELOC)"
    ])
    risk_levels: List[str] = field(default_factory=lambda: [
        "Conservative", "Moderate", "Aggressive"
    ])
    min_investment_tiers: List[float] = field(default_factory=lambda: [
        0, 500, 1000, 5000, 10000, 25000, 50000, 100000
    ])


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_version: str = "v1"
    rate_limit: int = 100  # requests per minute


@dataclass
class AppConfig:
    """Master application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    product: ProductConfig = field(default_factory=ProductConfig)
    api: APIConfig = field(default_factory=APIConfig)
    project_name: str = "Deep Learning Recommendation System for Financial Services"
    version: str = "1.0.0"


def get_config() -> AppConfig:
    """Get application configuration singleton."""
    return AppConfig()
