"""
REST API for the Deep Learning Recommendation System
Provides endpoints for recommendations, user profiles, and model metrics.
"""

import os
import sys
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.settings import get_config
from src.data.generator import FinancialDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.neumf import NeuMF
from src.models.trainer import Trainer
from src.models.recommender import RecommendationEngine
from src.models.metrics import RecommendationMetrics

# ─── Initialize App ──────────────────────────────────────────────
config = get_config()
app = FastAPI(
    title=config.project_name,
    version=config.version,
    description="Deep Learning-powered financial product recommendation API using Neural Collaborative Filtering (NeuMF)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────
engine: Optional[RecommendationEngine] = None
model: Optional[NeuMF] = None
training_metrics: Dict = {}


# ─── Pydantic Models ─────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10
    category_filter: Optional[str] = None
    risk_filter: Optional[str] = None
    exclude_owned: bool = True


class TrainRequest(BaseModel):
    num_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    num_users: int = 5000
    num_products: int = 200


# ─── Startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup."""
    global engine, model, training_metrics
    try:
        print("Initializing recommendation system...")
        engine, model, training_metrics = _initialize_system()
        print("System ready!")
    except Exception as e:
        print(f"Startup initialization deferred: {e}")


def _initialize_system(
    num_users=2000, num_products=100, num_epochs=5
):
    """Generate data, train model, and return engine."""
    cfg = get_config()
    cfg.data.num_users = num_users
    cfg.data.num_products = num_products
    cfg.model.num_epochs = num_epochs
    cfg.model.batch_size = 256

    # Generate data
    gen = FinancialDataGenerator(cfg)
    users, products, interactions = gen.generate_dataset()

    # Preprocess
    preprocessor = DataPreprocessor(cfg)
    processed = preprocessor.preprocess(users, products, interactions)
    loaders = preprocessor.create_dataloaders(processed)

    # Build model
    mdl = NeuMF(
        num_users=processed["num_users"],
        num_products=processed["num_products"],
        gmf_embedding_dim=cfg.model.embedding_dim,
        mlp_embedding_dim=cfg.model.embedding_dim,
        mlp_hidden_layers=cfg.model.mlp_layers,
        dropout_rate=cfg.model.dropout_rate,
    )

    # Train
    trainer = Trainer(mdl, cfg)
    history = trainer.fit(loaders["train"], loaders["val"], num_epochs)
    test_metrics = trainer.evaluate(loaders["test"])

    # Build engine
    eng = RecommendationEngine(mdl, users, products, interactions)

    metrics = {
        "training_history": {k: [round(v, 4) for v in vals] for k, vals in history.items()},
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "data_stats": {
            "num_users": len(users),
            "num_products": len(products),
            "num_interactions": len(interactions),
        },
    }

    return eng, mdl, metrics


# ─── API Endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "project": config.project_name,
        "version": config.version,
        "status": "running" if engine else "not initialized",
        "endpoints": [
            "/recommend", "/user/{user_id}/profile",
            "/products/trending", "/products/{product_id}/similar",
            "/metrics", "/train", "/health",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": engine is not None}


@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    """Get personalized product recommendations for a user."""
    if engine is None:
        raise HTTPException(503, "Model not initialized. POST /train first.")

    try:
        recs = engine.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            exclude_owned=request.exclude_owned,
            category_filter=request.category_filter,
            risk_filter=request.risk_filter,
        )
        return {"user_id": request.user_id, "recommendations": recs, "count": len(recs)}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/recommend/{user_id}")
async def recommend_get(
    user_id: int,
    top_k: int = Query(10, ge=1, le=50),
    category: Optional[str] = None,
    risk: Optional[str] = None,
):
    """GET endpoint for recommendations."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    recs = engine.recommend(
        user_id=user_id, top_k=top_k,
        category_filter=category, risk_filter=risk,
    )
    return {"user_id": user_id, "recommendations": recs, "count": len(recs)}


@app.get("/user/{user_id}/profile")
async def user_profile(user_id: int):
    """Get detailed user profile with financial attributes."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    profile = engine.get_user_profile(user_id)
    if "error" in profile:
        raise HTTPException(404, profile["error"])
    return profile


@app.get("/user/{user_id}/similar")
async def similar_users(user_id: int, top_k: int = Query(5, ge=1, le=20)):
    """Find users with similar financial profiles."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    return {"user_id": user_id, "similar_users": engine.get_similar_users(user_id, top_k)}


@app.get("/products/trending")
async def trending_products(top_k: int = Query(10, ge=1, le=50)):
    """Get trending financial products."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    return {"trending": engine.get_trending_products(top_k)}


@app.get("/products/{product_id}/similar")
async def similar_products(product_id: int, top_k: int = Query(5, ge=1, le=20)):
    """Find similar financial products."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    return {"product_id": product_id, "similar": engine.get_similar_products(product_id, top_k)}


@app.get("/products/categories")
async def product_categories():
    """List all product categories."""
    return {"categories": config.product.categories}


@app.get("/metrics")
async def get_metrics():
    """Get model training and evaluation metrics."""
    if not training_metrics:
        raise HTTPException(503, "No metrics available. Train the model first.")
    return training_metrics


@app.post("/train")
async def train_model(request: TrainRequest):
    """Retrain the recommendation model."""
    global engine, model, training_metrics
    try:
        engine, model, training_metrics = _initialize_system(
            num_users=request.num_users,
            num_products=request.num_products,
            num_epochs=request.num_epochs,
        )
        return {"status": "success", "metrics": training_metrics}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/data/stats")
async def data_stats():
    """Get dataset statistics."""
    if engine is None:
        raise HTTPException(503, "Model not initialized.")

    return {
        "num_users": len(engine.users_df),
        "num_products": len(engine.products_df),
        "num_interactions": len(engine.interactions_df),
        "product_categories": engine.products_df["category"].value_counts().to_dict(),
        "risk_distribution": engine.products_df["risk_level"].value_counts().to_dict(),
        "interaction_types": engine.interactions_df["interaction_type"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)
