"""Shared supervised-learning contract and model plugins."""

from src.supervised.data import SplitData, load_dataset, load_splits
from src.supervised.predictor import Predictor, fit_named_predictor, fit_predictor
from src.supervised.registry import available_models, build_model, register_model

__all__ = [
    "SplitData",
    "available_models",
    "build_model",
    "fit_named_predictor",
    "fit_predictor",
    "load_dataset",
    "load_splits",
    "Predictor",
    "register_model",
]
