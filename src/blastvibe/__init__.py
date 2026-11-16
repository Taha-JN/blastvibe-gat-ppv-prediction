"""BlastVibe-GNN: Production-Grade Scientific Machine Learning Pipeline."""

from blastvibe.models.predictor import BlastVibePredictor
from blastvibe.serving.app import app

__version__ = "1.0.0"
__all__ = ["BlastVibePredictor", "app"]
