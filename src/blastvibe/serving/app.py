"""
Production FastAPI Serving Layer for BlastVibe-GNN Inference Engine.
"""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from blastvibe.data.loader import compute_scaled_distance
from blastvibe.models.predictor import BlastVibePredictor
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.serving")

DEFAULT_MODEL_PATH = Path("artifacts/models/best_gat_model.pt")

app = FastAPI(
    title="BlastVibe-GNN Inference API",
    description=(
        "Production REST service serving Inductive Graph Attention Networks for "
        "blast-induced ground vibration (PPV) prediction."
    ),
    version="1.0.0",
)

_predictor_instance: BlastVibePredictor | None = None


class BlastRequest(BaseModel):
    """Input payload for single blast ground vibration prediction."""

    dis: float = Field(..., ge=0.0, description="Distance from blast face in meters (must be >= 0.0)")
    charge: float = Field(..., gt=0.0, description="Maximum explosive charge per delay in kilograms (must be > 0.0)")
    sd: float | None = Field(default=None, gt=0.0, description="Optional precalculated Scaled Distance (m/kg^0.5)")


class PredictionOutput(BaseModel):
    """Output prediction payload containing vibration prediction and physics metadata."""

    predicted_ppv: float = Field(..., ge=0.0, description="Predicted Peak Particle Velocity in mm/s (>= 0.0)")
    scaled_distance: float = Field(..., gt=0.0, description="USBM Scaled Distance in m/kg^0.5 (> 0.0)")
    unit: str = Field(default="mm/s", description="Measurement unit for ground vibration")


class BatchBlastRequest(BaseModel):
    """Batch input payload containing multiple blast monitoring queries."""

    samples: list[BlastRequest] = Field(..., min_length=1, description="List of blast monitoring queries")


class BatchPredictionOutput(BaseModel):
    """Batch output prediction payload."""

    predictions: list[PredictionOutput] = Field(..., description="List of vibration predictions")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


def get_model_path() -> Path:
    env_path = os.getenv("BLASTVIBE_MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()
    return DEFAULT_MODEL_PATH.resolve()


def get_predictor() -> BlastVibePredictor:
    """Retrieves or lazily initializes the BlastVibePredictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        model_path = get_model_path()
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model checkpoint not found at: '{model_path}'. "
                    "Run 'blastvibe-train' or 'python main.py' to generate production model artifacts."
                ),
            )
        try:
            logger.info(f"Initializing BlastVibePredictor singleton from: {model_path}")
            _predictor_instance = BlastVibePredictor(bundle_path=model_path, device="cpu")
        except Exception as exc:
            logger.error(f"Failed to load BlastVibePredictor from {model_path}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize model predictor: {exc!s}",
            ) from exc
    return _predictor_instance


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Verifies service operational health and model checkpoint availability."""
    model_path = get_model_path()
    return HealthResponse(
        status="healthy",
        model_loaded=_predictor_instance is not None or model_path.exists(),
        model_path=str(model_path),
    )


@app.get("/model-info", tags=["Metadata"])
def get_model_info() -> dict[str, Any]:
    """Returns architecture metadata and hyperparameter specifications."""
    predictor = get_predictor()
    return {
        "hyperparameters": predictor.hyperparameters,
        "feature_names": predictor.feature_names,
        "target_name": predictor.target_name,
        "reference_nodes": predictor.x_train_ref.size(0),
        "reference_edges": predictor.train_edge_index.size(1),
        "k_neighbors": predictor.k_neighbors,
        "device": str(predictor.device),
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_vibration(request: BlastRequest) -> PredictionOutput:
    """
    Executes single-shot blast vibration prediction using inductive GAT inference.
    """
    predictor = get_predictor()

    try:
        sd_val = request.sd if request.sd is not None else float(compute_scaled_distance(request.dis, request.charge))
        ppv_pred = predictor.predict_single(dis=request.dis, charge=request.charge, sd=sd_val)
        return PredictionOutput(
            predicted_ppv=round(float(ppv_pred), 3),
            scaled_distance=round(float(sd_val), 3),
            unit="mm/s",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Prediction failed for DIS={request.dis}, CHARGE={request.charge}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {exc!s}",
        ) from exc


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Inference"])
def predict_batch(request: BatchBlastRequest) -> BatchPredictionOutput:
    """
    Executes batch blast vibration prediction across multiple sensor queries.
    """
    predictor = get_predictor()

    try:
        raw_list = []
        sd_list = []
        for sample in request.samples:
            sd = sample.sd if sample.sd is not None else float(compute_scaled_distance(sample.dis, sample.charge))
            raw_list.append([sample.dis, sample.charge, sd])
            sd_list.append(sd)

        predictions = predictor.predict(raw_list)

        outputs = [
            PredictionOutput(
                predicted_ppv=round(float(pred), 3),
                scaled_distance=round(float(sd), 3),
                unit="mm/s",
            )
            for pred, sd in zip(predictions, sd_list)
        ]
        return BatchPredictionOutput(predictions=outputs)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Batch prediction failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {exc!s}",
        ) from exc
