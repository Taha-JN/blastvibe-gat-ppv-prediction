import pytest
from pydantic import ValidationError

from blastvibe.serving.app import (
    BatchBlastRequest,
    BatchPredictionOutput,
    BlastRequest,
    PredictionOutput,
    get_model_info,
    health_check,
    predict_batch,
    predict_vibration,
)


def test_blast_request_pydantic_validation():
    # Valid request
    req = BlastRequest(dis=350.0, charge=1500.0)
    assert req.dis == 350.0
    assert req.charge == 1500.0
    assert req.sd is None

    # Invalid negative distance (dis < 0)
    with pytest.raises(ValidationError):
        BlastRequest(dis=-10.0, charge=1500.0)

    # Invalid non-positive charge (charge <= 0)
    with pytest.raises(ValidationError):
        BlastRequest(dis=350.0, charge=0.0)

    with pytest.raises(ValidationError):
        BlastRequest(dis=350.0, charge=-50.0)

    # Invalid negative SD
    with pytest.raises(ValidationError):
        BlastRequest(dis=350.0, charge=100.0, sd=-5.0)


def test_prediction_output_schema():
    out = PredictionOutput(predicted_ppv=85.25, scaled_distance=12.5, unit="mm/s")
    assert out.predicted_ppv == 85.25
    assert out.scaled_distance == 12.5
    assert out.unit == "mm/s"

    with pytest.raises(ValidationError):
        PredictionOutput(predicted_ppv=-1.0, scaled_distance=10.0)


def test_health_check_endpoint():
    res = health_check()
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert "best_gat_model.pt" in res.model_path


def test_predict_single_endpoint():
    req = BlastRequest(dis=400.0, charge=1850.0)
    res = predict_vibration(req)
    assert isinstance(res, PredictionOutput)
    assert res.predicted_ppv > 0.0
    assert res.scaled_distance > 0.0
    assert res.unit == "mm/s"


def test_predict_single_with_precomputed_sd():
    req = BlastRequest(dis=400.0, charge=1850.0, sd=9.3)
    res = predict_vibration(req)
    assert isinstance(res, PredictionOutput)
    assert res.predicted_ppv > 0.0
    assert res.scaled_distance == 9.3


def test_predict_batch_endpoint():
    batch_req = BatchBlastRequest(
        samples=[
            BlastRequest(dis=300.0, charge=1850.0),
            BlastRequest(dis=500.0, charge=1950.0),
        ]
    )
    res = predict_batch(batch_req)
    assert isinstance(res, BatchPredictionOutput)
    assert len(res.predictions) == 2
    for pred in res.predictions:
        assert pred.predicted_ppv > 0.0
        assert pred.scaled_distance > 0.0
        assert pred.unit == "mm/s"


def test_get_model_info_endpoint():
    info = get_model_info()
    assert "hyperparameters" in info
    assert "feature_names" in info
    assert info["reference_nodes"] > 0
    assert info["k_neighbors"] > 0
