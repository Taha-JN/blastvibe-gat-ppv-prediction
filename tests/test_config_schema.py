import pytest
from pydantic import ValidationError

from blastvibe.configs.schema import (
    CVConfig,
    PipelineConfig,
    TrainingConfig,
)


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.cv.n_splits == 5
    assert cfg.training.epochs == 300
    assert cfg.data.target_col == "SUM"
    assert "DIS" in cfg.data.feature_cols

    # Ensure to_dict / model_dump serializes accurately
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["cv"]["n_splits"] == 5


def test_pipeline_config_validation_constraints():
    # n_splits must be >= 2
    with pytest.raises(ValidationError):
        CVConfig(n_splits=1)

    # epochs must be > 0
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)

    with pytest.raises(ValidationError):
        TrainingConfig(epochs=-10)

    # extra unknown fields must be forbidden
    with pytest.raises(ValidationError):
        CVConfig(n_splits=5, unknown_field="invalid")


def test_pipeline_config_from_raw_dict():
    raw = {
        "data": {
            "raw_path": "custom/path.csv",
            "feature_cols": ["A", "B"],
            "target_col": "Y",
        },
        "cv": {
            "n_splits": 10,
            "shuffle": False,
            "random_state": 123,
        },
        "training": {
            "epochs": 150,
            "patience": 15,
            "eval_interval": 5,
            "device": "cpu",
        },
    }
    cfg = PipelineConfig(**raw)
    assert cfg.data.raw_path == "custom/path.csv"
    assert cfg.cv.n_splits == 10
    assert cfg.training.epochs == 150
