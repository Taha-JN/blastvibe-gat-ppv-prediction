"""
BlastVibe-GNN Configuration Schema with Strict Pydantic Validation.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_path: str = Field(default="data/data.csv", description="Path to raw dataset (CSV or Excel)")
    feature_cols: List[str] = Field(
        default_factory=lambda: ["DIS", "CHARGE", "SD"],
        description="Feature column names used as inputs",
    )
    target_col: str = Field(default="SUM", description="Target vibration regression column name")


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_splits: int = Field(default=5, ge=2, description="Number of cross-validation splits (must be >= 2)")
    shuffle: bool = Field(default=True, description="Whether to shuffle data before splitting")
    random_state: int = Field(default=42, description="Deterministic random state for reproducibility")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=300, gt=0, description="Maximum training epochs (must be > 0)")
    patience: int = Field(default=30, ge=0, description="Early stopping patience epochs")
    eval_interval: int = Field(default=10, gt=0, description="Evaluation interval in epochs")
    device: str = Field(default="auto", description="Compute device: 'auto', 'cuda', or 'cpu'")


class GridSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_neighbors: List[int] = Field(default_factory=lambda: [4, 8, 12], description="K-NN neighbor values")
    hidden_dim: List[int] = Field(default_factory=lambda: [32, 64], description="Hidden layer dimensions")
    lr: List[float] = Field(default_factory=lambda: [0.01, 0.001], description="Learning rates")
    weight_decay: List[float] = Field(default_factory=lambda: [0.0001, 0.0], description="L2 weight decay")
    heads: List[int] = Field(default_factory=lambda: [4], description="Number of multi-head attention heads")
    dropout: List[float] = Field(default_factory=lambda: [0.2], description="Dropout probability")


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(default="artifacts", description="Directory for generated metrics and plots")
    model_dir: str = Field(default="artifacts/models", description="Directory for serialized models")
    metrics_file: str = Field(default="artifacts/grid_search_results.csv", description="CSV metrics output path")
    best_model_file: str = Field(default="artifacts/models/best_gat_model.pt", description="Best model checkpoint path")


class PipelineConfig(BaseModel):
    """Master pipeline configuration schema with fail-fast strict validation."""
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    grid_search: GridSearchConfig = Field(default_factory=GridSearchConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Converts validated schema into standard dictionary for downstream pipeline consumption."""
        return self.model_dump()
