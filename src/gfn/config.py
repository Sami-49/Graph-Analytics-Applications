"""Configuration module: load and parse config.yaml."""
from dataclasses import dataclass, field, asdict
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    root: str
    max_events: Optional[int] = None
    min_nodes: int = 2


@dataclass
class ModelConfig:
    logistic_regression: Dict[str, Any]
    random_forest: Dict[str, Any]
    gradient_boosting: Dict[str, Any]


@dataclass
class EvaluationConfig:
    random_state: int = 42
    train_test_split: float = 0.8
    n_folds: int = 5
    n_bootstrap: int = 1000
    ci_alpha: float = 0.05


@dataclass
class CASDIConfig:
    enabled: bool = True
    alpha: float = 0.30
    beta: float = 0.20
    gamma: float = 0.20
    delta: float = 0.15
    epsilon: float = 0.15

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"CASDI weights must sum to 1.0; got {total}")


@dataclass
class OutputConfig:
    root: str = "./outputs"
    save_features: bool = True
    save_plots: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    dataset: DatasetConfig
    models: ModelConfig
    evaluation: EvaluationConfig
    casdi: CASDIConfig
    output: OutputConfig

    @staticmethod
    def from_yaml(yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return Config(
            dataset=DatasetConfig(**data["dataset"]),
            models=ModelConfig(**data["models"]),
            evaluation=EvaluationConfig(**data["evaluation"]),
            casdi=CASDIConfig(**data["casdi"]),
            output=OutputConfig(**data["output"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return asdict(self)
