from dataclasses import dataclass
from typing import Literal

GnnType = Literal["GCN", "SAGE", "GAT"]


@dataclass(frozen=True)
class TrainConfig:
    gnn_type: GnnType = "SAGE"
    hidden_dim: int = 8
    dropout: float = 0.3
    learning_rate: float = 1e-3
    n_epochs: int = 1000

    # Fraction of known positives hidden from the supervised loss each epoch.
    mask_ratio: float = 0.3
    # Weight on -log p(masked positives).
    lambda_recovery: float = 0.5
    # Weight on mean(p) over all genes; keeps the number of predicted positives small.
    lambda_sparsity: float = 1.0
    positive_threshold: float = 0.5

    # Edges with weight at or below this value are dropped when a graph is loaded.
    edge_weight_threshold: float = 0.5

    fill_missing: Literal["zeros", "median"] = "zeros"
    seed: int | None = None
    log_every: int = 0  # 0 disables progress logging

    def __post_init__(self) -> None:
        if self.gnn_type == "GAT" and self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for GAT (4 attention heads)")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
