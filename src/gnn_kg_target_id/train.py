import torch
import torch.nn as nn

from gnn_kg_target_id.config import TrainConfig
from gnn_kg_target_id.data import TargetDataset
from gnn_kg_target_id.model import TargetGNN


def semi_supervised_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    masked: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Positive-unlabeled loss with held-out positives.

    - fit: squared error on positives that are not masked this epoch. Unlabeled genes carry
      no fit term because most of them are unlabeled positives rather than true negatives.
    - recovery: -log p on the masked positives, which teaches the model to score a known
      target highly from its features and neighbours alone.
    - sparsity: mean(p) over all genes, which stops the model from scoring everything high.
    """
    positive = y > cfg.positive_threshold
    visible = positive & ~masked
    fit = ((y[visible] - pred[visible]) ** 2).mean() if visible.any() else pred.new_zeros(())
    recovery = -torch.log(pred[masked] + 1e-8).mean()
    sparsity = pred.mean()
    total = fit + cfg.lambda_recovery * recovery + cfg.lambda_sparsity * sparsity
    return total, {"fit": fit.item(), "recovery": recovery.item(), "sparsity": sparsity.item()}


def _sample_mask(y: torch.Tensor, cfg: TrainConfig, rng: torch.Generator) -> torch.Tensor:
    positives = torch.where(y > cfg.positive_threshold)[0]
    n_mask = max(1, int(len(positives) * cfg.mask_ratio))
    chosen = positives[torch.randperm(len(positives), generator=rng)[:n_mask]]
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[chosen] = True
    return mask


def train(dataset: TargetDataset, cfg: TrainConfig) -> TargetGNN:
    rng = torch.Generator()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        rng.manual_seed(cfg.seed)

    data = dataset.graph
    model = TargetGNN(data.x.shape[1], cfg.hidden_dim, cfg.dropout, cfg.gnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for epoch in range(cfg.n_epochs):
        masked = _sample_mask(data.y, cfg, rng)
        pred = model(data)
        loss, parts = semi_supervised_loss(pred, data.y, masked, cfg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cfg.log_every and epoch % cfg.log_every == 0:
            recovered = (pred[masked] > cfg.positive_threshold).float().mean().item()
            print(f"epoch {epoch}: loss={loss.item():.4f} {parts} masked_recovery={recovered:.2%}")
    return model


@torch.no_grad()
def predict(model: nn.Module, dataset: TargetDataset) -> torch.Tensor:
    model.eval()
    return model(dataset.graph)
