from collections.abc import Collection, Mapping

import pandas as pd

from gnn_kg_target_id.config import TrainConfig
from gnn_kg_target_id.data import FeatureTable, GeneGraph, build_dataset
from gnn_kg_target_id.train import predict, train


def rank_novel_targets(
    candidate_genes: Collection[str],
    known_genes: Collection[str],
    feature_tables: Mapping[str, FeatureTable],
    graphs: Mapping[str, GeneGraph],
    cfg: TrainConfig | None = None,
) -> pd.DataFrame:
    """Train the GNN and rank candidate genes that are not already known targets.

    `candidate_genes` is the pool to rank, typically the differentially expressed genes. The
    gene universe the GNN sees is the candidates plus the known genes, so known targets
    contribute labels and neighbourhood context.

    Returns a frame with `gene_id`, `prediction_score` (min-max scaled to [0, 1] across the
    ranked genes) and `known_target`, sorted by descending score. Known genes are excluded.
    """
    cfg = cfg or TrainConfig()
    dataset = build_dataset({*candidate_genes, *known_genes}, known_genes, feature_tables, graphs, cfg)
    model = train(dataset, cfg)
    scores = pd.Series(predict(model, dataset).numpy(), index=dataset.genes, name="prediction_score")

    novel = scores[scores.index.isin(set(candidate_genes) - set(known_genes))]
    if novel.empty:
        raise ValueError("No candidate genes remain after excluding known targets")
    span = novel.max() - novel.min()
    scaled = (novel - novel.min()) / span if span > 0 else novel * 0.0
    out = scaled.sort_values(ascending=False).rename_axis("gene_id").reset_index()
    out["known_target"] = False
    return out
