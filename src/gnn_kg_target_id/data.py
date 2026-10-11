from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import Data

from gnn_kg_target_id.config import TrainConfig


@dataclass(frozen=True)
class FeatureTable:
    """Per-gene features indexed by gene id.

    `continuous` columns are z-scored. `indicator` columns (0/1 membership matrices such as
    tissue, gene type or subcellular location) are divided by log(1 + n_columns) so wide
    matrices do not dominate the feature space.
    """

    df: pd.DataFrame
    kind: Literal["continuous", "indicator"] = "continuous"


@dataclass(frozen=True)
class GeneGraph:
    """Weighted gene-gene adjacency with the gene id of every row and column."""

    adjacency: sparse.spmatrix
    genes: list[str]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "GeneGraph":
        return cls(sparse.csr_matrix(df.to_numpy(dtype=float)), df.index.astype(str).tolist())


@dataclass(frozen=True)
class TargetDataset:
    graph: Data
    genes: list[str]
    feature_names: list[str]


def _prepare_features(table: FeatureTable, genes: list[str], cfg: TrainConfig) -> pd.DataFrame:
    df = table.df.loc[table.df.index.isin(genes)]
    if df.empty:
        raise ValueError("No rows left after restricting a feature table to the gene universe")
    df = df.astype({c: float for c in df.columns if df[c].dtype == bool})
    df = df.select_dtypes("number")
    if df.shape[1] == 0:
        raise ValueError("Feature table has no numeric columns")

    df = df.fillna(0.0) if cfg.fill_missing == "zeros" else df.fillna(df.median())
    if table.kind == "continuous":
        std = df.std(ddof=0).replace(0.0, 1.0)
        return (df - df.mean()) / std
    return df / np.log1p(df.shape[1])


def _edge_index(graph: GeneGraph, node_of: Mapping[str, int], threshold: float) -> torch.Tensor:
    # Map graph rows to node ids by gene name so graphs built over a different gene order or
    # a superset of genes line up with the feature matrix.
    node_ids = np.array([node_of.get(g, -1) for g in graph.genes])
    coo = sparse.coo_matrix(graph.adjacency)
    src, dst = node_ids[coo.row], node_ids[coo.col]
    keep = (coo.data > threshold) & (src >= 0) & (dst >= 0)
    return torch.from_numpy(np.stack([src[keep], dst[keep]])).long()


def build_dataset(
    genes: Collection[str],
    known_genes: Collection[str],
    feature_tables: Mapping[str, FeatureTable],
    graphs: Mapping[str, GeneGraph],
    cfg: TrainConfig,
) -> TargetDataset:
    """Assemble node features, labels and the combined edge set for the gene universe.

    Labels are positive-unlabeled: genes in `known_genes` are 1, every other gene is 0 but is
    treated as unlabeled by the loss. The label is never an input feature.
    """
    universe = sorted(set(genes))
    positives = set(known_genes) & set(universe)
    if not positives:
        raise ValueError("None of the known genes are in the gene universe")
    if not graphs:
        raise ValueError("At least one gene-gene graph is required")

    columns = []
    for name, table in feature_tables.items():
        df = _prepare_features(table, universe, cfg)
        columns.append(df.add_prefix(f"{name}:"))
    # Genes missing from a table get 0 for that table's features.
    features = pd.concat(columns, axis=1, join="outer").reindex(universe).fillna(0.0)

    node_of = {g: i for i, g in enumerate(features.index)}
    # Edges present in several sources are kept once per source, so neighbour aggregation
    # weights an edge by how many knowledge sources support it.
    edge_index = torch.cat(
        [_edge_index(g, node_of, cfg.edge_weight_threshold) for g in graphs.values()], dim=1
    )

    y = torch.tensor([float(g in positives) for g in features.index])
    x = torch.tensor(features.to_numpy(dtype=np.float32))
    return TargetDataset(
        graph=Data(x=x, edge_index=edge_index, y=y),
        genes=features.index.tolist(),
        feature_names=features.columns.tolist(),
    )
