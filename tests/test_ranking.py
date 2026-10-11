import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gnn_kg_target_id import FeatureTable, GeneGraph, TrainConfig, build_dataset, rank_novel_targets

N_GENES = 150
N_MODULE = 30


@pytest.fixture
def planted():
    """A 30-gene module with shifted expression and dense internal edges, plus random background."""
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(N_GENES)]
    module = genes[:N_MODULE]

    lfc = pd.DataFrame({"logFC": rng.normal(size=N_GENES)}, index=genes)
    lfc.loc[module, "logFC"] += 2.0
    tissue = pd.DataFrame(rng.integers(0, 2, size=(N_GENES, 5)), index=genes)

    adj = sparse.random(N_GENES, N_GENES, density=0.02, random_state=0, data_rvs=lambda n: np.ones(n)).tolil()
    adj[:N_MODULE, :N_MODULE] = (rng.random((N_MODULE, N_MODULE)) < 0.4).astype(float)
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)

    tables = {"lfc": FeatureTable(lfc), "tissue": FeatureTable(tissue, kind="indicator")}
    return genes, module, tables, {"ppi": GeneGraph(adj, genes)}


def test_hidden_module_members_rank_above_background(planted):
    genes, module, tables, graphs = planted
    known, hidden = module[:20], module[20:]

    ranked = rank_novel_targets(genes, known, tables, graphs, TrainConfig(n_epochs=300, seed=0))

    assert set(ranked.gene_id).isdisjoint(known)
    assert ranked.prediction_score.between(0, 1).all()
    rank = {g: i for i, g in enumerate(ranked.gene_id)}
    background_median = np.median([rank[g] for g in genes[N_MODULE:]])
    assert np.mean([rank[g] for g in hidden]) < background_median


def test_graph_aligns_by_gene_name_and_applies_threshold(planted):
    genes, _, tables, _ = planted
    order = ["G3", "G1", "G2", "G0"]  # differs from the dataset's sorted node order
    adj = sparse.csr_matrix(
        np.array(
            [
                [0, 0.9, 0, 0.2],  # G3 -> G1 kept, G3 -> G0 below threshold
                [0, 0, 0.9, 0],  # G1 -> G2 kept
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )
    cfg = TrainConfig(edge_weight_threshold=0.5)
    ds = build_dataset(genes[:4], ["G0"], tables, {"g": GeneGraph(adj, order)}, cfg)

    node = {g: i for i, g in enumerate(ds.genes)}
    edges = set(map(tuple, ds.graph.edge_index.T.tolist()))
    assert edges == {(node["G3"], node["G1"]), (node["G1"], node["G2"])}
