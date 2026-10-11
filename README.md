# gnn-kg-for-novel-target-id

A graph neural network that ranks candidate genes as novel therapeutic targets for a disease. Each
gene is a node. Node features come from omics data and curated gene annotations, edges come from
knowledge-graph relationships, and the labels are genes already known to be associated with the
disease.

## Method

1. **Gene universe.** Candidate genes (usually the differentially expressed genes) plus the known
   disease genes.
2. **Node features.** One row per gene, built from the feature tables below. Continuous columns are
   z-scored. 0/1 membership matrices are divided by `log(1 + n_columns)`. Genes absent from a
   table get zeros for that table.
3. **Graph.** Gene-gene adjacency matrices from several sources are loaded by gene name, thresholded
   on edge weight, and concatenated into one edge set. An edge supported by several sources appears
   once per source, so neighbour aggregation weights it by that support.
4. **Labels.** Known disease genes are 1. All other genes are unlabeled, not negative. Labels are
   never node features.
5. **Model.** Two message-passing layers (`SAGE` by default, or `GCN` / `GAT`) and a two-layer MLP
   head with a sigmoid output (`src/gnn_kg_target_id/model.py`).
6. **Loss** (`src/gnn_kg_target_id/train.py`). Every epoch a random fraction (`mask_ratio`) of the
   known positives is held out of the fit term. The loss is the sum of:
   - squared error on the visible positives,
   - `lambda_recovery * -log p` on the held-out positives,
   - `lambda_sparsity * mean(p)` over all genes.

   The fit term has no negatives. The recovery term makes the model score a known target highly
   from features and neighbours alone, and the sparsity term keeps the number of high scores small.
7. **Ranking** (`src/gnn_kg_target_id/ranking.py`). After training, all genes are scored, known genes
   are removed, and the remaining candidates are min-max scaled to [0, 1] and sorted.

## Data sources

The model is agnostic to where the tables come from. These are the inputs it was built around. Each
is a table or matrix keyed by gene id.

| Input | Role | Form |
| --- | --- | --- |
| Differential expression (log fold change and a significance flag) for the disease contrast | Candidate pool and a continuous feature | Feature table, continuous |
| Network driver or centrality score per gene from a protein-protein interaction (PPI) analysis | Continuous feature | Feature table, continuous |
| Gene-disease associations from a knowledge graph | Positive labels | Gene list |
| Pathway co-membership between genes | Graph edges | Sparse adjacency |
| Transcription factor to target regulation | Graph edges | Sparse adjacency |
| PPI interactions | Graph edges | Sparse adjacency |
| Gene expression by tissue and cell type | Gene annotation | Feature table, indicator |
| Gene type (for example protein-coding, lncRNA) | Gene annotation | Feature table, indicator |
| Subcellular location (Gene Ontology cellular component terms) | Gene annotation | Feature table, indicator |

For small-molecule, antibody-drug-conjugate and PROTAC programs, only upregulated genes were kept
in the candidate pool. That filter is applied before the data reaches this package.

## Usage

```python
import pandas as pd
from scipy import sparse

from gnn_kg_target_id import FeatureTable, GeneGraph, TrainConfig, rank_novel_targets

lfc = pd.read_csv("lfc.csv", index_col="gene")  # columns: logFC, ...
tissue = pd.read_csv("gene_tissue.csv", index_col="gene")  # 0/1 columns
ppi = GeneGraph(sparse.load_npz("ppi.npz"), genes=open("ppi_genes.txt").read().split())

ranked = rank_novel_targets(
    candidate_genes=lfc.index[lfc["is_de"]],
    known_genes=known_disease_genes,
    feature_tables={"lfc": FeatureTable(lfc[["logFC"]]), "tissue": FeatureTable(tissue, kind="indicator")},
    graphs={"ppi": ppi},
    cfg=TrainConfig(gnn_type="SAGE", n_epochs=1000, seed=0),
)
```

Returns a frame with `gene_id`, `prediction_score` and `known_target`, best first.

Defaults match the values the original pipelines used: `SAGE`, hidden size 8, dropout 0.3, learning
rate 1e-3, `mask_ratio` 0.3, `lambda_recovery` 0.5, `lambda_sparsity` 1.0. They used 1000 epochs for
fewer than 1000 genes and 100 epochs for larger universes.

## Development

```
uv sync
uv run pytest
```
