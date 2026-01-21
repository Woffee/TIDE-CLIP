
# Multimodal TCR–Peptide Binding (Seq + Graph + CLIP Alignment)

This repository trains a **multimodal** model for **TCRβ CDR3–peptide binding prediction** under the TCHard-style split protocol.  
Each entity (TCRβ and peptide) is represented by:

- **Sequence embedding (global)**: precomputed vector embedding for the full sequence
- **Residue graph**: variable-size graph with node features and edges
- **CLIP-style alignment**: contrastive loss aligning *sequence tower* and *graph tower* **within the same entity**
- **Binding head**: classification on fused (seq+graph) embeddings of the two entities

---

## Model overview

For each batch:

1. **Seq towers**
   - `tcr_seq -> z_tcr_seq`
   - `pep_seq -> z_pep_seq`

2. **Graph towers (Simple MPNN + mean pooling)**
   - `tcr_graph -> z_tcr_g`
   - `pep_graph -> z_pep_g`

3. **Intra-entity fusion**
   - `tcr = fuse([z_tcr_seq, z_tcr_g])`
   - `pep = fuse([z_pep_seq, z_pep_g])`

4. **Binding prediction**
   - features: `[tcr, pep, |tcr-pep|, tcr*pep]`
   - output: 2-class logits (bind vs non-bind)

5. **Training objective**
   - `Loss = lambda_bind * CE(bind) + lambda_clip * 0.5*(CLIP(tcr_seq,tcr_graph)+CLIP(pep_seq,pep_graph))`

---

## Requirements

- Python 3.9+
- PyTorch (CUDA optional)
- NumPy, Pandas, scikit-learn

Example install:
```bash
pip install torch numpy pandas scikit-learn
````

---

## Expected project structure

### 1) Dataset CSVs

The script expects train/test CSVs at:

```
dataset/ds.hard-splits/pep+cdr3b/
  train/
    only-sampled-negs/
      train-<dataset_index>.csv
  test/
    only-sampled-negs/
      test-<dataset_index>.csv
```

Required columns in each CSV:

* `cdr3.beta` (TCRβ CDR3 sequence string)
* `antigen.epitope` (peptide string)
* `label` (0/1)

> Note: the code currently sets `val_df=None` and falls back to using the **test set as validation** for early stopping.

### 2) Precomputed embeddings + graphs

The script expects the following files under `embs/` (configurable via `--embedbase`):

```
embs/
  tcr_seq_dict.pkl
  peptide_seq_dict.pkl
  tcr_graph_dict.pkl
  peptide_graph_dict.pkl
```

#### Sequence dicts

* `tcr_seq_dict.pkl`: `Dict[str, np.ndarray]`

  * key: TCRβ string (`cdr3.beta`)
  * value: global embedding vector (float32), shape `[Dseq]`

* `peptide_seq_dict.pkl`: `Dict[str, np.ndarray]`

  * key: peptide string (`antigen.epitope`)
  * value: global embedding vector (float32), shape `[Dseq]`

#### Graph dicts

* `tcr_graph_dict.pkl`: `Dict[str, Dict[str, Any]]`
* `peptide_graph_dict.pkl`: `Dict[str, Dict[str, Any]]`

Each value must be a dict with at least:

```python
{
  "x": np.ndarray or torch.Tensor,        # shape [num_nodes, node_feat_dim]
  "edge_index": np.ndarray or torch.Tensor # shape [2, num_edges], int64, 0-indexed
}
```

Recommendations:

* `x` can be residue-level embeddings (e.g., ESM residue embeddings) or simpler features (one-hot / physicochemical).
* `edge_index` should include **both directions** for undirected edges.

---

## Training

### Single run

Run a single dataset split (default `--dataset_index 1`) with default hyperparameters:

```bash
python train.py \
  --train_base dataset/ds.hard-splits/pep+cdr3b \
  --embedbase embs \
  --mode only-sampled-negs \
  --results_dir multimodal_clip_binding_results \
  --dataset_index 1
```

Key outputs:

* Logs: `logs/<YYYYMMDD_HHMM>.log`
* Checkpoint: `multimodal_clip_binding_results/multimodal_only-sampled-negs/best_multimodal_<dataset_index>.pth`
* Metrics per run: `evaluation_<dataset_index>.csv`
* Aggregated: `summary.csv`

### Grid search (optional)

Enable lightweight grid search:

```bash
python train.py \
  --do_grid \
  --dataset_index 1
```

Outputs:

* Per-setting folder: `multimodal_clip_binding_results/grid_<param_str>/`
* All results: `multimodal_clip_binding_results/all_grid_results.csv`
* Best params: `multimodal_clip_binding_results/best_params.txt`

---

## Arguments

### Data paths

* `--train_base`: dataset root (default: `dataset/ds.hard-splits/pep+cdr3b`)
* `--embedbase`: embeddings/graphs root (default: `embs`)
* `--mode`: negative sampling / split mode (default: `only-sampled-negs`)
* `--results_dir`: output directory (default: `./multimodal_clip_binding_results`)

### Model

* `--proj_dim`: projection dim in CLIP space (default: 256)
* `--graph_hidden`: graph tower hidden dim (default: 256)
* `--graph_layers`: number of message passing layers (default: 2)
* `--dropout`: dropout rate (default: 0.2)

### Loss

* `--lambda_clip`: weight of CLIP alignment loss (default: 0.2)
* `--lambda_bind`: weight of binding CE loss (default: 1.0)
* `--temperature`: CLIP temperature (default: 0.07)

### Training

* `--batch_size` (default: 64)
* `--learning_rate` (default: 5e-4)
* `--num_epochs` (default: 60)
* `--patience` early stopping patience on val AUROC (default: 20)
* `--weight_decay` (default: 1e-3)
* `--optimizer` in `{adam, sgd}` (default: adam)
* `--dataset_index` which dataset split to run (default: 1)

---

## Evaluation metrics

The evaluation computes:

* Loss (weighted CrossEntropy)
* AUROC
* AUPR
* Accuracy / Precision / Recall / F1 (threshold = 0.5)

Metrics are written to:

* `evaluation_<dataset_index>.csv` (per run)
* `summary.csv` (aggregated for single-run mode)

---

## Reproducibility notes

The script sets a fixed seed (`seed=12`) and configures PyTorch for deterministic behavior:

* `torch.backends.cudnn.deterministic = True`
* `torch.backends.cudnn.benchmark = False`

---


## Output folders

After a successful run, the structure typically looks like:

```
logs/
  20260121_0230.log

multimodal_clip_binding_results/
  multimodal_only-sampled-negs/
    best_multimodal_1.pth
    evaluation_1.csv
    summary.csv
```
