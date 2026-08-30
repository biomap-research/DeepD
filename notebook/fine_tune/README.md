# GUE H3 classification with DeepD embeddings

This package provides a runnable example of the downstream GUE classification
workflow using the **H3 task** as the demonstration task.

DeepD embeddings can be obtained two ways:

1. **Live inference API (from scratch)** — the notebook's section 2 submits one
   sequence from the H3 training split to the online DeepD inference API
   (`embedding` task) and prints the returned `[L, 2048]` token embedding
   (shape and a slice of the values). This demonstrates that any sequence can be
   embedded on demand, with no precomputed package required.
2. **Downloaded precomputed embeddings** — from section 3 onward the notebook
   uses the H3 train/validation/test embeddings hosted on Hugging Face, so the
   classification walkthrough never needs to re-run inference.

The H3 sequence/label splits and their precomputed DeepD embeddings are hosted
on the Hugging Face dataset
[`biomap-research/DeepD`](https://huggingface.co/datasets/biomap-research/DeepD)
under the `fine_tune/H3/` prefix. The notebook downloads that subtree on first run when
the files are not already present locally.

## Repository layout

```text
  gue_h3_embedding_classification.ipynb

embedding_classifier/
  src/                         # reusable backend implementation
  configs/H3_example.yaml
  tests/smoke_gue_classification.py
```

The `src/` backend is kept separate from the notebook so the notebook remains a
readable scientific workflow rather than duplicating the full training framework.

## Expected Hugging Face layout

The H3 portion of [`biomap-research/DeepD`](https://huggingface.co/datasets/biomap-research/DeepD)
follows this layout:

```text
fine_tune/
└── H3/
    ├── H3_train.csv
    ├── H3_valid.csv
    ├── H3_test.csv
    ├── H3_train_embeddings/
    ├── H3_valid_embeddings/
    └── H3_test_embeddings/
```

The three CSV files should contain at least:

```text
unique_id,label
sample_001,0
sample_002,1
```

By default, each embedding directory is expected to contain files named:

```text
embedding_{unique_id}.pt
```

and each `.pt` file may contain a dictionary with the key
`layernorm_embedding`.

If the final Hugging Face upload uses different embedding directory names or a
different filename pattern, only the configuration block near the top of the notebook
(or the corresponding fields in `H3_example.yaml`) needs to be changed. The backend
code does not need to be modified.

## H3 classification workflow

```text
DeepD token embeddings
        (section 2: computed live with the inference API;
         from section 3 on: downloaded precomputed .pt files)
        -> remove two configured prefix tokens
        -> padded batching + valid-token mask
        -> masked mean pooling
        -> MLP: 2048 -> 512 -> 128 -> 2
        -> cross-entropy loss
        -> validation MCC for checkpoint selection / early stopping
        -> test MCC and accuracy
```

The H3 example uses the following training settings:

- input dimension: 2048
- hidden dimensions: 512 and 128
- loss: cross entropy
- optimizer: Adam
- learning rate: 1e-3
- batch size: 512
- validation interval: every 20 steps
- early stopping: validation MCC, patience 10
- maximum training steps: 10,000
- random seed: 42

## Using the Hugging Face interface

Open `gue_h3_embedding_classification.ipynb` in this directory. The default
repository is already set:

```python
HF_REPO_ID = "biomap-research/DeepD"
HF_PREFIX = "fine_tune"
HF_SUBDIR = "H3"
```

The notebook downloads only the `fine_tune/H3/**` portion of that dataset. Override
`HF_REPO_ID` only if you host a fork. For a private repository, authenticate
with Hugging Face or provide an `HF_TOKEN` environment variable.

A manually prepared local `data/H3/` directory with the same layout skips the
download. The Hugging Face dataset will be populated in a follow-up upload;
until then, place the H3 files locally or expect the download step to fail.

## Engineering smoke test

The smoke test uses synthetic embeddings and labels **only for software validation**.
It is not part of the scientific H3 evaluation and its MCC/accuracy values must not be
reported as benchmark results.

```bash
cd embedding_classifier
python tests/smoke_gue_classification.py
```

The smoke test is optional for publication, but is useful for checking installation,
data loading, pooling, training, and metric computation in a clean environment.
